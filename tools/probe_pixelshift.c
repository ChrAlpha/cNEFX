/* Research harness for the observed Microsoft x64 ABI, not a native backend.
 * Build: x86_64-w64-mingw32-gcc -O2 -Wall -Wextra -o probe_pixelshift.exe tools/probe_pixelshift.c
 * Run under Wine in an isolated research prefix. Never pass valuable originals.
 */
#include <windows.h>
#include <stdint.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    union { char inline_data[16]; char *pointer; } storage;
    uint64_t length;
    uint64_t capacity;
} MsvcString;

_Static_assert(sizeof(MsvcString) == 32, "Unexpected MSVC string layout");

typedef bool (*Progress)(uint32_t, uint32_t, uint32_t);
typedef bool (*Decision)(uint32_t);
typedef bool (*Combine)(uint32_t, const char **, uint32_t, const char **,
                        MsvcString *, bool, Progress, Decision);

static uint32_t cancel_at = UINT32_MAX;
static bool cancel_latched = false;

static bool progress(uint32_t total, uint32_t index, uint32_t percent) {
    printf("{\"event\":\"progress\",\"total\":%u,\"index\":%u,\"percent\":%u}\n",
           total, index, percent);
    fflush(stdout);
    cancel_latched = cancel_latched || percent >= cancel_at;
    return cancel_latched;
}

static bool decision(uint32_t code) {
    printf("{\"event\":\"decision\",\"code\":%u}\n", code);
    fflush(stdout);
    return code == 1;
}

int main(int argc, char **argv) {
    if (argc < 5) {
        fprintf(stderr, "Usage: probe_pixelshift DLL MODE OUTPUT_COUNT OUTPUT [INPUT...]\n");
        return 2;
    }
    if (strcmp(argv[2], "normal") && strcmp(argv[2], "cancel") &&
        strcmp(argv[2], "cancel-80") && strcmp(argv[2], "cancel-95") &&
        strcmp(argv[2], "cancel-100") && strcmp(argv[2], "correct") &&
        strcmp(argv[2], "invalid-count")) {
        fprintf(stderr, "Unknown probe mode.\n");
        return 2;
    }
    char *end;
    unsigned long output_count = strtoul(argv[3], &end, 10);
    if (*end || (output_count != 1 && output_count != 2 &&
                 output_count != 4 && output_count != 8)) {
        fprintf(stderr, "Invalid output count.\n");
        return 2;
    }
    char names[8][4096];
    const char *outputs[8];
    for (unsigned long index = 0; index < output_count; ++index) {
        int size = output_count == 1
            ? snprintf(names[index], sizeof(names[index]), "%s", argv[4])
            : snprintf(names[index], sizeof(names[index]), "%s.part-%lu.nefx", argv[4], index + 1);
        if (size < 0 || size >= (int)sizeof(names[index])) {
            fprintf(stderr, "Output path is too long.\n");
            return 2;
        }
        outputs[index] = names[index];
        if (GetFileAttributesA(outputs[index]) != INVALID_FILE_ATTRIBUTES) {
            fprintf(stderr, "Output already exists; refusing to overwrite.\n");
            return 2;
        }
    }
    SetErrorMode(SEM_FAILCRITICALERRORS | SEM_NOGPFAULTERRORBOX);
    HMODULE module = LoadLibraryA(argv[1]);
    if (!module) {
        printf("{\"event\":\"load_failed\",\"win32_error\":%lu}\n", GetLastError());
        return 3;
    }
    FARPROC entry = GetProcAddress(module, "PixelShiftCombine");
    Combine combine;
    memcpy(&combine, &entry, sizeof(combine));
    if (!combine) {
        printf("{\"event\":\"export_missing\"}\n");
        return 4;
    }
    printf("{\"event\":\"loaded\"}\n");
    fflush(stdout);
    MsvcString temporary = { .storage.inline_data = "NX Ctudio probe",
                             .length = 15, .capacity = 15 };
    uint32_t count = (uint32_t)(argc - 5);
    if (strcmp(argv[2], "cancel") == 0) cancel_at = 0;
    if (strcmp(argv[2], "cancel-80") == 0) cancel_at = 80;
    if (strcmp(argv[2], "cancel-95") == 0) cancel_at = 95;
    if (strcmp(argv[2], "cancel-100") == 0) cancel_at = 100;
    if (strcmp(argv[2], "invalid-count") == 0) count = 3;
    bool result = combine(count, (const char **)&argv[5], (uint32_t)output_count, outputs,
                          &temporary, strcmp(argv[2], "correct") == 0,
                          progress, decision);
    printf("{\"event\":\"return\",\"success\":%s}\n", result ? "true" : "false");
    FreeLibrary(module);
    return result ? 0 : 1;
}
