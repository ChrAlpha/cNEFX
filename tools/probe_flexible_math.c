/* Research-only integer leaf oracle. The runner pins the Rome2.dll hash.
 * Synthetic input is a bounded little-endian record stream, not image data.
 */
#include <windows.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#define MAX_CASES 100000
#define MAX_NODES 4096

typedef int32_t (*Divide)(int32_t, int32_t);
typedef int32_t (*Clamp)(int32_t, int32_t, int32_t);
typedef uint16_t (*Interpolate)(int32_t, const void *);

int main(int argc, char **argv) {
    if (argc != 3) {
        fprintf(stderr, "Usage: probe_flexible_math Rome2.dll cases.bin\n");
        return 2;
    }
    FILE *stream = fopen(argv[2], "rb");
    char magic[8];
    uint32_t count;
    if (!stream) return 2;
    if (fread(magic, 1, 8, stream) != 8 || memcmp(magic, "NXFM001", 8) ||
        fread(&count, 4, 1, stream) != 1 || !count || count > MAX_CASES) {
        fclose(stream);
        return 2;
    }
    SetErrorMode(SEM_FAILCRITICALERRORS | SEM_NOGPFAULTERRORBOX);
    HMODULE module = LoadLibraryA(argv[1]);
    if (!module) {
        fprintf(stderr, "{\"event\":\"load_failed\",\"code\":%lu}\n", GetLastError());
        fclose(stream);
        return 3;
    }
    void *address = (uint8_t *)module + 0x29d8e0;
    Divide divide;
    memcpy(&divide, &address, sizeof(divide));
    address = (uint8_t *)module + 0x29d8d0;
    Clamp clamp;
    memcpy(&clamp, &address, sizeof(clamp));
    const uint32_t rvas[] = {0x2ae600, 0x2ae640, 0x2ae680, 0x2ae6c0, 0x2ae700};
    const uint32_t shifts[] = {6, 7, 6, 4, 4};
    Interpolate interpolate[5];
    for (size_t i = 0; i < 5; ++i) {
        address = (uint8_t *)module + rvas[i];
        memcpy(&interpolate[i], &address, sizeof(interpolate[i]));
    }
    int status = 0;
    for (uint32_t index = 0; index < count; ++index) {
        uint32_t operation, length;
        int32_t a, b, c;
        uint16_t guarded[MAX_NODES + 16], original[MAX_NODES + 16];
        memset(guarded, 0xa5, sizeof(guarded));
        uint16_t *table = guarded + 8;
        if (fread(&operation, 4, 1, stream) != 1 || fread(&a, 4, 1, stream) != 1 ||
            fread(&b, 4, 1, stream) != 1 || fread(&c, 4, 1, stream) != 1 ||
            fread(&length, 4, 1, stream) != 1 || length > MAX_NODES ||
            fread(table, 2, length, stream) != length) {
            status = 2;
            break;
        }
        memcpy(original, guarded, sizeof(guarded));
        int accepted = 0;
        int32_t result = 0;
        if (operation == 0 && length == 0 && b > 0) {
            accepted = 1;
            result = divide(a, b);
        } else if (operation == 1 && length == 0 && b <= c) {
            accepted = 1;
            result = clamp(a, b, c);
        } else if (operation >= 2 && operation <= 6 && length >= 2 && a >= 0 &&
                   ((uint32_t)a >> shifts[operation - 2]) < length - 1) {
            accepted = 1;
            uint16_t bits = interpolate[operation - 2](a, table);
            result = bits;
            if ((operation == 4 || operation == 5) && bits >= 32768)
                result -= 65536;
        }
        int unchanged = !memcmp(original, guarded, sizeof(guarded));
        printf("{\"event\":\"case\",\"index\":%u,\"accepted\":%s,"
               "\"result\":%d,\"input_and_guards_unchanged\":%s}\n",
               index, accepted ? "true" : "false", result, unchanged ? "true" : "false");
        if (!unchanged) { status = 4; break; }
    }
    if (!status && fgetc(stream) != EOF) status = 2;
    fclose(stream);
    FreeLibrary(module);
    if (!status) printf("{\"event\":\"complete\",\"cases\":%u}\n", count);
    return status;
}
