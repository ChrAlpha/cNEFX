/* Research-only probe for normalized NP3 state to Rome2 table generation. */
#include <windows.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef void (*GenerateTables)(void *, void *, void *);

static int read_state(const char *path, int32_t values[41]) {
    FILE *stream = fopen(path, "rb");
    if (!stream) return 0;
    int ok = fread(values, sizeof(*values), 41, stream) == 41;
    int extra = fgetc(stream);
    fclose(stream);
    return ok && extra == EOF;
}

int main(int argc, char **argv) {
    if (argc != 4) {
        fprintf(stderr, "Usage: probe_kaiji_state Rome2.dll state.bin output.bin\n");
        return 2;
    }
    SetErrorMode(SEM_FAILCRITICALERRORS | SEM_NOGPFAULTERRORBOX);
    HMODULE module = LoadLibraryA(argv[1]);
    if (!module) return 3;
    int32_t values[41];
    if (!read_state(argv[2], values)) {
        FreeLibrary(module);
        return 4;
    }
    uint8_t *state = calloc(0x17f8 + 16, 1);
    uint8_t *output = calloc(0x34370 + 16, 1);
    if (!state || !output) return 5;
    memset(state + 0x17f8, 0xa5, 16);
    memset(output + 0x34370, 0xa5, 16);
    *(uint32_t *)(state + 0x0c) = 1071;
    *(uint32_t *)(state + 0x151c) = 12;
    *(uint32_t *)(state + 0x17ec) = 0x10000;
    *(uint16_t *)(state + 0x17f0) = 2;
    memcpy(state + 0x1764, values + 6, 24 * sizeof(int32_t));
    memcpy(state + 0x17c4, values + 30, 9 * sizeof(int32_t));
    *(uint16_t *)(state + 0x17e8) = (uint16_t)values[39];
    *(int16_t *)(state + 0x17ea) = (int16_t)values[40];
    void *address = (uint8_t *)module + 0x29f0b0;
    GenerateTables generate;
    memcpy(&generate, &address, sizeof(generate));
    generate(state, NULL, output);
    int guards = 1;
    for (size_t i = 0; i < 16; ++i) {
        if (state[0x17f8 + i] != 0xa5 || output[0x34370 + i] != 0xa5) guards = 0;
    }
    HANDLE file = CreateFileA(argv[3], GENERIC_WRITE, 0, NULL, CREATE_NEW,
                              FILE_ATTRIBUTE_NORMAL, NULL);
    DWORD written = 0;
    int success = 0;
    if (file != INVALID_HANDLE_VALUE) {
        success = WriteFile(file, output, 0x34370, &written, NULL)
            && written == 0x34370;
        CloseHandle(file);
    }
    printf("{\"event\":\"kaiji_state\",\"output_bytes\":%lu,"
           "\"guards_intact\":%s,\"success\":%s}\n",
           written, guards ? "true" : "false", success ? "true" : "false");
    free(output);
    free(state);
    FreeLibrary(module);
    return guards && success ? 0 : 6;
}
