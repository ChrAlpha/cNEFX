/* Research-only probe for normalized NP3 state to Rome2 table generation. */
#include <windows.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>

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
    if (argc != 4 && argc != 8 && argc != 12 && argc != 14) {
        fprintf(stderr, "Usage: probe_kaiji_state Rome2.dll state.bin output.bin "
                "[highlights shadows white black [model strength saturation matrix "
                "[contrast brightness]]]\n");
        return 2;
    }
    int16_t settings[10] = {0};
    const int low[10] = {-120, -120, -100, -100, 0, 0, -50, 0, -256, 0};
    const int high[10] = {120, 120, 100, 100, 14, 256, 50, 2, 256, 12};
    for (int i = 0; i < argc - 4; ++i) {
        char *end = NULL;
        errno = 0;
        long value = strtol(argv[4 + i], &end, 10);
        if (errno || end == argv[4 + i] || *end || value < low[i] || value > high[i])
            return 2;
        if (i == 4 && value != 0 && value != 14) return 2;
        settings[i] = (int16_t)value;
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
    if (argc >= 8) memcpy(state + 0x1758, settings, 4 * sizeof(int16_t));
    if (argc >= 12) {
        memcpy(state + 0x1518, settings + 4, sizeof(int16_t));
        memcpy(state + 0x1520, settings + 5, sizeof(int16_t));
        memcpy(state + 0x1760, settings + 6, sizeof(int16_t));
        memcpy(state + 0x1524, settings + 7, sizeof(int16_t));
    }
    if (argc == 14) {
        memcpy(state + 0x1548, settings + 8, sizeof(int16_t));
        state[0x154a] = (uint8_t)settings[9];
    }
    void *address = (uint8_t *)module + 0x29f0b0;
    GenerateTables generate;
    memcpy(&generate, &address, sizeof(generate));
    generate(state, NULL, output);
    char moon_path[MAX_PATH];
    HMODULE moon = GetModuleHandleA("Moon2.dll");
    DWORD moon_length = moon ? GetModuleFileNameA(moon, moon_path, MAX_PATH) : 0;
    if (!moon_length || moon_length >= MAX_PATH) return 7;
    for (DWORD i = 0; i < moon_length; ++i) {
        if (moon_path[i] == '\\') moon_path[i] = '/';
        if (moon_path[i] == '"' || (unsigned char)moon_path[i] < 32) return 7;
    }
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
           "\"tone\":[%d,%d,%d,%d],\"model\":%d,\"strength\":%d,"
           "\"saturation\":%d,\"matrix\":%d,\"contrast\":%d,\"brightness\":%d,"
           "\"moon_path\":\"%s\",\"guards_intact\":%s,\"success\":%s}\n",
           written, settings[0], settings[1], settings[2], settings[3],
           settings[4], settings[5], settings[6], settings[7],
           settings[8], settings[9],
           moon_path,
           guards ? "true" : "false", success ? "true" : "false");
    free(output);
    free(state);
    FreeLibrary(module);
    return guards && success ? 0 : 6;
}
