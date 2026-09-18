/* Research-only probe for the recovered Rome2 Flexible Color table generator. */
#include <windows.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef void (*GenerateTables)(void *, void *, void *);

int main(int argc, char **argv) {
    if (argc != 3 && argc != 4) {
        fprintf(stderr, "Usage: probe_kaiji_tables Rome2.dll output.bin [mode]\n");
        return 2;
    }
    SetErrorMode(SEM_FAILCRITICALERRORS | SEM_NOGPFAULTERRORBOX);
    HMODULE module = LoadLibraryA(argv[1]);
    if (!module) return 3;
    uint8_t *state = calloc(0x17f8 + 16, 1);
    uint8_t *output = calloc(0x34370 + 16, 1);
    if (!state || !output) return 4;
    memset(state + 0x17f8, 0xa5, 16);
    memset(output + 0x34370, 0xa5, 16);
    *(uint32_t *)(state + 0x0c) = 1071;
    *(uint32_t *)(state + 0x151c) = 12;
    *(uint32_t *)(state + 0x17ec) = 0x10000;
    *(uint16_t *)(state + 0x17ee) = 0;
    *(uint16_t *)(state + 0x17f0) = 2;
    const char *mode = argc == 4 ? argv[3] : "neutral";
    if (!strcmp(mode, "blender-red-hue")) *(int32_t *)(state + 0x1764) = 64;
    else if (!strcmp(mode, "blender-red-chroma")) *(int32_t *)(state + 0x1768) = 64;
    else if (!strcmp(mode, "blender-red-brightness")) *(int32_t *)(state + 0x176c) = 64;
    else if (!strcmp(mode, "grading-highlights-hue")) *(int32_t *)(state + 0x17c4) = 64;
    else if (!strcmp(mode, "grading-highlights-chroma")) *(int32_t *)(state + 0x17c8) = 64;
    else if (!strcmp(mode, "grading-highlights-brightness")) *(int32_t *)(state + 0x17cc) = 64;
    else if (!strcmp(mode, "blending")) *(uint16_t *)(state + 0x17e8) = 64;
    else if (!strcmp(mode, "balance")) *(int16_t *)(state + 0x17ea) = 64;
    else if (strcmp(mode, "neutral")) {
        free(output);
        free(state);
        FreeLibrary(module);
        return 4;
    }
    void *address = (uint8_t *)module + 0x29f0b0;
    GenerateTables generate;
    memcpy(&generate, &address, sizeof(generate));
    generate(state, NULL, output);
    int guards = 1;
    for (size_t i = 0; i < 16; ++i) {
        if (state[0x17f8 + i] != 0xa5 || output[0x34370 + i] != 0xa5) guards = 0;
    }
    HANDLE file = CreateFileA(argv[2], GENERIC_WRITE, 0, NULL, CREATE_NEW,
                              FILE_ATTRIBUTE_NORMAL, NULL);
    DWORD written = 0;
    int success = 0;
    if (file != INVALID_HANDLE_VALUE) {
        success = WriteFile(file, output, 0x34370, &written, NULL)
            && written == 0x34370;
        CloseHandle(file);
    }
    printf("{\"event\":\"kaiji_tables\",\"mode\":\"%s\",\"output_bytes\":%lu,"
           "\"guards_intact\":%s,\"success\":%s}\n",
           mode, written, guards ? "true" : "false", success ? "true" : "false");
    free(output);
    free(state);
    FreeLibrary(module);
    return guards && success ? 0 : 5;
}
