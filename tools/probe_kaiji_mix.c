/* Research-only batch oracle for Rome2 RVA0x2ace00, camera1071/family12. */
#include <windows.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum { STATE_BYTES = 0x17f8, GUARD = 32, MAX_CASES = 100000 };
typedef uint32_t (*BlendMix)(const void *);
typedef void (*MatrixProduct)(void *, const void *, const void *, void *);

int main(int argc, char **argv) {
    if (argc != 3) return 2;
    FILE *stream = fopen(argv[2], "rb");
    char magic[8];
    uint32_t count;
    if (!stream) return 2;
    int valid = fread(magic, 1, 8, stream) == 8 && !memcmp(magic, "NXCM001", 8)
             && fread(&count, 4, 1, stream) == 1 && count && count <= MAX_CASES;
    if (!valid) { fclose(stream); return 2; }
    int32_t *inputs = malloc((size_t)count * 13 * sizeof(int32_t));
    if (!inputs) { fclose(stream); return 3; }
    valid = fread(inputs, sizeof(int32_t) * 13, count, stream) == count
         && fgetc(stream) == EOF && !ferror(stream);
    fclose(stream);
    if (!valid) { free(inputs); return 2; }
    for (uint32_t i = 0; i < count; ++i) {
        const int32_t *v = inputs + i * 13;
        for (unsigned j = 0; j < 11; ++j)
            if (v[j] < -32768 || v[j] > 32767) { free(inputs); return 2; }
        if (v[11] < 0 || v[11] > 21) { free(inputs); return 2; }
    }
    SetErrorMode(SEM_FAILCRITICALERRORS | SEM_NOGPFAULTERRORBOX);
    HMODULE module = LoadLibraryA(argv[1]);
    if (!module) { free(inputs); return 4; }
    char path[MAX_PATH];
    DWORD length = GetModuleFileNameA(module, path, MAX_PATH);
    if (!length || length >= MAX_PATH) { FreeLibrary(module); free(inputs); return 4; }
    printf("{\"event\":\"provider\",\"path_hex\":\"");
    for (DWORD i = 0; i < length; ++i) printf("%02x", (unsigned char)path[i]);
    puts("\"}");
    void *address = (uint8_t *)module + 0x2ace00;
    BlendMix blend;
    memcpy(&blend, &address, sizeof(blend));
    MatrixProduct product;
    address = (uint8_t *)module + 0x29d0b0;
    memcpy(&product, &address, sizeof(product));
    uint64_t allocation[(STATE_BYTES + 2 * GUARD) / 8];
    uint8_t copy[STATE_BYTES + 2 * GUARD];
    uint8_t *data = (uint8_t *)allocation, *state = data + GUARD;
    int intact = 1;
    for (uint32_t index = 0; index < count; ++index) {
        const int32_t *v = inputs + index * 13;
        memset(data, 0xa5, sizeof(allocation));
        memset(state, 0, STATE_BYTES);
        uint32_t camera = 1071, family = 12;
        memcpy(state + 0xc, &camera, 4);
        memcpy(state + 0x151c, &family, 4);
        for (unsigned j = 0; j < 9; ++j) {
            int16_t word = (int16_t)v[j];
            memcpy(state + 0xb4 + j * 2, &word, 2);
        }
        int16_t red = (int16_t)v[9], blue = (int16_t)v[10];
        memcpy(state + 0x152e, &red, 2);
        memcpy(state + 0x1530, &blue, 2);
        memcpy(state + 0x66c, &v[11], 4);
        memcpy(state + 0x674, &v[12], 4);
        memcpy(copy, data, sizeof(copy));
        uint32_t result = blend(state);
        struct { uint64_t before[4]; int16_t values[9]; uint8_t after[32]; } matrix;
        memset(&matrix, 0xa5, sizeof(matrix));
        product(NULL, (uint8_t *)module + 0x4f3c78, state + 0xb4, matrix.values);
        intact &= !memcmp(copy, data, sizeof(copy));
        for (unsigned j = 0; j < 32; ++j)
            intact &= ((uint8_t *)matrix.before)[j] == 0xa5 && matrix.after[j] == 0xa5;
        printf("{\"event\":\"case\",\"index\":%u,\"result\":%u,\"matrix\":[", index, result);
        for (unsigned j = 0; j < 9; ++j) printf("%s%d", j ? "," : "", matrix.values[j]);
        puts("]}");
        if (!intact) break;
    }
    printf("{\"event\":\"complete\",\"cases\":%u,\"state_and_guards_unchanged\":%s}\n",
           count, intact ? "true" : "false");
    FreeLibrary(module);
    free(inputs);
    return intact ? 0 : 5;
}
