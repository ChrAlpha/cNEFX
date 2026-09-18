/* Research-only batch oracle for the camera-table selector at Rome2 RVA0x27e230. */
#include <windows.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum { COEFFICIENT_BYTES = 422416, GUARD = 32, MAX_CASES = 100000 };
typedef uint64_t (*SelectIndex)(uint16_t, uint16_t, uint8_t, uint8_t, uint8_t,
                               int16_t, uint8_t, uint8_t, const void *, int32_t, int32_t);

int main(int argc, char **argv) {
    if (argc != 4) return 2;
    FILE *stream = fopen(argv[2], "rb");
    if (!stream) return 2;
    uint8_t *data = malloc(COEFFICIENT_BYTES + 2 * GUARD);
    uint8_t *copy = malloc(COEFFICIENT_BYTES + 2 * GUARD);
    if (!data || !copy) return 3;
    memset(data, 0xa5, COEFFICIENT_BYTES + 2 * GUARD);
    if (fread(data + GUARD, 1, COEFFICIENT_BYTES, stream) != COEFFICIENT_BYTES
        || fgetc(stream) != EOF || ferror(stream)) return 2;
    fclose(stream);
    memcpy(copy, data, COEFFICIENT_BYTES + 2 * GUARD);
    stream = fopen(argv[3], "rb");
    char magic[8];
    uint32_t count;
    if (!stream || fread(magic, 1, 8, stream) != 8 || memcmp(magic, "NXCI001", 8)
        || fread(&count, 4, 1, stream) != 1 || !count || count > MAX_CASES) return 2;
    SetErrorMode(SEM_FAILCRITICALERRORS | SEM_NOGPFAULTERRORBOX);
    HMODULE module = LoadLibraryA(argv[1]);
    if (!module) return 4;
    void *address = (uint8_t *)module + 0x27e230;
    SelectIndex select_index;
    memcpy(&select_index, &address, sizeof(select_index));
    const int32_t low[10] = {0, 0, 0, 0, 0, -256, 0, 0, 0, 1};
    const int32_t high[10] = {65535, 65535, 19, 12, 24, 256, 40, 108, 2, 4};
    for (uint32_t index = 0; index < count; ++index) {
        int32_t v[10];
        if (fread(v, sizeof(int32_t), 10, stream) != 10) return 2;
        for (int i = 0; i < 10; ++i)
            if (v[i] < low[i] || v[i] > high[i]) return 2;
        if (v[8] != 0 && v[8] != 2) return 2;
        uint8_t result = (uint8_t)select_index(
            (uint16_t)v[0], (uint16_t)v[1], (uint8_t)v[2], (uint8_t)v[3],
            (uint8_t)v[4], (int16_t)v[5], (uint8_t)v[6], (uint8_t)v[7],
            data + GUARD, v[8], v[9]);
        if (result > 108) return 5;
        printf("{\"event\":\"case\",\"index\":%u,\"result\":%u}\n", index, result);
    }
    if (fgetc(stream) != EOF || ferror(stream)) return 2;
    fclose(stream);
    int intact = !memcmp(data, copy, COEFFICIENT_BYTES + 2 * GUARD);
    printf("{\"event\":\"complete\",\"cases\":%u,\"input_and_guards_unchanged\":%s}\n",
           count, intact ? "true" : "false");
    FreeLibrary(module);
    free(data);
    free(copy);
    return intact ? 0 : 6;
}
