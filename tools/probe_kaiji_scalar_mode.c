/* Research-only probe for Kaiji scalar output-mode selection. */
#include <windows.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
typedef void (*GenerateTables)(void *, void *, void *);
typedef void (*EvaluateScalar)(void *, int *, int *, int *, void *);
int main(int argc, char **argv) {
    if (argc != 3) return 2;
    HMODULE module = LoadLibraryA(argv[1]);
    if (!module) return 3;
    uint8_t *state = calloc(0x17f8 + 16, 1), *tables = calloc(0x34370 + 16, 1);
    uint8_t *context = calloc(0x40 + 16, 1);
    int16_t *red = calloc(32768 + 16, 2), *green = calloc(32768 + 16, 2), *blue = calloc(32768 + 16, 2);
    if (!state || !tables || !context || !red || !green || !blue) return 4;
    memset(state + 0x17f8, 0xa5, 16); memset(tables + 0x34370, 0xa5, 16);
    memset(context + 0x40, 0xa5, 16); memset(red + 32768, 0xa5, 32);
    memset(green + 32768, 0xa5, 32); memset(blue + 32768, 0xa5, 32);
    *(uint32_t *)(state + 0x0c) = 1071; *(uint32_t *)(state + 0x151c) = 12;
    *(uint32_t *)(state + 0x17ec) = 0x10000; *(uint16_t *)(state + 0x17f0) = 2;
    *(uint8_t *)(context + 0x00) = 1; *(uint32_t *)(context + 0x08) = 32;
    *(uint32_t *)(context + 0x0c) = (uint32_t)strtoul(argv[2], NULL, 0);
    *(void **)(context + 0x18) = red; *(void **)(context + 0x20) = green;
    *(void **)(context + 0x28) = blue; *(void **)(context + 0x38) = tables;
    void *address = (uint8_t *)module + 0x29f0b0; GenerateTables generate;
    memcpy(&generate, &address, sizeof(generate)); generate(state, NULL, tables);
    address = (uint8_t *)module + 0x2aee40; EvaluateScalar evaluate;
    memcpy(&evaluate, &address, sizeof(evaluate)); int r = 16, g = 16, b = 16;
    evaluate(NULL, &r, &g, &b, context); size_t index = (16 * 32 + 16) * 32 + 16;
    int guards = state[0x17f8] == 0xa5 && tables[0x34370] == 0xa5
        && ((uint8_t *)red)[32768 * 2] == 0xa5 && ((uint8_t *)green)[32768 * 2] == 0xa5
        && ((uint8_t *)blue)[32768 * 2] == 0xa5 && context[0x40] == 0xa5;
    printf("{\"event\":\"kaiji_scalar_mode\",\"flags\":%lu,\"values\":[%d,%d,%d],\"guards_intact\":%s}\n",
           strtoul(argv[2], NULL, 0), red[index], green[index], blue[index], guards ? "true" : "false");
    free(blue); free(green); free(red); free(context); free(tables); free(state); FreeLibrary(module);
    return guards ? 0 : 5;
}
