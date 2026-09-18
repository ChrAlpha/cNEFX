/* Research-only lattice probe for Kaiji matrix and output-mode initialization. */
#include <windows.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
typedef void (*EvaluateScalar)(void *, int *, int *, int *, void *);

enum { GUARD = 32, TABLE_BYTES = 0x34370, CONTEXT_BYTES = 0x40, GRID = 33,
       GRID_POINTS = GRID * GRID * GRID, PLANE_BYTES = GRID_POINTS * 2 };

static uint8_t *guarded(size_t size) {
    uint8_t *allocation = malloc(size + 2 * GUARD);
    if (!allocation) return NULL;
    memset(allocation, 0xa5, size + 2 * GUARD);
    return allocation + GUARD;
}

static int intact(const uint8_t *data, size_t size) {
    for (int i = 0; i < GUARD; ++i)
        if (data[i - GUARD] != 0xa5 || data[size + i] != 0xa5) return 0;
    return 1;
}

static int read_table(const char *path, uint8_t *table) {
    FILE *stream = fopen(path, "rb");
    if (!stream) return 0;
    int ok = fread(table, 1, TABLE_BYTES, stream) == TABLE_BYTES;
    int extra = fgetc(stream);
    ok = ok && extra == EOF && !ferror(stream);
    fclose(stream);
    return ok;
}

static int write_planes(const char *path, uint8_t *planes[3]) {
    HANDLE file = CreateFileA(path, GENERIC_WRITE, 0, NULL, CREATE_NEW,
                              FILE_ATTRIBUTE_NORMAL, NULL);
    if (file == INVALID_HANDLE_VALUE) return 0;
    int ok = 1;
    for (int i = 0; i < 3 && ok; ++i) {
        DWORD written = 0;
        ok = WriteFile(file, planes[i], PLANE_BYTES, &written, NULL)
             && written == PLANE_BYTES;
    }
    if (!CloseHandle(file)) ok = 0;
    return ok;
}

int main(int argc, char **argv) {
    if (argc != 7 || (strcmp(argv[4], "0") && strcmp(argv[4], "1"))
        || (strcmp(argv[5], "0") && strcmp(argv[5], "1"))
        || (strcmp(argv[6], "0") && strcmp(argv[6], "2"))) {
        fprintf(stderr, "Usage: probe_kaiji_scalar_mode Rome2.dll table.bin "
                "planes.bin flags[0|1] matrix[0|1] processing[0|2]\n");
        return 2;
    }
    SetErrorMode(SEM_FAILCRITICALERRORS | SEM_NOGPFAULTERRORBOX);
    uint8_t *table = guarded(TABLE_BYTES), *context = guarded(CONTEXT_BYTES);
    uint8_t *planes[3] = {guarded(PLANE_BYTES), guarded(PLANE_BYTES),
                          guarded(PLANE_BYTES)};
    uint8_t *indices[3] = {guarded(sizeof(int)), guarded(sizeof(int)),
                           guarded(sizeof(int))};
    uint8_t *snapshot = malloc(TABLE_BYTES);
    if (!table || !context || !snapshot) return 3;
    for (int i = 0; i < 3; ++i) if (!planes[i] || !indices[i]) return 3;
    if (!read_table(argv[2], table)) return 4;
    uint32_t flags = (uint32_t)atoi(argv[4]), matrix = (uint32_t)atoi(argv[5]);
    uint32_t processing = (uint32_t)atoi(argv[6]), grid = GRID;
    const double rgb_to_xyz[9] = {
        0.436071339, 0.385066415, 0.143057225,
        0.222502613, 0.716880769, 0.060614910,
        0.013922978, 0.097068147, 0.713984047
    };
    memset(table, 0, sizeof(rgb_to_xyz));
    if (matrix) memcpy(table, rgb_to_xyz, sizeof(rgb_to_xyz));
    memcpy(snapshot, table, TABLE_BYTES);
    memset(context, 0, CONTEXT_BYTES);
    context[0] = 1;
    memcpy(context + 4, &processing, sizeof(processing));
    memcpy(context + 8, &grid, sizeof(grid));
    memcpy(context + 12, &flags, sizeof(flags));
    for (int i = 0; i < 3; ++i)
        memcpy(context + 0x18 + i * 8, &planes[i], sizeof(planes[i]));
    memcpy(context + 0x38, &table, sizeof(table));
    uint8_t context_snapshot[CONTEXT_BYTES];
    memcpy(context_snapshot, context, CONTEXT_BYTES);
    HMODULE module = LoadLibraryA(argv[1]);
    if (!module) return 5;
    void *address = (uint8_t *)module + 0x2aee40;
    EvaluateScalar evaluate;
    memcpy(&evaluate, &address, sizeof(evaluate));
    int unchanged = 1, guards = 1;
    unsigned int calls = 0;
    for (int r = 0; r < GRID; ++r) for (int g = 0; g < GRID; ++g)
        for (int b = 0; b < GRID; ++b) {
            int input[3] = {r, g, b};
            for (int i = 0; i < 3; ++i) memcpy(indices[i], &input[i], sizeof(int));
            evaluate(NULL, (int *)indices[0], (int *)indices[1], (int *)indices[2],
                     context);
            ++calls;
            for (int i = 0; i < 3; ++i) {
                unchanged &= memcmp(indices[i], &input[i], sizeof(int)) == 0;
                guards &= intact(indices[i], sizeof(int));
            }
        }
    unchanged &= memcmp(table, snapshot, TABLE_BYTES) == 0;
    unchanged &= memcmp(context, context_snapshot, CONTEXT_BYTES) == 0;
    guards &= intact(table, TABLE_BYTES) && intact(context, CONTEXT_BYTES);
    for (int i = 0; i < 3; ++i) guards &= intact(planes[i], PLANE_BYTES);
    int success = guards && unchanged && write_planes(argv[3], planes);
    printf("{\"event\":\"kaiji_scalar_lattice\",\"flags\":%u,\"matrix\":%u,"
           "\"processing\":%u,\"selector\":0,\"grid\":%u,\"calls\":%u,"
           "\"guards_intact\":%s,\"inputs_unchanged\":%s,\"success\":%s}\n",
           flags, matrix, processing, grid, calls, guards ? "true" : "false",
           unchanged ? "true" : "false", success ? "true" : "false");
    for (int i = 0; i < 3; ++i) {
        free(indices[i] - GUARD);
        free(planes[i] - GUARD);
    }
    free(snapshot);
    free(context - GUARD);
    free(table - GUARD);
    FreeLibrary(module);
    return success ? 0 : 6;
}
