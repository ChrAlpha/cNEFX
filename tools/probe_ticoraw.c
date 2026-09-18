/* Research-only decoder oracle for the pinned DLL at RVA 0x3414c0.
 * Input: NXCT, uint32 width,height,compressed_bytes, then one raw strip.
 * Output: NXCR, uint32 width,height, then width*height uint16 samples.
 * The caller must verify the DLL hash and extract TIFF metadata structurally.
 */
#include <windows.h>
#include <stdint.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <io.h>
#include <fcntl.h>

typedef struct { void *begin, *end, *capacity; } Vector;
typedef bool (*Decode)(void *, Vector *, uint32_t, uint32_t, Vector *);
_Static_assert(sizeof(Vector) == 24, "Unexpected vector layout");

int main(int argc, char **argv) {
    if (argc != 2) {
        fprintf(stderr, "Usage: probe_ticoraw DLL < compressed.bin > decoded.bin\n");
        return 2;
    }
    _setmode(_fileno(stdin), _O_BINARY);
    _setmode(_fileno(stdout), _O_BINARY);
    uint32_t header[4];
    if (fread(header, sizeof(header), 1, stdin) != 1 ||
        memcmp(header, "NXCT", 4) || header[1] == 0 || header[2] == 0 ||
        (uint64_t)header[1] * header[2] > 64000000 || header[3] == 0 ||
        header[3] > 100000000) {
        fprintf(stderr, "Invalid bounded input header.\n");
        return 2;
    }
    size_t count = (size_t)header[1] * header[2];
    uint8_t *compressed = malloc(header[3]);
    uint16_t *samples = malloc(count * sizeof(uint16_t));
    if (!compressed || !samples) {
        fprintf(stderr, "Allocation failed.\n");
        free(compressed);
        free(samples);
        return 3;
    }
    if (fread(compressed, 1, header[3], stdin) != header[3] || fgetc(stdin) != EOF) {
        fprintf(stderr, "Invalid compressed data size.\n");
        free(compressed);
        free(samples);
        return 2;
    }
    SetErrorMode(SEM_FAILCRITICALERRORS | SEM_NOGPFAULTERRORBOX);
    HMODULE module = LoadLibraryA(argv[1]);
    if (!module) {
        fprintf(stderr, "{\"event\":\"load_failed\",\"code\":%lu}\n", GetLastError());
        free(compressed);
        free(samples);
        return 4;
    }
    void *address = (uint8_t *)module + 0x3414c0;
    Decode decode;
    memcpy(&decode, &address, sizeof(decode));
    Vector input = {compressed, compressed + header[3], compressed + header[3]};
    Vector output = {samples, samples + count, samples + count};
    bool ok = decode(NULL, &input, header[1], header[2], &output);
    int status = 5;
    if (ok && output.begin == samples && output.end == samples + count &&
        output.capacity == samples + count) {
        memcpy(header, "NXCR", 4);
        if (fwrite(header, 4, 3, stdout) == 3 &&
            fwrite(samples, sizeof(uint16_t), count, stdout) == count &&
            fflush(stdout) == 0) status = 0;
    }
    fprintf(stderr, "{\"event\":\"return\",\"success\":%s,\"status\":%d}\n",
            ok ? "true" : "false", status);
    FreeLibrary(module);
    free(compressed);
    free(samples);
    return status;
}
