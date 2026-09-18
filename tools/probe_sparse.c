/* Bounded internal-function oracle for the pinned PixelShiftCombine DLL.
 * Input: NXCS, uint32 case count, then per case 24 selector bytes and
 * four 16x12 little-endian uint16 planes. Output: NXCO, count, result planes.
 * Verify the DLL SHA-256 in the caller before invoking this research tool.
 */
#include <windows.h>
#include <stdint.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <fcntl.h>
#include <io.h>

enum { WIDTH = 16, HEIGHT = 12, SAMPLES = WIDTH * HEIGHT };
typedef struct { uint16_t *begin, *end, *capacity; } Vector;
typedef struct {
    int32_t width, height;
    Vector *planes[4];
    uint8_t selector[24];
} Descriptor;
typedef uint64_t (*Repair)(Descriptor *);
_Static_assert(offsetof(Descriptor, selector) == 40, "Unexpected ABI layout");
_Static_assert(sizeof(Descriptor) == 64, "Unexpected ABI size");
_Static_assert(sizeof(Vector) == 24, "Unexpected vector size");

static uint16_t word(const uint8_t *data) {
    return (uint16_t)(data[0] | ((uint16_t)data[1] << 8));
}

static int valid_selection(const uint8_t *s) {
    int row = word(s + 8) - ((int)word(s + 2) - HEIGHT) / 2;
    int lower = ((int)word(s + 4) - WIDTH) / 4;
    unsigned iterations = (unsigned)s[16] * s[15] * (s[18] + s[20] + 1);
    if (iterations > 1000) return 0;
    for (int outer = 0; outer < s[16]; ++outer) {
        if (outer) row += word(s + 12);
        for (int middle = 0; middle < s[15]; ++middle) {
            if (middle) row += word(s + 10);
            for (int inner = 0; inner < s[18] + s[20] + 1; ++inner) {
                if (inner) row += s[14];
                if (row < 0 || row > HEIGHT) continue;
                int column = 1;
                for (int half = 1; half <= WIDTH / 2 - 2; ++half) {
                    if (half < word(s + 6) / 2 || half < lower ||
                        half > lower + WIDTH / 2) continue;
                    if (half != 1) column += 2;
                    if (row < 2 || row >= HEIGHT - 2 || column >= WIDTH - 1) return 0;
                }
            }
        }
    }
    return 1;
}

int main(int argc, char **argv) {
    if (argc != 2) {
        fprintf(stderr, "Usage: probe_sparse DLL < cases.bin > results.bin\n");
        return 2;
    }
    SetErrorMode(SEM_FAILCRITICALERRORS | SEM_NOGPFAULTERRORBOX);
    HMODULE module = LoadLibraryA(argv[1]);
    if (!module) {
        fprintf(stderr, "{\"event\":\"load_failed\",\"code\":%lu}\n", GetLastError());
        return 3;
    }
    void *address = (uint8_t *)module + 0x34ba60;
    Repair repair;
    memcpy(&repair, &address, sizeof(repair));
    _setmode(_fileno(stdin), _O_BINARY);
    _setmode(_fileno(stdout), _O_BINARY);
    char magic[4];
    uint32_t count;
    if (fread(magic, 1, 4, stdin) != 4 || memcmp(magic, "NXCS", 4) ||
        fread(&count, 4, 1, stdin) != 1 || count == 0 || count > 100000) {
        fprintf(stderr, "Invalid case header.\n");
        return 2;
    }
    if (fwrite("NXCO", 1, 4, stdout) != 4 || fwrite(&count, 4, 1, stdout) != 1) return 4;
    for (uint32_t index = 0; index < count; ++index) {
        uint16_t samples[4][SAMPLES];
        Vector vectors[4];
        Descriptor descriptor = { .width = WIDTH, .height = HEIGHT };
        if (fread(descriptor.selector, 1, 24, stdin) != 24 ||
            fread(samples, sizeof(samples), 1, stdin) != 1 ||
            !valid_selection(descriptor.selector)) {
            fprintf(stderr, "Invalid input at case %u.\n", index);
            return 2;
        }
        for (int frame = 0; frame < 4; ++frame) {
            vectors[frame] = (Vector){samples[frame], samples[frame] + SAMPLES,
                                     samples[frame] + SAMPLES};
            descriptor.planes[frame] = &vectors[frame];
        }
        if (repair(&descriptor) != 1) {
            fprintf(stderr, "Repair failed at case %u.\n", index);
            return 5;
        }
        for (int frame = 0; frame < 4; ++frame) {
            if (vectors[frame].begin != samples[frame] ||
                vectors[frame].end != samples[frame] + SAMPLES ||
                vectors[frame].capacity != samples[frame] + SAMPLES) {
                fprintf(stderr, "Unexpected vector replacement at case %u.\n", index);
                return 5;
            }
        }
        if (fwrite(samples, sizeof(samples), 1, stdout) != 1) return 4;
    }
    if (fgetc(stdin) != EOF || fflush(stdout) != 0) return 4;
    FreeLibrary(module);
    return 0;
}
