/* Research-only metadata oracle for pinned Rome2.dll RVA 0x132ca0.
 * This reads format version, profile ID and name; it does not import or render.
 * The runner must verify the module hash and provide structurally checked input.
 */
#include <windows.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef uint64_t (*ReadHeader)(const void *, uint64_t, uint32_t *, uint32_t *, void *);

int main(int argc, char **argv) {
    if (argc != 3) {
        fprintf(stderr, "Usage: probe_np3 Rome2.dll input.NP3\n");
        return 2;
    }
    FILE *stream = fopen(argv[2], "rb");
    if (!stream || fseek(stream, 0, SEEK_END)) return 2;
    long size = ftell(stream);
    if (size < 16 || size > 4 * 1024 * 1024 || fseek(stream, 0, SEEK_SET)) {
        fclose(stream);
        return 2;
    }
    uint8_t *data = calloc((size_t)size + 64, 1);
    if (!data) { fclose(stream); return 3; }
    size_t received = fread(data, 1, (size_t)size, stream);
    fclose(stream);
    if (received != (size_t)size) { free(data); return 2; }
    SetErrorMode(SEM_FAILCRITICALERRORS | SEM_NOGPFAULTERRORBOX);
    HMODULE module = LoadLibraryA(argv[1]);
    if (!module) {
        fprintf(stderr, "{\"event\":\"load_failed\",\"code\":%lu}\n", GetLastError());
        free(data);
        return 4;
    }
    void *address = (uint8_t *)module + 0x132ca0;
    ReadHeader read_header;
    memcpy(&read_header, &address, sizeof(read_header));
    uint32_t version = 0xffffffff, profile = 0xffffffff;
    uint8_t name[20] = {0};
    uint64_t status = read_header(data, (uint64_t)size, &version, &profile, name);
    printf("{\"event\":\"header\",\"status\":%llu,\"version_mask\":%u,"
           "\"profile_id\":%u,\"name_hex\":\"",
           (unsigned long long)status, version, profile);
    for (size_t i = 0; i < sizeof(name); ++i) printf("%02x", name[i]);
    printf("\"}\n");
    FreeLibrary(module);
    free(data);
    return status == 0 ? 0 : 5;
}
