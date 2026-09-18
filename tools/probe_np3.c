/* Research-only NP3 metadata and normalized parameter oracle for Rome2.dll.
 * This reads file records and normalized settings; it does not render images.
 * The runner must verify the module hash and provide structurally checked input.
 */
#include <windows.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef uint64_t (*ReadHeader)(const void *, uint64_t, uint32_t *, uint32_t *, void *);
typedef void *(*ReadParameters)(const void *, uint64_t, void *, void *, void *, void *, void *);
typedef uint64_t (*NormalizeFlexible)(const void *, int32_t *);

static void hex_field(const char *key, const uint8_t *data, size_t size) {
    printf(",\"%s\":\"", key);
    for (size_t i = 0; i < size; ++i) printf("%02x", data[i]);
    printf("\"");
}

int main(int argc, char **argv) {
    if (argc != 3 && !(argc == 4 && !strcmp(argv[3], "parameters"))) {
        fprintf(stderr, "Usage: probe_np3 Rome2.dll input.NP3 [parameters]\n");
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
    if (argc == 4 && status == 0 && version == 64) {
        uint8_t base[108 + 16] = {0}, curve[578 + 16] = {0}, flexible[88 + 16] = {0};
        uint8_t comment[2000 + 16] = {0}, extra[256 + 16] = {0};
        uint8_t *buffers[] = {base, curve, flexible, comment, extra};
        const size_t sizes[] = {108, 578, 88, 2000, 256};
        for (size_t i = 0; i < 5; ++i) memset(buffers[i] + sizes[i], 0xa5, 16);
        address = (uint8_t *)module + 0x137080;
        ReadParameters read_parameters;
        memcpy(&read_parameters, &address, sizeof(read_parameters));
        void *error = read_parameters(data, (uint64_t)size, base, curve, flexible,
                                      comment, extra);
        int guards = 1;
        for (size_t i = 0; i < 5; ++i)
            for (size_t j = 0; j < 16; ++j)
                if (buffers[i][sizes[i] + j] != 0xa5) guards = 0;
        printf("{\"event\":\"parameters\",\"success\":%s,\"guards_intact\":%s",
               error ? "false" : "true", guards ? "true" : "false");
        hex_field("base_hex", base, 108);
        hex_field("curve_hex", curve, 578);
        hex_field("flexible_hex", flexible, 88);
        hex_field("comment_hex", comment, 2000);
        hex_field("extra_hex", extra, 256);
        if (!error && guards) {
            struct { int32_t values[41]; uint8_t guard[16]; } normalized = {0};
            memset(normalized.guard, 0xa5, sizeof(normalized.guard));
            address = (uint8_t *)module + 0x131b90;
            NormalizeFlexible normalize;
            memcpy(&normalize, &address, sizeof(normalize));
            uint64_t normalized_status = normalize(flexible, normalized.values);
            int normalized_guard = 1;
            for (size_t i = 0; i < sizeof(normalized.guard); ++i)
                if (normalized.guard[i] != 0xa5) normalized_guard = 0;
            printf(",\"normalized_return\":%llu,\"normalized_guard_intact\":%s,"
                   "\"normalized_values\":[", (unsigned long long)normalized_status,
                   normalized_guard ? "true" : "false");
            for (size_t i = 0; i < 41; ++i)
                printf("%s%d", i ? "," : "", normalized.values[i]);
            printf("]");
            if (!normalized_guard) status = 1;
        }
        printf("}\n");
        if (error || !guards) status = 1;
    } else if (argc == 4) {
        status = 1;
    }
    FreeLibrary(module);
    free(data);
    return status == 0 ? 0 : 5;
}
