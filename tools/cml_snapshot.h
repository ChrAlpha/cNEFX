/* Pinned CML10 research state; not a supported vendor ABI. */
#ifndef NXC_CML_SNAPSHOT_H
#define NXC_CML_SNAPSHOT_H
#include <xmmintrin.h>

typedef void *(*CmlBufferAccess)(void *);
typedef void (*CmlBufferRelease)(void *);

static uint32_t cml_u32(const void *base, size_t offset) {
    uint32_t value;
    memcpy(&value, (const uint8_t *)base + offset, sizeof(value));
    return value;
}

static uintptr_t cml_pointer(const void *base, size_t offset) {
    uintptr_t value;
    memcpy(&value, (const uint8_t *)base + offset, sizeof(value));
    return value;
}

static int cml_save(const char *prefix, const char *suffix, const void *data, DWORD size) {
    char path[4096];
    int length = snprintf(path, sizeof(path), "%s.%s.bin", prefix, suffix);
    if (length < 0 || (size_t)length >= sizeof(path)) return 0;
    HANDLE file = CreateFileA(path, GENERIC_WRITE, 0, NULL, CREATE_NEW, 0, NULL);
    if (file == INVALID_HANDLE_VALUE) return 0;
    DWORD written = 0;
    int ok = WriteFile(file, data, size, &written, NULL) && written == size;
    ok &= CloseHandle(file) != 0;
    return ok;
}

static int cml_snapshot(HMODULE dll, void *xform, void *transformer,
                        const char *prefix, unsigned rounding_before) {
    CmlBufferAccess access;
    CmlBufferRelease release;
    uintptr_t address = (uintptr_t)dll + 0xe380;
    memcpy(&access, &address, sizeof(access));
    address = (uintptr_t)dll + 0xe080;
    memcpy(&release, &address, sizeof(release));
    void *core = (void *)cml_pointer(xform, 0x8a8);
    if (!core || cml_u32(xform, 0x8a0) != 3 || cml_u32(xform, 0x8a4) != 3
        || cml_u32(transformer, 0x590) != 16 || cml_u32(transformer, 0x594) != 16)
        return 1;
    uintptr_t vtable = cml_pointer(core, 0);
    if (vtable != (uintptr_t)dll + 0x32b90) return 1;
    uintptr_t entry = cml_pointer((void *)vtable, 0x38);
    uintptr_t inner = cml_pointer(core, 0xb0);
    if (entry != (uintptr_t)dll + 0x1f320 || inner != (uintptr_t)dll + 0x245c0
        || cml_u32(core, 0x90) != 32 || cml_u32(core, 0x94) != 32
        || cml_u32(core, 0x98) != 32 || cml_u32(core, 0xa0) != 3
        || cml_u32(core, 0xa4) != 3 || ((uint8_t *)core)[0xa8] != 1)
        return 1;
    printf("{\"event\":\"cml_state\",\"dimensions\":[%u,%u,%u],"
           "\"channels\":[%u,%u],\"mode\":%u,\"entry_rva\":%llu,\"inner_rva\":%llu,"
           "\"flags\":[%u,%u,%u,%u],\"identity_masks\":[%u,%u],"
           "\"mxcsr_rounding\":[%u,%u]}\n",
           cml_u32(core, 0x90), cml_u32(core, 0x94), cml_u32(core, 0x98),
           cml_u32(core, 0xa0), cml_u32(core, 0xa4), ((uint8_t *)core)[0xa8],
           (unsigned long long)(entry - (uintptr_t)dll),
           (unsigned long long)(inner - (uintptr_t)dll),
           ((uint8_t *)xform)[0x161], ((uint8_t *)xform)[0x162],
           ((uint8_t *)xform)[0x163], ((uint8_t *)xform)[0x164],
           cml_u32(transformer, 0xc8), cml_u32(transformer, 0xcc),
           rounding_before, (_mm_getcsr() >> 13) & 3);
    void *buffers[] = {(uint8_t *)transformer + 0x598,
                       (uint8_t *)transformer + 0x620, (uint8_t *)core + 8};
    const char *names[] = {"input-lut", "output-lut", "grid"};
    for (unsigned i = 0; i < 3; ++i) {
        uintptr_t count = cml_pointer(buffers[i], 0);
        if (count != (i < 2 ? 3 * 65536 : 32 * 32 * 32 * 4))
            return 2;
        uint16_t *copy = calloc(count, sizeof(*copy));
        if (!copy) return 3;
        const uint16_t *data = access(buffers[i]);
        if (data) {
            if (i == 2) {
                /* The fourth grid word is unused, uninitialized padding. */
                for (uintptr_t node = 0; node < count; node += 4)
                    memcpy(copy + node, data + node, 3 * sizeof(*copy));
            } else memcpy(copy, data, count * sizeof(*copy));
        }
        release(buffers[i]);
        int ok = data && cml_save(prefix, names[i], copy, (DWORD)(count * 2));
        free(copy);
        if (!ok) return 3;
    }
    return 0;
}
#endif
