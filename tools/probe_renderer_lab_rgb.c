/* Research-only bounded Lab/RGB kernel oracle with reconstructed pointer state. */
#include <windows.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <xmmintrin.h>

#define KERNEL_RVA 0x53a2f0u
#define CURVE_WORDS 65536u
typedef void (*Kernel)(const void *, const void *, const void *, void *, void *, void *);
typedef struct { uint8_t *allocation, *data, *copy; size_t size, pages; } Buffer;
typedef struct { void *pixels; uint32_t bits, pixel_bytes, row_bytes; } Descriptor;

static int allocate(Buffer *b, size_t size) {
    SYSTEM_INFO info;
    GetSystemInfo(&info);
    size_t page = info.dwPageSize;
    b->size = size;
    b->pages = (size + 64 + page - 1) / page * page;
    b->allocation = VirtualAlloc(NULL, b->pages + page * 2, MEM_RESERVE, PAGE_NOACCESS);
    if (!b->allocation) return 0;
    uint8_t *memory = VirtualAlloc(b->allocation + page, b->pages, MEM_COMMIT, PAGE_READWRITE);
    if (!memory) return 0;
    memset(memory, 0xa5, b->pages);
    b->data = memory + 32;
    return 1;
}

static int freeze(Buffer *b) {
    DWORD old;
    b->copy = malloc(b->size);
    if (!b->copy) return 0;
    memcpy(b->copy, b->data, b->size);
    return VirtualProtect(b->data - 32, b->pages, PAGE_READONLY, &old) != 0;
}

static int intact(const Buffer *b) {
    if (!b->data || (b->copy && memcmp(b->copy, b->data, b->size))) return 0;
    for (size_t i = 0; i < 32; ++i)
        if ((b->data - 32)[i] != 0xa5 || b->data[b->size + i] != 0xa5) return 0;
    return 1;
}

static int release(Buffer *b) {
    free(b->copy);
    return !b->allocation || VirtualFree(b->allocation, 0, MEM_RELEASE);
}

static int read_buffer(const char *path, Buffer *b) {
    FILE *file = fopen(path, "rb");
    if (!file) return 0;
    int ok = !fseek(file, 0, SEEK_END);
    long length = ok ? ftell(file) : -1;
    ok = length > 0 && length <= 4 * 1024 * 1024 && !fseek(file, 0, SEEK_SET);
    if (ok) ok = allocate(b, (size_t)length);
    if (ok) ok = fread(b->data, 1, b->size, file) == b->size &&
                 fgetc(file) == EOF && !ferror(file) && freeze(b);
    if (fclose(file)) ok = 0;
    return ok;
}

static int width_value(const char *text, unsigned *value) {
    if (!*text) return 0;
    unsigned n = 0;
    for (; *text; ++text) {
        if (*text < '0' || *text > '9' || n > 409) return 0;
        n = n * 10 + (unsigned)(*text - '0');
        if (n > 4096) return 0;
    }
    *value = n;
    return n != 0;
}

int main(int argc, char **argv) {
    unsigned width;
    if (argc != 7 || !width_value(argv[5], &width)) return 2;
    SetErrorMode(SEM_FAILCRITICALERRORS | SEM_NOGPFAULTERRORBOX);
    Buffer matrix = {0}, curves = {0}, input = {0}, row = {0}, output = {0};
    Buffer state = {0}, job = {0}, scratch = {0};
    HMODULE dll = NULL;
    const char *failure = NULL;
    size_t height = 0, rows = 0, row_bytes = (size_t)width * 6;
    unsigned before = 0, after = 0, pitch = (width + 3u) & ~3u;
    int saved = 0, guards = 0, released = 1;
    if (!read_buffer(argv[2], &matrix) || !read_buffer(argv[3], &curves) ||
        !read_buffer(argv[4], &input)) { failure = "read_input"; goto cleanup; }
    if (matrix.size != 36 || curves.size != CURVE_WORDS * 12 || input.size % row_bytes) {
        failure = "input_dimensions"; goto cleanup;
    }
    for (size_t i = 0; i < 9; ++i) {
        float value; memcpy(&value, matrix.data + i * 4, 4);
        if (!isfinite(value) || fabsf(value) > 16) { failure = "matrix_value"; goto cleanup; }
    }
    for (size_t i = 0; i < CURVE_WORDS * 3; ++i) {
        float value; memcpy(&value, curves.data + i * 4, 4);
        if (!isfinite(value) || value < 0 || value > 65535) {
            failure = "curve_value"; goto cleanup;
        }
    }
    height = input.size / row_bytes;
    if (!allocate(&row, row_bytes) || !allocate(&output, input.size) ||
        !allocate(&state, 0x890) || !allocate(&job, 0xb0) ||
        !allocate(&scratch, (size_t)pitch * 6 * 4)) {
        failure = "allocation"; goto cleanup;
    }
    memset(state.data, 0, state.size);
    memset(job.data, 0, job.size);
    memcpy(state.data + 0x3c8, matrix.data, matrix.size);
    state.data[0x230] = 1;
    for (unsigned i = 0; i < 3; ++i) {
        uint32_t identity = 6, sampled = 5;
        memcpy(state.data + 0x558 + i * 0x78, &identity, 4);
        uint8_t *curve = state.data + 0x3f0 + i * 0x78;
        memcpy(curve, &sampled, 4);
        uintptr_t pointer = (uintptr_t)curves.data + i * CURVE_WORDS * 4;
        memcpy(curve + 0x40, &pointer, 8);
        pointer += CURVE_WORDS * 4;
        memcpy(curve + 0x48, &pointer, 8);
        memcpy(curve + 0x50, &pointer, 8);
    }
    memcpy(job.data + 0x7c, &pitch, 4);
    if (!freeze(&state) || !freeze(&job)) { failure = "state_protection"; goto cleanup; }
    dll = LoadLibraryA(argv[1]);
    if (!dll) { failure = "load_provider"; goto cleanup; }
    char path[MAX_PATH];
    DWORD length = GetModuleFileNameA(dll, path, MAX_PATH);
    if (!length || length >= MAX_PATH) { failure = "provider_path"; goto cleanup; }
    printf("{\"event\":\"provider\",\"path_hex\":\"");
    for (DWORD i = 0; i < length; ++i) printf("%02x", (unsigned char)path[i]);
    puts("\"}");
    uintptr_t address = (uintptr_t)dll + KERNEL_RVA;
    MEMORY_BASIC_INFORMATION region;
    if (VirtualQuery((const void *)address, &region, sizeof(region)) != sizeof(region) ||
        region.AllocationBase != (void *)dll || region.Type != MEM_IMAGE ||
        region.State != MEM_COMMIT || region.Protect != PAGE_EXECUTE_READ) {
        failure = "kernel_memory"; goto cleanup;
    }
    Kernel kernel; memcpy(&kernel, &address, sizeof(kernel));
    before = _mm_getcsr();
    if ((before & ~0x20u) != 0x1f80u) { failure = "mxcsr_before"; goto cleanup; }
    printf("{\"event\":\"ready\",\"kernel_rva\":%u,\"width\":%u,"
           "\"height\":%llu,\"read_only_inputs\":true,\"mxcsr\":%u}\n",
           KERNEL_RVA, width, (unsigned long long)height, before);
    fflush(stdout);
    for (size_t y = 0; y < height; ++y) {
        memset(row.data, 0xa5, row.size);
        memset(scratch.data, 0, scratch.size);
        int32_t rectangle[4] = {0, 0, (int32_t)width, 1};
        Descriptor in = {input.data + y * row_bytes, 48, 6, (uint32_t)row_bytes};
        Descriptor out = {row.data, 48, 6, (uint32_t)row_bytes};
        kernel(state.data, job.data, rectangle, &in, &out, scratch.data);
        after = _mm_getcsr();
        if (!intact(&row) || !intact(&scratch) || (after & ~0x20u) != 0x1f80u) {
            failure = "row_guards_or_mxcsr"; goto cleanup;
        }
        memcpy(output.data + y * row_bytes, row.data, row_bytes);
        ++rows;
    }
    guards = intact(&matrix) && intact(&curves) && intact(&input) &&
             intact(&state) && intact(&job) && intact(&output);
    if (!guards) { failure = "inputs_or_guards"; goto cleanup; }
    HANDLE target = CreateFileA(argv[6], GENERIC_WRITE, 0, NULL, CREATE_NEW, 0, NULL);
    if (target != INVALID_HANDLE_VALUE) {
        DWORD written = 0;
        saved = WriteFile(target, output.data, (DWORD)output.size, &written, NULL) &&
                written == output.size;
        if (!CloseHandle(target)) saved = 0;
    }
    if (!saved) failure = "write_output";
cleanup:
    released &= release(&scratch); released &= release(&job); released &= release(&state);
    released &= release(&row); released &= release(&output); released &= release(&input);
    released &= release(&curves); released &= release(&matrix);
    if (dll) released &= FreeLibrary(dll) != 0;
    if (failure) printf("{\"event\":\"failure\",\"stage\":\"%s\"}\n", failure);
    printf("{\"event\":\"complete\",\"rows\":%llu,\"bytes\":%llu,\"mxcsr\":[%u,%u],"
           "\"inputs_and_guards_intact\":%s,\"saved\":%s,\"released\":%s}\n",
           (unsigned long long)rows, (unsigned long long)output.size, before, after,
           guards ? "true" : "false", saved ? "true" : "false", released ? "true" : "false");
    return !failure && saved && guards && released ? 0 : 6;
}
