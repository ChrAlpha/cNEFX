/* Research-only image_render RGB/Lab row oracle; no vendor patching. */
#include <windows.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <xmmintrin.h>

#define GRID_BYTES (65u * 65u * 65u * 6u)
#define KERNEL_RVA 0x528460u
typedef void (*RowKernel)(const void *, const void *, intptr_t, void *, intptr_t, int);
typedef struct { uint8_t *allocation, *data, *copy; size_t size, pages; } Buffer;

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

static int number(const char *text, unsigned *value) {
    if (!*text) return 0;
    unsigned n = 0;
    for (; *text; ++text) {
        if (*text < '0' || *text > '9' || n > 6553) return 0;
        n = n * 10 + (unsigned)(*text - '0');
        if (n > 65536) return 0;
    }
    *value = n;
    return n != 0;
}

int main(int argc, char **argv) {
    unsigned width;
    if (argc != 6 || !number(argv[4], &width)) return 2;
    SetErrorMode(SEM_FAILCRITICALERRORS | SEM_NOGPFAULTERRORBOX);
    Buffer grid = {0}, input = {0}, row = {0}, output = {0}, state = {0};
    HMODULE dll = NULL;
    const char *failure = NULL;
    size_t height = 0, rows = 0, row_bytes = (size_t)width * 6;
    unsigned before = 0, after = 0;
    int saved = 0, guards = 0, released = 1;
    if (!read_buffer(argv[2], &grid) || !read_buffer(argv[3], &input)) {
        failure = "read_input"; goto cleanup;
    }
    if (grid.size != GRID_BYTES || input.size % row_bytes) {
        failure = "input_dimensions"; goto cleanup;
    }
    height = input.size / row_bytes;
    if (!allocate(&row, row_bytes) || !allocate(&output, input.size) || !allocate(&state, 0x50)) {
        failure = "allocation"; goto cleanup;
    }
    memset(state.data, 0, state.size);
    uintptr_t pointer = (uintptr_t)grid.data;
    memcpy(state.data + 8, &pointer, 8);
    pointer += grid.size;
    memcpy(state.data + 0x10, &pointer, 8);
    for (unsigned i = 0; i < 3; ++i) {
        state.data[0x20 + i] = 65;
        float scale = 64.0f * (16384.0f / 65535.0f);
        memcpy(state.data + 0x24 + i * 4, &scale, 4);
    }
    uint64_t strides[3] = {65 * 65 * 3, 65 * 3, 3};
    memcpy(state.data + 0x30, strides, sizeof(strides));
    if (!freeze(&state)) { failure = "state_protection"; goto cleanup; }
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
    RowKernel kernel;
    memcpy(&kernel, &address, sizeof(kernel));
    before = _mm_getcsr();
    if ((before & ~0x20u) != 0x1f80u) { failure = "mxcsr_before"; goto cleanup; }
    printf("{\"event\":\"ready\",\"kernel_rva\":%u,\"nodes\":65,\"width\":%u,"
           "\"height\":%llu,\"read_only_inputs\":true,\"mxcsr\":%u}\n",
           KERNEL_RVA, width, (unsigned long long)height, before);
    fflush(stdout);
    for (size_t y = 0; y < height; ++y) {
        memset(row.data, 0xa5, row.size);
        kernel(state.data, input.data + y * row_bytes, 1, row.data, 1, (int)width);
        after = _mm_getcsr();
        if (!intact(&row) || (after & ~0x20u) != 0x1f80u) {
            failure = "row_guards_or_mxcsr"; goto cleanup;
        }
        memcpy(output.data + y * row_bytes, row.data, row_bytes);
        ++rows;
    }
    guards = intact(&grid) && intact(&input) && intact(&state) && intact(&output);
    if (!guards) { failure = "inputs_or_guards"; goto cleanup; }
    HANDLE target = CreateFileA(argv[5], GENERIC_WRITE, 0, NULL, CREATE_NEW, 0, NULL);
    if (target != INVALID_HANDLE_VALUE) {
        DWORD written = 0;
        saved = WriteFile(target, output.data, (DWORD)output.size, &written, NULL) &&
                written == output.size;
        if (!CloseHandle(target)) saved = 0;
    }
    if (!saved) failure = "write_output";
cleanup:
    released &= release(&state);
    released &= release(&row);
    released &= release(&output);
    released &= release(&input);
    released &= release(&grid);
    if (dll) released &= FreeLibrary(dll) != 0;
    if (failure) printf("{\"event\":\"failure\",\"stage\":\"%s\"}\n", failure);
    printf("{\"event\":\"complete\",\"rows\":%llu,\"bytes\":%llu,\"mxcsr\":[%u,%u],"
           "\"inputs_and_guards_intact\":%s,\"saved\":%s,\"released\":%s}\n",
           (unsigned long long)rows, (unsigned long long)output.size, before, after,
           guards ? "true" : "false", saved ? "true" : "false", released ? "true" : "false");
    return !failure && saved && guards && released ? 0 : 6;
}
