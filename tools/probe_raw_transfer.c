/* Research-only utilities GetStdLut oracle; vendor runtime is never a backend. */
#include <windows.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef const uint16_t *(*GetTable)(int, int, int, int);

static int readable(const void *pointer, size_t size) {
    uintptr_t cursor = (uintptr_t)pointer;
    if (!cursor || size > UINTPTR_MAX - cursor) return 0;
    uintptr_t end = cursor + size;
    while (cursor < end) {
        MEMORY_BASIC_INFORMATION info;
        if (!VirtualQuery((void *)cursor, &info, sizeof(info)) || info.State != MEM_COMMIT
            || (info.Protect & (PAGE_GUARD | PAGE_NOACCESS))) return 0;
        DWORD mode = info.Protect & 255;
        if (mode != PAGE_READONLY && mode != PAGE_READWRITE && mode != PAGE_WRITECOPY
            && mode != PAGE_EXECUTE_READ && mode != PAGE_EXECUTE_READWRITE
            && mode != PAGE_EXECUTE_WRITECOPY) return 0;
        uintptr_t next = (uintptr_t)info.BaseAddress + info.RegionSize;
        if (next <= cursor) return 0;
        cursor = next < end ? next : end;
    }
    return 1;
}

int main(int argc, char **argv) {
    if (argc != 3) return 2;
    SetErrorMode(SEM_FAILCRITICALERRORS | SEM_NOGPFAULTERRORBOX);
    HMODULE module = LoadLibraryA(argv[1]);
    if (!module) return 3;
    FARPROC address = GetProcAddress(module,
        "?GetStdLut@utilities@@YAPEBGW4curveType@1@W4gammaType@1@HH@Z");
    GetTable get;
    memcpy(&get, &address, sizeof(get));
    char path[MAX_PATH];
    DWORD length = GetModuleFileNameA(module, path, MAX_PATH);
    if (!get || !length || length >= MAX_PATH) return 3;
    uint8_t *copy = malloc(262144);
    if (!copy) return 4;
    const uint16_t *tables[2] = {NULL, NULL};
    int ok = 1;
    for (int i = 0; i < 2; ++i) {
        tables[i] = get(0, i, 16, 16);
        if (!readable(tables[i], 131072)) { ok = 0; break; }
        memcpy(copy + i * 131072, tables[i], 131072);
    }
    if (ok) ok = !memcmp(tables[0], copy, 131072)
        && !memcmp(tables[1], copy + 131072, 131072);
    int unchanged = ok, saved = 0;
    if (ok) {
        HANDLE target = CreateFileA(argv[2], GENERIC_WRITE, 0, NULL, CREATE_NEW, 0, NULL);
        if (target != INVALID_HANDLE_VALUE) {
            DWORD written = 0;
            saved = WriteFile(target, copy, 262144, &written, NULL) && written == 262144;
            if (!CloseHandle(target)) saved = 0;
        }
    }
    free(copy);
    int released = FreeLibrary(module) != 0;
    printf("{\"event\":\"raw_transfer\",\"provider_hex\":\"");
    for (DWORD i = 0; i < length; ++i) printf("%02x", (unsigned char)path[i]);
    printf("\",\"bytes\":262144,\"unchanged\":%s,\"saved\":%s,\"released\":%s}\n",
           unchanged ? "true" : "false", saved ? "true" : "false", released ? "true" : "false");
    return unchanged && saved && released ? 0 : 6;
}
