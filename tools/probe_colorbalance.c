/* Research-only read through the actual Nikon NEF metadata loader. */
#include <windows.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef void *(*Factory)(void **);
typedef void *(*SetPath)(void *, const char *);
typedef void *(*Load)(void *, void *, void *);
typedef void *(*Read)(void *, void *, void *);
typedef void *(*GetTag)(void *, uint32_t, uint32_t, uint64_t, void *, uint32_t);
typedef uint8_t (*Property)(void *, uint32_t, const void *, uint32_t *, void *, void *);
typedef void (*Release)(void *);

static void *slot(void *object, unsigned offset) {
    return (*(void ***)object)[offset / sizeof(void *)];
}

int main(int argc, char **argv) {
    if (argc != 4 || strlen(argv[2]) >= 260) return 2;
    SetErrorMode(SEM_FAILCRITICALERRORS | SEM_NOGPFAULTERRORBOX);
    HMODULE module = LoadLibraryA(argv[1]);
    if (!module) return 3;
    char strato_path[MAX_PATH];
    HMODULE strato = GetModuleHandleA("Strato10.dll");
    DWORD path_length = strato ? GetModuleFileNameA(strato, strato_path, MAX_PATH) : 0;
    if (!path_length || path_length >= MAX_PATH) return 3;
    for (DWORD i = 0; i < path_length; ++i) {
        if (strato_path[i] == '\\') strato_path[i] = '/';
        if (strato_path[i] == '"' || (unsigned char)strato_path[i] < 32) return 3;
    }
    void *stream = NULL, *tags = NULL, *loader = NULL;
    void *address;
    Factory create;
    const uintptr_t factories[] = {0x4c75c0, 0x4cfbe0, 0x4c8ab0};
    void **objects[] = {&stream, &tags, &loader};
    for (unsigned i = 0; i < 3; ++i) {
        address = (uint8_t *)module + factories[i];
        memcpy(&create, &address, sizeof(create));
        if (create(objects[i]) || !*objects[i]) return 4;
    }
    char path[260] = {0};
    memcpy(path, argv[2], strlen(argv[2]));
    SetPath set_path;
    address = slot(stream, 0x70);
    memcpy(&set_path, &address, sizeof(set_path));
    if (set_path(stream, path)) return 5;
    Load load;
    address = slot(loader, 0);
    memcpy(&load, &address, sizeof(load));
    void *error = load(loader, stream, tags);
    printf("{\"event\":\"load_init\",\"success\":%s}\n", error ? "false" : "true");
    fflush(stdout);
    if (error) return 6;
    Read read;
    address = slot(loader, 0x18);
    memcpy(&read, &address, sizeof(read));
    /* Optional metadata consumers must be explicit null ABI arguments. */
    error = read(loader, NULL, NULL);
    printf("{\"event\":\"read_metadata\",\"strato_path\":\"%s\",\"success\":%s}\n",
           strato_path, error ? "false" : "true");
    fflush(stdout);
    if (error) return 7;
    uint8_t allocation[0xd08 + 64];
    memset(allocation, 0xa5, sizeof(allocation));
    uint8_t *buffer = allocation + 32;
    GetTag get_tag;
    address = slot(tags, 0x28);
    memcpy(&get_tag, &address, sizeof(get_tag));
    error = get_tag(tags, 0x80097, 0x109, 0xd08, buffer, 0);
    if (error) return 8;
    uint8_t saved[0xd08];
    memcpy(saved, buffer, sizeof(saved));
    uint32_t result[3] = {0xa5a5a5a5, 0xffffffff, 0xa5a5a5a5};
    Property property;
    address = slot(tags, 0xd0);
    memcpy(&property, &address, sizeof(property));
    unsigned status = property(tags, 0x100016b, buffer, result + 1, NULL, NULL);
    int guards = result[0] == 0xa5a5a5a5 && result[2] == 0xa5a5a5a5;
    for (int i = 0; i < 32; ++i)
        guards &= allocation[i] == 0xa5 && buffer[0xd08 + i] == 0xa5;
    guards &= !memcmp(saved, buffer, sizeof(saved));
    int success = status == 1 && guards;
    if (success) {
        HANDLE file = CreateFileA(argv[3], GENERIC_WRITE, 0, NULL, CREATE_NEW, FILE_ATTRIBUTE_NORMAL, NULL);
        DWORD written = 0;
        success = file != INVALID_HANDLE_VALUE;
        if (success) {
            success = WriteFile(file, buffer, sizeof(saved), &written, NULL) && written == sizeof(saved);
            if (!CloseHandle(file)) success = 0;
        }
    }
    printf("{\"event\":\"colorbalance_iso\",\"status\":%u,\"value\":%u,"
           "\"guards_intact\":%s,\"success\":%s}\n", status, result[1], guards ? "true" : "false", success ? "true" : "false");
    void *release_objects[] = {loader, tags, stream};
    for (unsigned i = 0; i < 3; ++i) {
        Release release;
        address = slot(release_objects[i], 0x10);
        memcpy(&release, &address, sizeof(release));
        release(release_objects[i]);
    }
    FreeLibrary(module);
    return success ? 0 : 9;
}
