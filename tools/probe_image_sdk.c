/* Research-only SDK metadata loading and real TOPParam initialization. */
#include <windows.h>
#include <tlhelp32.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef int (*Initialize)(const void *);
typedef int (*Create)(void **);
typedef void (*Close)(void);
typedef void (*GetLog)(void **);
typedef uint64_t (*Count)(void *);
typedef const void *(*Entry)(void *, uint64_t);
typedef int (*Request)(void *, const void *, void **);
typedef int (*StartMetadata)(void *, void *, const void *);
typedef uint8_t (*IsDone)(void *);
typedef int (*Step)(void *);
typedef void (*Release)(void *);
typedef void *(*BuildGenerator)(void *, void *, uint32_t);
typedef uint64_t (*InitializeTOP)(void *, void *);
typedef uint64_t (*ApplyTOP)(void *, void *, double);
typedef void *(*DestroyGenerator)(void *, uint64_t);
typedef void *(*GetPicture)(void *);
typedef uint64_t (*ImportPicture)(void *, const void *, uint64_t, uint8_t, uint8_t);
typedef void *(*GetFlexible)(void *, void *);

static void *slot(void *object, unsigned offset) {
    return (*(void ***)object)[offset / sizeof(void *)];
}

static void print_hex(const uint8_t *bytes, size_t size) {
    for (size_t i = 0; i < size; ++i) printf("%02x", bytes[i]);
}

static int import_picture(HMODULE dll, void *image, const char *path) {
    FILE *file = fopen(path, "rb");
    if (!file) return 0;
    if (fseek(file, 0, SEEK_END)) { fclose(file); return 0; }
    long length = ftell(file);
    if (length < 52 || length > 4 * 1024 * 1024 || fseek(file, 0, SEEK_SET)) {
        fclose(file); return 0;
    }
    uint8_t *allocation = malloc((size_t)length + 64), *copy = malloc((size_t)length);
    if (!allocation || !copy) { free(allocation); free(copy); fclose(file); return 0; }
    memset(allocation, 0xa5, (size_t)length + 64);
    uint8_t *data = allocation + 32;
    int complete = fread(data, 1, (size_t)length, file) == (size_t)length
                && fgetc(file) == EOF && !ferror(file);
    fclose(file);
    if (!complete) { free(allocation); free(copy); return 0; }
    memcpy(copy, data, (size_t)length);
    GetPicture get;
    ImportPicture import;
    GetFlexible flexible;
    void *address = (uint8_t *)dll + 0x7bb80;
    memcpy(&get, &address, sizeof(get));
    address = (uint8_t *)dll + 0xe7480;
    memcpy(&import, &address, sizeof(import));
    address = (uint8_t *)dll + 0xe8e20;
    memcpy(&flexible, &address, sizeof(flexible));
    void *picture = get(image);
    if (!picture) { free(allocation); free(copy); return 0; }
    /* This concrete property ABI differs from the older PCEngine wrapper. */
    uint64_t result = import(picture, data, (uint64_t)length, 0, 1);
    struct { uint64_t before[4]; int32_t values[41]; uint8_t after[32]; } output;
    memset(&output, 0xa5, sizeof(output));
    if (!result) flexible(picture, output.values);
    int guards = 1;
    for (unsigned i = 0; i < 32; ++i)
        guards &= allocation[i] == 0xa5 && data[length + i] == 0xa5
               && ((uint8_t *)output.before)[i] == 0xa5 && output.after[i] == 0xa5;
    int unchanged = !memcmp(copy, data, (size_t)length);
    printf("{\"event\":\"picture_imported\",\"result\":%llu,"
           "\"guards_intact\":%s,\"input_unchanged\":%s,\"values\":[",
           (unsigned long long)result, guards ? "true" : "false", unchanged ? "true" : "false");
    for (unsigned i = 0; i < 41; ++i) printf("%s%d", i ? "," : "", output.values[i]);
    puts("]}");
    fflush(stdout);
    free(copy);
    free(allocation);
    return !result && guards && unchanged;
}

static int providers(void) {
    HANDLE snapshot = CreateToolhelp32Snapshot(TH32CS_SNAPMODULE, 0);
    if (snapshot == INVALID_HANDLE_VALUE) return 0;
    MODULEENTRY32 entry = {0};
    entry.dwSize = sizeof(entry);
    BOOL present = Module32First(snapshot, &entry);
    int success = present;
    while (present) {
        printf("{\"event\":\"provider\",\"path_hex\":\"");
        print_hex((const uint8_t *)entry.szExePath, strlen(entry.szExePath));
        puts("\"}");
        present = Module32Next(snapshot, &entry);
    }
    success &= GetLastError() == ERROR_NO_MORE_FILES;
    success &= CloseHandle(snapshot);
    return success;
}

static void dump_log(HMODULE dll) {
    GetLog get_log;
    FARPROC proc = GetProcAddress(dll, "nikon_get_result_log");
    memcpy(&get_log, &proc, sizeof(get_log));
    void *log = NULL;
    if (!get_log) return;
    get_log(&log);
    if (!log) return;
    Count count;
    Entry entry;
    void *address = slot(log, 0x10);
    memcpy(&count, &address, sizeof(count));
    address = slot(log, 0x18);
    memcpy(&entry, &address, sizeof(entry));
    uint64_t size = count(log);
    if (size > 64) size = 64;
    for (uint64_t i = 0; i < size; ++i) {
        const uint8_t *value = entry(log, i);
        if (!value) continue;
        uint32_t code;
        const char *message;
        memcpy(&code, value, sizeof(code));
        memcpy(&message, value + 8, sizeof(message));
        printf("{\"event\":\"sdk_log\",\"code\":%u,\"message_hex\":\"", code);
        if (message) print_hex((const uint8_t *)message, strnlen(message, 2048));
        puts("\"}");
    }
    fflush(stdout);
}

static int load_metadata(void *services, void *image, const char *path) {
    const uint8_t uuid[24] = {0x60,0xb2,0x86,0xd5,0xcc,0x85,0xdd,0x11,
                             0xad,0x8b,0x08,0x00,0x20,0x0c,0x9a,0x66};
    Request request;
    void *address = slot(services, 8);
    memcpy(&request, &address, sizeof(request));
    void *metadata = NULL;
    int result = request(services, uuid, &metadata);
    printf("{\"event\":\"metadata_service\",\"result\":%d,\"present\":%s}\n",
           result, metadata ? "true" : "false");
    fflush(stdout);
    if (result || !metadata) return 0;
    StartMetadata load;
    IsDone done;
    Step step, status;
    Release release;
    address = slot(metadata, 0x28); memcpy(&load, &address, sizeof(load));
    address = slot(metadata, 0x30); memcpy(&step, &address, sizeof(step));
    address = slot(metadata, 0x38); memcpy(&status, &address, sizeof(status));
    address = slot(metadata, 0x40); memcpy(&done, &address, sizeof(done));
    address = slot(metadata, 0x18); memcpy(&release, &address, sizeof(release));
    uintptr_t options[6] = {(uintptr_t)path, 0, 0, 0, 0, 0};
    result = load(metadata, image, options);
    printf("{\"event\":\"metadata_start\",\"result\":%d}\n", result);
    fflush(stdout);
    DWORD start = GetTickCount();
    unsigned steps = 0;
    while (!result && !done(metadata) && GetTickCount() - start < 30000 && steps < 10000) {
        step(metadata);
        ++steps;
    }
    int complete = !result && done(metadata);
    if (complete) result = status(metadata);
    printf("{\"event\":\"metadata_complete\",\"result\":%d,\"complete\":%s,\"steps\":%u}\n",
           result, complete ? "true" : "false", steps);
    fflush(stdout);
    /* End the isolated process if the vendor task did not reach a terminal state. */
    if (!complete && !result) ExitProcess(10);
    release(metadata);
    return !result && complete;
}

static int save(const char *prefix, const char *suffix, const uint8_t *data, DWORD size) {
    char path[512];
    int count = snprintf(path, sizeof(path), "%s.%s.bin", prefix, suffix);
    if (count < 0 || count >= (int)sizeof(path)) return 0;
    HANDLE file = CreateFileA(path, GENERIC_WRITE, 0, NULL, CREATE_NEW, FILE_ATTRIBUTE_NORMAL, NULL);
    if (file == INVALID_HANDLE_VALUE) return 0;
    DWORD written = 0;
    int success = WriteFile(file, data, size, &written, NULL) && written == size;
    return CloseHandle(file) && success;
}

static int collect_top(HMODULE dll, void *image, const char *prefix) {
    BuildGenerator build;
    InitializeTOP initialize;
    ApplyTOP apply;
    DestroyGenerator destroy;
    void *address = (uint8_t *)dll + 0x28ce20;
    memcpy(&build, &address, sizeof(build));
    address = (uint8_t *)dll + 0x289130;
    memcpy(&initialize, &address, sizeof(initialize));
    address = (uint8_t *)dll + 0x28a190;
    memcpy(&apply, &address, sizeof(apply));
    uint8_t generator[0x438 + 64], buffer[0x4d0 + 64];
    memset(generator, 0xa5, sizeof(generator));
    memset(buffer, 0xa5, sizeof(buffer));
    uint8_t *object = generator + 32, *top = buffer + 32;
    memset(object, 0, 0x438);
    memset(top, 0, 0x4d0);
    build(object, image, 1071);
    uint64_t result = initialize(object, top);
    printf("{\"event\":\"top_initialized\",\"result\":%llu}\n", (unsigned long long)result);
    fflush(stdout);
    int saved = !result && save(prefix, "initial", top, 0x4d0);
    if (saved) {
        result = apply(object, top, 1.0);
        printf("{\"event\":\"top_applied\",\"result\":%llu}\n", (unsigned long long)result);
        fflush(stdout);
        saved = !result && save(prefix, "active", top, 0x4d0);
    }
    address = slot(object, 0);
    memcpy(&destroy, &address, sizeof(destroy));
    destroy(object, 0);
    int guards = 1;
    for (unsigned i = 0; i < 32; ++i)
        guards &= generator[i] == 0xa5 && generator[32 + 0x438 + i] == 0xa5
               && buffer[i] == 0xa5 && buffer[32 + 0x4d0 + i] == 0xa5;
    printf("{\"event\":\"top_guards\",\"intact\":%s,\"saved\":%s}\n",
           guards ? "true" : "false", saved ? "true" : "false");
    fflush(stdout);
    return saved && guards;
}

#include "image_kaiji.h"
#include "image_render.h"

int main(int argc, char **argv) {
    int render_mode = argc == 7 && !strcmp(argv[6], "render");
    if (argc < 5 || argc > 8 || strlen(argv[2]) >= 260 || strlen(argv[3]) >= 260
        || (argc >= 7 && !render_mode && strcmp(argv[6], "kaiji"))
        || (argc == 8 && strcmp(argv[7], "prepare"))) return 2;
    SetErrorMode(SEM_FAILCRITICALERRORS | SEM_NOGPFAULTERRORBOX);
    HMODULE dll = LoadLibraryA(argv[1]);
    if (!dll) return 3;
    Initialize initialize;
    Create create, services;
    Close close_sdk;
    FARPROC address = GetProcAddress(dll, "nikon_initialize_sdk");
    memcpy(&initialize, &address, sizeof(initialize));
    address = GetProcAddress(dll, "nikon_create_image");
    memcpy(&create, &address, sizeof(create));
    address = GetProcAddress(dll, "nikon_get_services");
    memcpy(&services, &address, sizeof(services));
    address = GetProcAddress(dll, "nikon_close_sdk");
    memcpy(&close_sdk, &address, sizeof(close_sdk));
    if (!initialize || !create || !services || !close_sdk) return 4;
    uint64_t options[12] = {1, 0, 64, 0x40000};
    options[5] = (uintptr_t)argv[2];
    int result = initialize(options);
    printf("{\"event\":\"initialized\",\"result\":%d}\n", result);
    fflush(stdout);
    if (result) { dump_log(dll); return 5; }
    void *image = NULL, *service = NULL;
    result = create(&image);
    printf("{\"event\":\"created\",\"result\":%d,\"present\":%s}\n",
           result, image ? "true" : "false");
    fflush(stdout);
    int success = !result && image;
    if (success) {
        result = services(&service);
        printf("{\"event\":\"services\",\"result\":%d,\"present\":%s}\n",
               result, service ? "true" : "false");
        fflush(stdout);
        success = !result && service && load_metadata(service, image, argv[3]);
    }
    if (success && argc >= 6) success = import_picture(dll, image, argv[5]);
    if (success && render_mode) success = collect_render(dll, image, service, argv[4]);
    if (success && !render_mode) success = collect_top(dll, image, argv[4]);
    if (success && !render_mode && argc >= 7)
        success = collect_kaiji(dll, image, service, argv[4], argc == 8);
    dump_log(dll);
    success &= providers();
    close_sdk();
    puts("{\"event\":\"closed\"}");
    fflush(stdout);
    FreeLibrary(dll);
    return success ? 0 : 6;
}
