/* Research-only coefficient oracle for the pinned Moon2.dll. */
#include <windows.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

typedef void *(*GetTable)(uint32_t, uint32_t);
typedef uint32_t (*GetWord)(void *);
typedef const void *(*GetData)(void *);
typedef void (*Release)(void *);

int main(int argc, char **argv) {
    if (argc != 3) {
        fprintf(stderr, "Usage: probe_color_coefficients Moon2.dll output.bin\n");
        return 2;
    }
    SetErrorMode(SEM_FAILCRITICALERRORS | SEM_NOGPFAULTERRORBOX);
    HMODULE module = LoadLibraryA(argv[1]);
    if (!module) return 3;
    FARPROC address = GetProcAddress(module, "?GetCoefTable@MIPU@@YAPEBVICoefTable@1@II@Z");
    if (!address) { FreeLibrary(module); return 4; }
    GetTable get_table;
    memcpy(&get_table, &address, sizeof(get_table));
    void *table = get_table(0x1002, 1071);
    if (!table) {
        printf("{\"event\":\"coefficients\",\"loaded\":false}\n");
        FreeLibrary(module);
        return 5;
    }
    void **vtable = *(void ***)table;
    Release release;
    GetWord get_engine, get_camera, get_size;
    GetData get_data;
    memcpy(&release, vtable + 1, sizeof(release));
    memcpy(&get_engine, vtable + 2, sizeof(get_engine));
    memcpy(&get_camera, vtable + 3, sizeof(get_camera));
    memcpy(&get_size, vtable + 4, sizeof(get_size));
    memcpy(&get_data, vtable + 5, sizeof(get_data));
    uint32_t engine = get_engine(table), camera = get_camera(table), size = get_size(table);
    const void *data = get_data(table);
    int status = 6;
    DWORD written = 0;
    if (engine == 0x1002 && camera == 1071 && size == 422416 && data) {
        HANDLE output = CreateFileA(argv[2], GENERIC_WRITE, 0, NULL, CREATE_NEW,
                                    FILE_ATTRIBUTE_NORMAL, NULL);
        if (output != INVALID_HANDLE_VALUE) {
            BOOL ok = WriteFile(output, data, size, &written, NULL);
            BOOL closed = CloseHandle(output);
            if (ok && closed && written == size) status = 0;
        }
    }
    printf("{\"event\":\"coefficients\",\"loaded\":true,\"engine\":%u,"
           "\"camera\":%u,\"size_bytes\":%u,\"written_bytes\":%lu,\"success\":%s}\n",
           engine, camera, size, written, status == 0 ? "true" : "false");
    release(table);
    FreeLibrary(module);
    return status;
}
