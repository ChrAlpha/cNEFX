/* Research-only NP3 custom-curve import through the Kaiji table generator. */
#include <windows.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef void *(*ReadParameters)(const void *, uint64_t, void *, void *, void *, void *, void *);
typedef uint64_t (*LoadCurve)(void *, const void *);
typedef void *(*GetCurve)(void *, int);
typedef void (*DestroyGroup)(void *, unsigned);
typedef void (*GenerateTables)(void *, void *, void *);
typedef uint64_t (*NormalizeFlexible)(const void *, int32_t *);
typedef int32_t (*RoundDivide)(int32_t, int32_t);
enum { GUARD = 32, STATE_BYTES = 0x17f8, TABLE_BYTES = 0x34370 };

static uint8_t *guarded(size_t size) {
    uint8_t *allocation = malloc(size + 2 * GUARD);
    if (!allocation) return NULL;
    memset(allocation, 0xa5, size + 2 * GUARD);
    memset(allocation + GUARD, 0, size);
    return allocation + GUARD;
}

static int intact(const uint8_t *data, size_t size) {
    for (int i = 0; i < GUARD; ++i)
        if (data[i - GUARD] != 0xa5 || data[size + i] != 0xa5) return 0;
    return 1;
}

int main(int argc, char **argv) {
    if ((argc != 5 && !(argc == 6 && !strcmp(argv[5], "normalized")))
        || (strcmp(argv[4], "neutral") && strcmp(argv[4], "extreme")))
        return 2;
    int normalized = argc == 6;
    SetErrorMode(SEM_FAILCRITICALERRORS | SEM_NOGPFAULTERRORBOX);
    FILE *stream = fopen(argv[2], "rb");
    if (!stream || fseek(stream, 0, SEEK_END)) return 3;
    long size = ftell(stream);
    if (size < 16 || size > 4194304 || fseek(stream, 0, SEEK_SET)) return 3;
    uint8_t *input = guarded((size_t)size);
    if (!input || fread(input, 1, (size_t)size, stream) != (size_t)size) return 3;
    if (fgetc(stream) != EOF || ferror(stream)) return 3;
    fclose(stream);
    uint8_t *base = guarded(108), *curve = guarded(578), *flex = guarded(88);
    uint8_t *comment = guarded(2000), *extra = guarded(256), *owner = guarded(0x1000);
    uint8_t *state = guarded(STATE_BYTES), *table = guarded(TABLE_BYTES);
    uint8_t *group_view = guarded(0x20), *curve_list = guarded(sizeof(void *));
    uint8_t *normalized_data = guarded(41 * sizeof(int32_t));
    if (!base || !curve || !flex || !comment || !extra || !owner || !state || !table
        || !group_view || !curve_list || !normalized_data) return 3;
    HMODULE module = LoadLibraryA(argv[1]);
    if (!module) return 4;
    void *address = (uint8_t *)module + 0x137080;
    ReadParameters read_parameters;
    memcpy(&read_parameters, &address, sizeof(read_parameters));
    if (read_parameters(input, (uint64_t)size, base, curve, flex, comment, extra)) return 5;
    if (curve[8] < 2 || curve[8] > 20) return 5;
    LoadCurve load_curve;
    address = (uint8_t *)module + 0x132770;
    memcpy(&load_curve, &address, sizeof(load_curve));
    uint64_t status = load_curve(owner, curve);
    void *group = *(void **)(owner + 0x80);
    if (status || !group) return 5;
    uintptr_t *vtable = *(uintptr_t **)group;
    GetCurve get_curve;
    memcpy(&get_curve, &vtable[0x50 / 8], sizeof(get_curve));
    void *object = get_curve(group, 0);
    if (!object) return 5;
    /* The public sample wrapper forwards its internal curve pointer at +0x20. */
    void *raw_curve = *(void **)((uint8_t *)object + 0x20);
    if (!raw_curve) return 5;
    memcpy(curve_list, &raw_curve, sizeof(raw_curve));
    memcpy(group_view + 8, &curve_list, sizeof(curve_list));
    *(int32_t *)(group_view + 0x18) = 1;
    memcpy(state + 0x678, &group_view, sizeof(group_view));
    state[0x1756] = 1;
    *(uint32_t *)(state + 0x0c) = 1071;
    *(uint16_t *)(state + 0x1518) = 14;
    *(uint32_t *)(state + 0x151c) = 12;
    *(uint16_t *)(state + 0x1520) = 256;
    *(uint16_t *)(state + 0x17ee) = 1;
    *(uint16_t *)(state + 0x17f0) = 2;
    uint64_t normalization_return = 0;
    int32_t *values = (int32_t *)normalized_data;
    if (normalized) {
        if (memcmp(base, "0310", 4) || *(uint16_t *)(base + 48) != 32) return 5;
        NormalizeFlexible normalize;
        address = (uint8_t *)module + 0x131b90;
        memcpy(&normalize, &address, sizeof(normalize));
        normalization_return = normalize(flex, values);
        for (int i = 0; i < 41; ++i) {
            int low = -100, high = 100;
            if (i == 5) { low = -50; high = 50; }
            if (i >= 30 && i <= 38 && (i - 30) % 3 == 0) { low = 0; high = 360; }
            if ((i >= 30 && i <= 38 && (i - 30) % 3 == 1) || i == 39) low = 0;
            if (values[i] < low || values[i] > high) return 5;
        }
        RoundDivide divide;
        address = (uint8_t *)module + 0x29d8e0;
        memcpy(&divide, &address, sizeof(divide));
        *(int16_t *)(state + 0x1548) = (int16_t)divide(values[0] * 64, 25);
        state[0x154a] = 6;
        state[0x1522] = (uint8_t)(divide(values[5] * 6, 25) + 12);
        state[0x1526] = 16;
        *(uint16_t *)(state + 0x1538) = 128;
        *(int16_t *)(state + 0x1758) = (int16_t)divide(values[1] * 6, 5);
        *(int16_t *)(state + 0x175a) = (int16_t)divide(values[2] * 6, 5);
        *(int16_t *)(state + 0x175c) = (int16_t)values[3];
        *(int16_t *)(state + 0x175e) = (int16_t)values[4];
        *(int16_t *)(state + 0x1760) = (int16_t)values[5];
        memcpy(state + 0x1764, values + 6, 24 * sizeof(int32_t));
        memcpy(state + 0x17c4, values + 30, 9 * sizeof(int32_t));
        *(uint16_t *)(state + 0x17e8) = (uint16_t)values[39];
        *(int16_t *)(state + 0x17ea) = (int16_t)values[40];
    }
    if (!strcmp(argv[4], "extreme")) {
        const int16_t tone[4] = {120, -120, 100, -100};
        memcpy(state + 0x1758, tone, sizeof(tone));
    }
    GenerateTables generate;
    address = (uint8_t *)module + 0x29f0b0;
    memcpy(&generate, &address, sizeof(generate));
    generate(state, NULL, table);
    char moon_path[MAX_PATH];
    HMODULE moon = GetModuleHandleA("Moon2.dll");
    DWORD moon_length = moon ? GetModuleFileNameA(moon, moon_path, MAX_PATH) : 0;
    if (!moon_length || moon_length >= MAX_PATH) return 7;
    for (DWORD i = 0; i < moon_length; ++i) {
        if (moon_path[i] == '\\') moon_path[i] = '/';
        if (moon_path[i] == '"' || (unsigned char)moon_path[i] < 32) return 7;
    }
    int guards = intact(input, (size_t)size) && intact(base, 108) && intact(curve, 578)
        && intact(flex, 88) && intact(comment, 2000) && intact(extra, 256)
        && intact(owner, 0x1000) && intact(state, STATE_BYTES) && intact(table, TABLE_BYTES)
        && intact(group_view, 0x20) && intact(curve_list, sizeof(void *))
        && intact(normalized_data, 41 * sizeof(int32_t));
    HANDLE file = CreateFileA(argv[3], GENERIC_WRITE, 0, NULL, CREATE_NEW,
                              FILE_ATTRIBUTE_NORMAL, NULL);
    DWORD written = 0;
    int success = 0;
    if (file != INVALID_HANDLE_VALUE) {
        success = WriteFile(file, table, TABLE_BYTES, &written, NULL) && written == TABLE_BYTES;
        if (!CloseHandle(file)) success = 0;
    }
    printf("{\"event\":\"kaiji_custom_curve\",\"mode\":\"%s\",\"status\":%llu,"
           "\"normalized\":%s,\"normalization_return\":%llu,"
           "\"normalized_values\":[",
           argv[4], (unsigned long long)status, normalized ? "true" : "false",
           (unsigned long long)normalization_return);
    for (int i = 0; i < 41; ++i) printf("%s%d", i ? "," : "", values[i]);
    printf("],"
           "\"moon_path\":\"%s\",\"guards_intact\":%s,\"output_bytes\":%lu,\"success\":%s}\n",
           moon_path, guards ? "true" : "false", written,
           success ? "true" : "false");
    DestroyGroup destroy;
    memcpy(&destroy, &vtable[0x118 / 8], sizeof(destroy));
    destroy(group, 1);
    uint8_t *allocations[] = {input, base, curve, flex, comment, extra, owner,
                              state, table, group_view, curve_list, normalized_data};
    for (unsigned i = 0; i < sizeof(allocations) / sizeof(allocations[0]); ++i)
        free(allocations[i] - GUARD);
    FreeLibrary(module);
    return guards && success ? 0 : 6;
}
