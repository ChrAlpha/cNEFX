/* Research-only CML interleaved RGB/Lab16 pixel oracle. */
#include <windows.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef int (*Create)(void **);
typedef int (*CreateTransformer)(void **, void *);
typedef int (*SetProfile)(void *, const void *, uint32_t);
typedef int (*SetNumber)(void *, uint32_t);
typedef int (*SetTransform)(void *, void *);
typedef int (*Destroy)(void *);
typedef int (*Process)(void *, int, int, const void *, void *, intptr_t, intptr_t);

typedef struct { uint8_t *allocation, *data, *copy; uint32_t size; } Buffer;

static int read_buffer(const char *name, Buffer *buffer) {
    FILE *file = fopen(name, "rb");
    if (!file) return 0;
    if (fseek(file, 0, SEEK_END)) { fclose(file); return 0; }
    long length = ftell(file);
    if (length <= 0 || length > 16 * 1024 * 1024 || fseek(file, 0, SEEK_SET)) {
        fclose(file); return 0;
    }
    buffer->size = (uint32_t)length;
    buffer->allocation = malloc((size_t)length + 64);
    buffer->copy = malloc((size_t)length);
    if (!buffer->allocation || !buffer->copy) { fclose(file); return 0; }
    memset(buffer->allocation, 0xa5, (size_t)length + 64);
    buffer->data = buffer->allocation + 32;
    int complete = fread(buffer->data, 1, (size_t)length, file) == (size_t)length
                && fgetc(file) == EOF && !ferror(file);
    fclose(file);
    if (complete) memcpy(buffer->copy, buffer->data, (size_t)length);
    return complete;
}

static int intact(const Buffer *buffer, int unchanged) {
    if (!buffer->allocation) return 0;
    int ok = !unchanged || !memcmp(buffer->data, buffer->copy, buffer->size);
    for (unsigned i = 0; i < 32; ++i)
        ok &= buffer->allocation[i] == 0xa5 && buffer->data[buffer->size + i] == 0xa5;
    return ok;
}

#define LOAD(var, name) do { FARPROC address = GetProcAddress(dll, name); \
    if (!address) { failure = "missing_export"; goto cleanup; } \
    memcpy(&var, &address, sizeof(var)); } while (0)
#define STEP(name, expression) do { int result = (expression); \
    printf("{\"event\":\"%s\",\"result\":%d}\n", name, result); fflush(stdout); \
    if (result) { failure = name; goto cleanup; } } while (0)

int main(int argc, char **argv) {
    if (argc != 6) return 2;
    SetErrorMode(SEM_FAILCRITICALERRORS | SEM_NOGPFAULTERRORBOX);
    HMODULE dll = LoadLibraryA(argv[1]);
    if (!dll) return 3;
    Create create;
    CreateTransformer create_transformer;
    SetProfile input_profile, output_profile;
    SetNumber intent, proof, input_bits, output_bits;
    SetTransform set_transform;
    Process process;
    Destroy destroy_transformer = NULL, destroy_xform = NULL;
    Buffer inputs[3] = {{0}}, output = {0};
    void *xform = NULL, *transformer = NULL;
    const char *failure = NULL;
    int saved = 0, guards = 0, released = 1;
    uint32_t pixels = 0;
    LOAD(destroy_transformer, "NKCMDestroyTransformer");
    LOAD(destroy_xform, "NKCMDestroyXForm");
    LOAD(create, "NKCMCreateXForm");
    LOAD(create_transformer, "NKCMCreateTransformer");
    LOAD(input_profile, "NKCMSetInputProfileBuf");
    LOAD(output_profile, "NKCMSetOutputProfileBuf");
    LOAD(intent, "NKCMSetIntent");
    LOAD(proof, "NKCMSetProofMode");
    LOAD(input_bits, "NKCMSetInputBitSize");
    LOAD(output_bits, "NKCMSetOutputBitSize");
    LOAD(set_transform, "NKCMSetTransform");
    LOAD(process, "NKCMProcessInterleavedData");
    char path[MAX_PATH];
    DWORD length = GetModuleFileNameA(dll, path, MAX_PATH);
    if (!length || length >= MAX_PATH) { failure = "provider_path"; goto cleanup; }
    printf("{\"event\":\"provider\",\"path_hex\":\"");
    for (DWORD i = 0; i < length; ++i) printf("%02x", (unsigned char)path[i]);
    puts("\"}");
    for (unsigned i = 0; i < 3; ++i)
        if (!read_buffer(argv[2 + i], &inputs[i])) { failure = "read_input"; goto cleanup; }
    if (inputs[2].size % 6) { failure = "pixel_size"; goto cleanup; }
    pixels = inputs[2].size / 6;
    output.size = inputs[2].size;
    output.allocation = malloc(output.size + 64);
    if (!output.allocation) { failure = "output_allocation"; goto cleanup; }
    memset(output.allocation, 0xa5, output.size + 64);
    output.data = output.allocation + 32;
    STEP("xform_created", create(&xform));
    if (!xform) { failure = "null_xform"; goto cleanup; }
    STEP("input_profile", input_profile(xform, inputs[0].data, inputs[0].size));
    STEP("output_profile", output_profile(xform, inputs[1].data, inputs[1].size));
    STEP("intent", intent(xform, 0));
    STEP("proof", proof(xform, 0));
    STEP("transformer_created", create_transformer(&transformer, NULL));
    if (!transformer) { failure = "null_transformer"; goto cleanup; }
    STEP("transform_set", set_transform(transformer, xform));
    STEP("input_bits", input_bits(transformer, 16));
    STEP("output_bits", output_bits(transformer, 16));
    STEP("processed", process(transformer, (int)pixels, 1, inputs[2].data,
                              output.data, inputs[2].size, output.size));
    guards = intact(&output, 0);
    for (unsigned i = 0; i < 3; ++i) guards &= intact(&inputs[i], 1);
    if (!guards) { failure = "buffer_guards"; goto cleanup; }
    HANDLE target = CreateFileA(argv[5], GENERIC_WRITE, 0, NULL, CREATE_NEW, 0, NULL);
    DWORD written = 0;
    if (target != INVALID_HANDLE_VALUE) {
        saved = WriteFile(target, output.data, output.size, &written, NULL) && written == output.size;
        CloseHandle(target);
    }
    if (!saved) failure = "write_output";
cleanup:
    if (transformer) released &= destroy_transformer(transformer) == 0;
    if (xform) released &= destroy_xform(xform) == 0;
    if (guards) {
        guards &= intact(&output, 0);
        for (unsigned i = 0; i < 3; ++i) guards &= intact(&inputs[i], 1);
    }
    if (failure) printf("{\"event\":\"failure\",\"stage\":\"%s\"}\n", failure);
    printf("{\"event\":\"complete\",\"pixels\":%u,\"saved\":%s,"
           "\"inputs_and_guards_intact\":%s,\"released\":%s}\n", pixels,
           saved ? "true" : "false", guards ? "true" : "false", released ? "true" : "false");
    for (unsigned i = 0; i < 3; ++i) { free(inputs[i].allocation); free(inputs[i].copy); }
    free(output.allocation);
    FreeLibrary(dll);
    return !failure && saved && guards && released ? 0 : 6;
}
