/* Bounded RGB16 render target for the pinned SDK research probe. */
#ifndef NXC_IMAGE_RENDER_H
#define NXC_IMAGE_RENDER_H

typedef struct {
    int32_t x, y, width, height;
    uint32_t format, reserved14;
    const uint8_t *pixels;
    int64_t stride;
    uint64_t reserved28;
} NxcRenderData;
typedef struct NxcRenderTarget NxcRenderTarget;
typedef struct {
    uint32_t (*receive)(NxcRenderTarget *, const NxcRenderData *);
    void (*release)(NxcRenderTarget *);
    void *(*destroy)(NxcRenderTarget *, uint32_t);
} NxcRenderVtable;
struct NxcRenderTarget {
    NxcRenderVtable *vtable;
    CRITICAL_SECTION lock;
    uint8_t *pixels, *coverage;
    int32_t width, height;
    uint32_t calls, covered, overlaps, rejected, released, destroyed;
};

static int render_readable(const uint8_t *data, size_t size) {
    uintptr_t cursor = (uintptr_t)data;
    if (!cursor || size > UINTPTR_MAX - cursor) return 0;
    uintptr_t end = cursor + size;
    while (cursor < end) {
        MEMORY_BASIC_INFORMATION info;
        if (!VirtualQuery((const void *)cursor, &info, sizeof(info))
            || info.State != MEM_COMMIT || (info.Protect & (PAGE_GUARD | PAGE_NOACCESS)))
            return 0;
        DWORD mode = info.Protect & 0xff;
        if (mode != PAGE_READONLY && mode != PAGE_READWRITE && mode != PAGE_WRITECOPY
            && mode != PAGE_EXECUTE_READ && mode != PAGE_EXECUTE_READWRITE
            && mode != PAGE_EXECUTE_WRITECOPY) return 0;
        uintptr_t next = (uintptr_t)info.BaseAddress + info.RegionSize;
        if (next <= cursor) return 0;
        cursor = next < end ? next : end;
    }
    return 1;
}

static uint32_t render_receive(NxcRenderTarget *target, const NxcRenderData *data) {
    EnterCriticalSection(&target->lock);
    int valid = data && !target->released && target->pixels && target->coverage;
    if (valid) {
        valid = data->format == 1 && data->x >= 0 && data->y >= 0
            && data->width > 0 && data->height > 0
            && data->width <= target->width && data->height <= target->height
            && data->x <= target->width - data->width
            && data->y <= target->height - data->height
            && data->stride >= (int64_t)data->width * 6
            && data->stride <= 64 * 1024 * 1024;
    }
    if (valid) {
        uint64_t span = (uint64_t)(data->height - 1) * data->stride + data->width * 6;
        valid = span <= 64 * 1024 * 1024 && render_readable(data->pixels, (size_t)span);
    }
    if (valid) {
        for (int32_t y = 0; y < data->height; ++y) {
            size_t index = (size_t)(data->y + y) * target->width + data->x;
            for (int32_t x = 0; x < data->width; ++x) {
                if (target->coverage[index + x]) ++target->overlaps;
            }
        }
        if (target->overlaps) valid = 0;
    }
    if (valid) {
        for (int32_t y = 0; y < data->height; ++y) {
            size_t index = (size_t)(data->y + y) * target->width + data->x;
            memcpy(target->pixels + index * 6, data->pixels + y * data->stride,
                   (size_t)data->width * 6);
            memset(target->coverage + index, 1, (size_t)data->width);
        }
        target->covered += (uint32_t)data->width * data->height;
        ++target->calls;
    } else ++target->rejected;
    LeaveCriticalSection(&target->lock);
    return valid ? 0 : 0x15;
}

static void render_release_target(NxcRenderTarget *target) {
    EnterCriticalSection(&target->lock);
    ++target->released;
    LeaveCriticalSection(&target->lock);
}

static void *render_destroy_target(NxcRenderTarget *target, uint32_t flags) {
    (void)flags;
    EnterCriticalSection(&target->lock);
    ++target->destroyed;
    LeaveCriticalSection(&target->lock);
    return target;
}

static int render_guards(const uint8_t *allocation, size_t size) {
    for (unsigned i = 0; i < 32; ++i)
        if (allocation[i] != 0xa5 || allocation[32 + size + i] != 0xa5) return 0;
    return 1;
}

static void render_abort(HMODULE dll, const char *stage) {
    printf("{\"event\":\"render_failure\",\"stage\":\"%s\"}\n", stage);
    dump_log(dll);
    fflush(stdout);
    /* Registered callbacks may still own the target; process exit owns cleanup. */
    ExitProcess(11);
}

#define RENDER_BIND(var, object, offset) do { void *p = slot(object, offset); \
    memcpy(&var, &p, sizeof(var)); } while (0)
#define RENDER_STEP(name, expression) do { int result_ = (expression); \
    printf("{\"event\":\"%s\",\"result\":%d}\n", name, result_); fflush(stdout); \
    if (result_) { failure = name; goto cleanup; } } while (0)

static int collect_render(HMODULE dll, void *image, void *services, const char *prefix) {
    typedef int (*Compose)(void *, void *, const void *, void **, int);
    typedef int (*Setup)(void *, void *, void *, int, int);
    typedef int (*SetTarget)(void *, void *);
    typedef int (*SetScale)(void *, double);
    typedef int (*SetBits)(void *, uint32_t);
    typedef int (*SetProfile)(void *, const char *);
    typedef int (*GetSize)(void *, int32_t *, int32_t *);
    const uint8_t render_id[24] = {0x0a,0xea,0x59,0x8d,0x9e,0xa3,0xf4,0x4c,
                                  0x84,0x1e,0x1b,0x4a,0xe4,0xd7,0x16,0x34};
    const uint8_t composer_id[24] = {0x70,0x9d,0x50,0x39,0xcc,0x85,0xdd,0x11,
                                    0xad,0x8b,0x08,0x00,0x20,0x0c,0x9a,0x66};
    const uint8_t descriptor[24] = {0xfa,0xa4,0x5f,0x05,0xd8,0x8f,0x0e,0x4f,
                                   0x9e,0x0f,0xa2,0xbb,0xa3,0x24,0x33,0xc9};
    void *render = NULL, *composer = NULL, *filter = NULL;
    uint8_t *pixels = NULL, *coverage = NULL;
    struct { uint64_t before[4]; NxcRenderTarget target; uint8_t after[32]; } box;
    struct { uint64_t before[4]; int32_t width, height; uint8_t after[32]; } size;
    memset(&box, 0xa5, sizeof(box));
    memset(&box.target, 0, sizeof(box.target));
    memset(&size, 0xa5, sizeof(size));
    size.width = size.height = 0;
    NxcRenderVtable vtable = {render_receive, render_release_target, render_destroy_target};
    box.target.vtable = &vtable;
    InitializeCriticalSection(&box.target.lock);
    Release release_render = NULL, release_composer = NULL, release_filter = NULL;
    Request request;
    Compose compose;
    Setup setup;
    SetTarget set_target;
    SetScale set_scale;
    SetBits set_bits;
    SetProfile set_profile;
    GetSize get_size;
    Step start, wait;
    int registered = 0, success = 0;
    const char *failure = NULL;
    char profile[260];
    DWORD length = GetEnvironmentVariableA("NXC_RENDER_PROFILE", profile, sizeof(profile));
    if (!length || length >= sizeof(profile)) { failure = "profile_path"; goto cleanup; }
    RENDER_BIND(request, services, 8);
    RENDER_STEP("render_service", request(services, render_id, &render));
    if (!render) { failure = "null_renderer"; goto cleanup; }
    RENDER_BIND(release_render, render, 0x18);
    RENDER_STEP("composer_service", request(services, composer_id, &composer));
    if (!composer) { failure = "null_composer"; goto cleanup; }
    RENDER_BIND(release_composer, composer, 0x18);
    RENDER_BIND(compose, composer, 0x28);
    RENDER_BIND(setup, render, 0x50);
    RENDER_BIND(set_target, render, 0x80);
    RENDER_BIND(set_scale, render, 0x68);
    RENDER_BIND(set_bits, render, 0xe8);
    RENDER_BIND(set_profile, render, 0x120);
    RENDER_BIND(get_size, render, 0xa0);
    RENDER_BIND(start, render, 0x88);
    RENDER_BIND(wait, render, 0xb8);
    RENDER_STEP("render_composed", compose(composer, image, descriptor, &filter, 0));
    if (!filter) { failure = "null_filter"; goto cleanup; }
    RENDER_BIND(release_filter, filter, 0x18);
    RENDER_STEP("render_setup", setup(render, image, filter, 0, 2));
    /* CImageRenderService vtable+0x120 -> image_render RVA0x3652c0 takes a path. */
    RENDER_STEP("render_profile", set_profile(render, profile));
    printf("{\"event\":\"render_profile_path\",\"path_hex\":\"");
    print_hex((const uint8_t *)profile, length);
    puts("\"}");
    RENDER_STEP("render_scale", set_scale(render, 1.0 / 32.0));
    RENDER_STEP("render_bits", set_bits(render, 16));
    RENDER_STEP("render_size", get_size(render, &size.width, &size.height));
    if (!render_guards((const uint8_t *)&size, 8) || size.width <= 0 || size.height <= 0
        || size.width > 1024 || size.height > 1024) { failure = "dimensions"; goto cleanup; }
    size_t count = (size_t)size.width * size.height;
    pixels = malloc(count * 6 + 64);
    coverage = malloc(count + 64);
    if (!pixels || !coverage) { failure = "allocation"; goto cleanup; }
    memset(pixels, 0xa5, count * 6 + 64);
    memset(coverage, 0xa5, count + 64);
    memset(coverage + 32, 0, count);
    box.target.pixels = pixels + 32;
    box.target.coverage = coverage + 32;
    box.target.width = size.width;
    box.target.height = size.height;
    registered = 1;
    RENDER_STEP("render_target", set_target(render, &box.target));
    release_filter(filter); filter = NULL;
    release_composer(composer); composer = NULL;
    puts("{\"event\":\"render_dependencies_released\"}");
    RENDER_STEP("render_started", start(render));
    RENDER_STEP("render_waited", wait(render));
    release_render(render); render = NULL;
    puts("{\"event\":\"render_released\"}");
    EnterCriticalSection(&box.target.lock);
    size_t missing = 0;
    for (size_t i = 0; i < count; ++i) missing += coverage[32 + i] != 1;
    int guards = render_guards(pixels, count * 6) && render_guards(coverage, count)
        && render_guards((const uint8_t *)&box, sizeof(box.target))
        && render_guards((const uint8_t *)&size, 8);
    int metadata = box.target.vtable == &vtable && box.target.pixels == pixels + 32
        && box.target.coverage == coverage + 32 && box.target.width == size.width
        && box.target.height == size.height;
    success = guards && metadata && !missing && !box.target.overlaps && !box.target.rejected
        && box.target.covered == count && box.target.calls > 0 && box.target.calls <= count
        && box.target.released == 1 && box.target.destroyed == 0;
    if (success) success = save(prefix, "rgb16", pixels + 32, (DWORD)(count * 6));
    printf("{\"event\":\"render_complete\",\"width\":%d,\"height\":%d,\"format\":1,"
           "\"bits\":16,\"scale\":0.03125,\"bytes\":%llu,\"calls\":%u,"
           "\"covered\":%u,\"missing\":%llu,\"overlaps\":%u,\"rejected\":%u,"
           "\"target_released\":%u,\"target_destroyed\":%u,\"guards_intact\":%s,"
           "\"metadata_intact\":%s,\"saved\":%s}\n", size.width, size.height,
           (unsigned long long)(count * 6), box.target.calls, box.target.covered,
           (unsigned long long)missing, box.target.overlaps, box.target.rejected,
           box.target.released, box.target.destroyed, guards ? "true" : "false",
           metadata ? "true" : "false", success ? "true" : "false");
    LeaveCriticalSection(&box.target.lock);
    if (!success) { failure = "render_validation"; goto cleanup; }
cleanup:
    if (failure && registered) render_abort(dll, failure);
    if (filter && release_filter) release_filter(filter);
    if (composer && release_composer) release_composer(composer);
    if (render && release_render) release_render(render);
    if (failure) printf("{\"event\":\"render_failure\",\"stage\":\"%s\"}\n", failure);
    DeleteCriticalSection(&box.target.lock);
    free(coverage);
    free(pixels);
    fflush(stdout);
    return success;
}
#undef RENDER_BIND
#undef RENDER_STEP
#endif
