/* Research-only bridge from live SDK image controls to vendor-owned Kaiji. */
#ifndef NXC_IMAGE_KAIJI_H
#define NXC_IMAGE_KAIJI_H

static int kaiji_guards(uint8_t *const buffers[5], const size_t sizes[5]) {
    int intact = 1;
    for (unsigned i = 0; i < 5; ++i) {
        if (!buffers[i]) continue;
        for (unsigned j = 0; j < 32; ++j)
            intact &= buffers[i][j] == 0xa5 && buffers[i][32 + sizes[i] + j] == 0xa5;
    }
    return intact;
}

static int serialize_kaiji(HMODULE dll, uint8_t *engine, const char *prefix) {
    typedef void *(*Serialize)(void *, void *, uint64_t);
    Serialize serialize;
    void *address = (uint8_t *)dll + 0x28d510;
    memcpy(&serialize, &address, sizeof(serialize));
    uint8_t *parameter;
    memcpy(&parameter, engine + 0xde0, sizeof(parameter));
    if (!parameter) return 0;
    uint64_t lengths[2], capacities[2];
    const uint8_t *text[2];
    for (unsigned i = 0; i < 2; ++i) {
        const uint8_t *string = parameter + 0xdf8 + i * 32;
        memcpy(&lengths[i], string + 16, 8);
        memcpy(&capacities[i], string + 24, 8);
        text[i] = string;
        if (capacities[i] > 15) memcpy(&text[i], string, sizeof(text[i]));
        if (!text[i] || lengths[i] > 254 || lengths[i] > capacities[i]) return 0;
    }
    uint64_t size = UINT64_C(33) * 33 * 33 * 6 + 0x6218
                  + ((lengths[1] + 12) & ~UINT64_C(3))
                  + ((lengths[0] * 2 + 28) & ~UINT64_C(3));
    uint8_t *allocation = malloc((size_t)size + 64);
    if (!allocation) return 0;
    memset(allocation, 0xa5, (size_t)size + 64);
    memset(allocation + 32, 0, (size_t)size);
    /* CCMSUtilities computes and stores this size before invoking the serializer. */
    memcpy(engine + 0xe20, &size, sizeof(size));
    void *error = serialize(engine, allocation + 32, size);
    int guards = 1;
    for (unsigned i = 0; i < 32; ++i)
        guards &= allocation[i] == 0xa5 && allocation[32 + size + i] == 0xa5;
    int success = !error && guards && save(prefix, "profile", allocation + 32, (DWORD)size);
    printf("{\"event\":\"kaiji_profile\",\"error\":%s,\"guards_intact\":%s,"
           "\"saved\":%s,\"bytes\":%llu,\"description_hex\":\"",
           error ? "true" : "false", guards ? "true" : "false",
           success ? "true" : "false", (unsigned long long)size);
    print_hex(text[0], (size_t)lengths[0]);
    printf("\",\"copyright_hex\":\"");
    print_hex(text[1], (size_t)lengths[1]);
    puts("\"}");
    fflush(stdout);
    if (error) {
        typedef void (*DestroyError)(void *);
        DestroyError destroy;
        address = slot(error, 0); memcpy(&destroy, &address, sizeof(destroy));
        destroy(error);
    }
    free(allocation);
    return success;
}

static int prepare_kaiji(HMODULE dll, uint8_t *engine, const char *prefix) {
    typedef void *(*Prepare)(void *, void *);
    Prepare prepare;
    void *address = (uint8_t *)dll + 0x2ae7b0;
    memcpy(&prepare, &address, sizeof(prepare));
    uint32_t grid, selector, flags;
    memcpy(&grid, engine + 0xdd4, 4);
    memcpy(&selector, engine + 0x9c0, 4);
    memcpy(&flags, engine + 0x9c4, 4);
    if (grid != 33 || selector != 0 || flags != 0) return 0;
    void *error = prepare(engine, NULL);
    int saved = 0;
    if (!error) {
        saved = save(prefix, "prepared-table", engine + 0xe70, 0x34370);
        for (unsigned i = 0; i < 3; ++i) {
            uint8_t *plane;
            memcpy(&plane, engine + 0xde8 + i * 16, sizeof(plane));
            char name[16];
            snprintf(name, sizeof(name), "plane%u", i);
            saved &= plane && save(prefix, name, plane, 33 * 33 * 33 * 2);
        }
    }
    printf("{\"event\":\"kaiji_prepared\",\"error\":%s,\"grid\":%u,"
           "\"selector\":%u,\"output_flags\":%u,\"saved\":%s}\n",
           error ? "true" : "false", grid, selector, flags, saved ? "true" : "false");
    fflush(stdout);
    if (error) {
        typedef void (*DestroyError)(void *);
        DestroyError destroy;
        address = slot(error, 0); memcpy(&destroy, &address, sizeof(destroy));
        destroy(error);
    }
    return !error && saved && serialize_kaiji(dll, engine, prefix);
}

static int collect_kaiji(HMODULE dll, void *image, void *services, const char *prefix,
                         int prepare_grid) {
    typedef void (*BuildRequest)(void *, void *, const char *);
    typedef void *(*ConstructPicture)(void *);
    typedef uint8_t (*BuildConfig)(void *, void *, void *, void *, uint32_t);
    typedef void *(*GetUtilities)(void *);
    typedef uint32_t (*SelectEngine)(uint8_t, uint32_t, uint32_t);
    typedef void *(*Factory)(void *, const void *, uint32_t, uint8_t, uint8_t);
    typedef void (*Generate)(void *, void *, void *, uint32_t);
    typedef void (*Destructor)(void *);
    const size_t sizes[5] = {0x2f0, 0xaa0, 0xdc8, 16, 0x34370};
    uint8_t *buffers[5] = {0}, *data[5] = {0}, *control = NULL;
    void *cms = NULL;
    int constructed = 0, guards = 1, saved = 0;
    const char *failure = NULL;
    Destructor destroy_picture;
    void *address = (uint8_t *)dll + 0x12f390;
    memcpy(&destroy_picture, &address, sizeof(destroy_picture));

    Request request_service;
    address = slot(services, 8);
    memcpy(&request_service, &address, sizeof(request_service));
    int result = request_service(services, (uint8_t *)dll + 0x789f00, &cms);
    printf("{\"event\":\"kaiji_cms\",\"result\":%d,\"present\":%s}\n",
           result, cms ? "true" : "false");
    fflush(stdout);
    if (result || !cms) { failure = "cms_service"; goto cleanup; }

    GetUtilities get_utilities;
    address = slot(cms, 0x70);
    memcpy(&get_utilities, &address, sizeof(get_utilities));
    void *utilities = get_utilities(cms);
    HMODULE renderer = GetModuleHandleA("image_render.n5m");
    if (!utilities) { failure = "cms_utilities"; goto cleanup; }
    if (!renderer) { failure = "renderer_module"; goto cleanup; }

    BuildRequest build_request;
    ConstructPicture construct;
    BuildConfig build_config;
    SelectEngine selector;
    Factory factory;
    Generate generate;
    address = (uint8_t *)renderer + 0x48fa20;
    memcpy(&build_request, &address, sizeof(build_request));
    address = (uint8_t *)dll + 0x12f1d0;
    memcpy(&construct, &address, sizeof(construct));
    address = (uint8_t *)dll + 0x19b7d0;
    memcpy(&build_config, &address, sizeof(build_config));
    address = (uint8_t *)dll + 0x20a700;
    memcpy(&selector, &address, sizeof(selector));
    address = (uint8_t *)dll + 0x28fe80;
    memcpy(&factory, &address, sizeof(factory));
    address = (uint8_t *)dll + 0x29f0b0;
    memcpy(&generate, &address, sizeof(generate));

    for (unsigned i = 0; i < 5; ++i) {
        buffers[i] = malloc(sizes[i] + 64);
        if (!buffers[i]) { failure = "buffer_allocation"; goto cleanup; }
        memset(buffers[i], 0xa5, sizes[i] + 64);
        data[i] = buffers[i] + 32;
        memset(data[i], 0, sizes[i]);
    }
    build_request(image, data[0], NULL);
    if (!kaiji_guards(buffers, sizes)) { failure = "request_guards"; goto cleanup; }
    construct(data[1]);
    constructed = 1;
    if (!kaiji_guards(buffers, sizes)) { failure = "picture_guards"; goto cleanup; }

    /* The builder defines AL only; the upper return-register bits are not status. */
    result = build_config(utilities, data[0], data[2], data[1], 0);
    printf("{\"event\":\"kaiji_config\",\"result\":%d}\n", result);
    fflush(stdout);
    if (result) { failure = "configuration"; goto cleanup; }
    if (!kaiji_guards(buffers, sizes)) { failure = "configuration_guards"; goto cleanup; }
    uint32_t profile, variant;
    memcpy(&profile, data[2] + 4, sizeof(profile));
    memcpy(&variant, data[2] + 0x120, sizeof(variant));
    uint32_t selected = selector(data[2][0x90], profile, variant);
    printf("{\"event\":\"kaiji_engine\",\"value\":%u}\n", selected);
    fflush(stdout);
    if (selected != 9) { failure = "unsupported_engine"; goto cleanup; }

    factory(data[3], data[2], 32, 0, 0);
    uint8_t *wrapper, *kaiji;
    memcpy(&wrapper, data[3], sizeof(wrapper));
    memcpy(&control, data[3] + 8, sizeof(control));
    if (!kaiji_guards(buffers, sizes)) { failure = "factory_guards"; goto cleanup; }
    if (!wrapper || !control) { failure = "factory_shared_pair"; goto cleanup; }
    memcpy(&kaiji, wrapper + 0x18, sizeof(kaiji));
    if (!kaiji) { failure = "factory_parameters"; goto cleanup; }

    saved = save(prefix, "request", data[0], 0x2f0);
    saved &= save(prefix, "config", data[2], 0xdc8);
    saved &= save(prefix, "kaiji", kaiji, 0x17f8);
    generate(kaiji, NULL, data[4], 0);
    saved &= save(prefix, "table", data[4], 0x34370);
    guards = kaiji_guards(buffers, sizes);
    printf("{\"event\":\"kaiji_tables\",\"guards_intact\":%s,\"saved\":%s}\n",
           guards ? "true" : "false", saved ? "true" : "false");
    fflush(stdout);
    if (!guards) failure = "table_guards";
    else if (!saved) failure = "snapshot_write";
    if (!failure && prepare_grid) {
        uint8_t *engine;
        memcpy(&engine, wrapper + 8, sizeof(engine));
        if (!engine || !prepare_kaiji(dll, engine, prefix)) failure = "engine_prepare";
    }

cleanup:
    /* Shared engine destruction still needs the copied picture and live image. */
    if (control && InterlockedDecrement((LONG *)(control + 8)) == 0) {
        Destructor destroy;
        address = slot(control, 0);
        memcpy(&destroy, &address, sizeof(destroy));
        destroy(control);
        if (InterlockedDecrement((LONG *)(control + 12)) == 0) {
            address = slot(control, 8);
            memcpy(&destroy, &address, sizeof(destroy));
            destroy(control);
        }
    }
    if (constructed) destroy_picture(data[1]);
    if (cms) {
        Release release;
        address = slot(cms, 0x18);
        memcpy(&release, &address, sizeof(release));
        release(cms);
    }
    guards &= kaiji_guards(buffers, sizes);
    for (unsigned i = 0; i < 5; ++i) free(buffers[i]);
    if (failure) printf("{\"event\":\"kaiji_failure\",\"stage\":\"%s\"}\n", failure);
    printf("{\"event\":\"kaiji_released\",\"guards_intact\":%s}\n",
           guards ? "true" : "false");
    fflush(stdout);
    return saved && guards && !failure;
}

#endif
