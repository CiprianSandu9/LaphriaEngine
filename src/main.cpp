// TODO: Move STB_IMAGE_IMPLEMENTATION to a dedicated stb_image.cpp translation unit
// to decouple the application entry point from image-loading compilation.
#define STB_IMAGE_IMPLEMENTATION
#include "Core/EngineAuxiliary.h"
#include "Core/EngineCore.h"

int main() {
    try {
        EngineCore app;
        app.run();
    } catch (const std::exception &e) {
        LOGE("%s", e.what());
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}
