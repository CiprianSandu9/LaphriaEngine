#include "Core/EngineAuxiliary.h"
#include "Core/EngineHost.h"

int main() {
    try {
        EngineHost app;
        app.run();
    } catch (const std::exception &e) {
        LOGE("%s", e.what());
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}
