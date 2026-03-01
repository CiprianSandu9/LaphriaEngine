# Findimgui.cmake
#
# Finds the Dear ImGui library (optimized for vcpkg)
#
# This provides the following imported targets:
#    imgui::imgui
#

# 1. Search for the package using modern CMake config mode
# When using vcpkg, this will look for imgui-config.cmake provided by the port
find_package(imgui CONFIG QUIET)

# 2. Fallback: If not found via CONFIG, try simple MODULE search
if (NOT imgui_FOUND)
    find_package(imgui QUIET)
endif ()

# 3. Handle the results
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(imgui
        REQUIRED_VARS imgui_DIR
        HANDLE_COMPONENTS
)

# Note: vcpkg's imgui defines its own imgui::imgui target.
# Be aware that you may still need to link specific backends
# (e.g., GLFW, SDL2, Vulkan) depending on your vcpkg features.