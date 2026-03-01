# Findfastgltf.cmake
#
# Finds the fastgltf library (optimized for vcpkg)
#
# This provides the following imported targets:
#    fastgltf::fastgltf
#

# 1. Search for the package using modern CMake config mode
# When using vcpkg, this will look for fastgltfConfig.cmake provided by the port
find_package(fastgltf CONFIG QUIET)

# 2. Fallback: Removed recursive MODULE search to avoid infinite loops
# if (NOT fastgltf_FOUND)
#     find_package(fastgltf QUIET)
# endif ()

# 3. Handle the results
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(fastgltf
        REQUIRED_VARS fastgltf_DIR
        HANDLE_COMPONENTS
)

# Note: fastgltf defines its own fastgltf::fastgltf target.
# There is no need to manually create INTERFACE libraries or 
# link nlohmann_json, as fastgltf handles its own dependencies.