cmake_minimum_required(VERSION 3.29)
project(LaphriaEngine)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/CMake")

find_package(glfw3 REQUIRED)
find_package(glm REQUIRED)
find_package(Vulkan 1.4 REQUIRED)
find_package(fastgltf REQUIRED)
find_package(imgui REQUIRED)
find_package(KTX REQUIRED)
find_package(nlohmann_json REQUIRED)
find_package(stb REQUIRED)

find_program(SLANGC_EXECUTABLE slangc HINTS $ENV{VULKAN_SDK}/bin REQUIRED)

# Compile Slang shaders to SPIR-V
set(SHADER_SOURCES
    src/Shaders/LaphriaEngine.slang
    src/Shaders/Compute.slang
    src/Shaders/Shadow.slang
    src/Shaders/Physics.slang
    src/Shaders/ClosestHit.slang
    src/Shaders/AnyHit.slang
    src/Shaders/Miss.slang
    src/Shaders/Raygen.slang
)
# Multi-config generators (VS, Ninja Multi-Config) automatically append <Config>/ to
# RUNTIME_OUTPUT_DIRECTORY, so the exe lands in LaphriaEngine/<Config>/. Single-config
# generators (Ninja) do not, so the exe lands directly in LaphriaEngine/.
if(CMAKE_CONFIGURATION_TYPES)
    set(SHADERS_DIR "${CMAKE_BINARY_DIR}/LaphriaEngine/$<CONFIG>/Shaders")
else()
    set(SHADERS_DIR "${CMAKE_BINARY_DIR}/LaphriaEngine/Shaders")
endif()
set(GENERATED_SPV_FILES "")

foreach(SHADER_SOURCE ${SHADER_SOURCES})
    get_filename_component(SHADER_NAME ${SHADER_SOURCE} NAME)
    set(SHADER_OUTPUT "${SHADERS_DIR}/${SHADER_NAME}.spv")

    file(READ ${SHADER_SOURCE} SHADER_CONTENT)
    set(ENTRY_POINTS "")
    if(SHADER_CONTENT MATCHES "vertMain")
        list(APPEND ENTRY_POINTS -entry vertMain)
    endif()
    if(SHADER_CONTENT MATCHES "fragMain")
        list(APPEND ENTRY_POINTS -entry fragMain)
    endif()
    if(SHADER_CONTENT MATCHES "computeMain")
        list(APPEND ENTRY_POINTS -entry computeMain)
    endif()
    if(SHADER_CONTENT MATCHES "physicsMain")
        list(APPEND ENTRY_POINTS -entry physicsMain)
    endif()
    if(SHADER_CONTENT MATCHES "shadowVert")
        list(APPEND ENTRY_POINTS -entry shadowVert)
    endif()
    if(SHADER_CONTENT MATCHES "shadowFrag")
        list(APPEND ENTRY_POINTS -entry shadowFrag)
    endif()
    if(NOT ENTRY_POINTS)
        message(WARNING "No standard entry points found in ${SHADER_NAME}, defaulting to '-entry main'")
        list(APPEND ENTRY_POINTS -entry main)
    endif()

    add_custom_command(
        OUTPUT ${SHADER_OUTPUT}
        COMMAND ${CMAKE_COMMAND} -E make_directory ${SHADERS_DIR}
        COMMAND ${SLANGC_EXECUTABLE} ${SHADER_SOURCE}
            -target spirv -profile spirv_1_5+spvRayQueryKHR
            -emit-spirv-directly ${ENTRY_POINTS}
            -o ${SHADER_OUTPUT} -fvk-use-entrypoint-name
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
        DEPENDS ${SHADER_SOURCE}
        COMMENT "Compiling Slang shader: ${SHADER_NAME}"
        VERBATIM
    )
    list(APPEND GENERATED_SPV_FILES ${SHADER_OUTPUT})
endforeach()

add_custom_target(LaphriaEngine_shaders DEPENDS ${GENERATED_SPV_FILES})

add_executable(LaphriaEngine
    src/main.cpp
    src/Core/Camera.cpp
    src/Core/Camera.h
    src/Core/EngineCore.cpp
    src/Core/EngineCore.h
    src/Core/FrameContext.cpp
    src/Core/FrameContext.h
    src/Core/InputSystem.cpp
    src/Core/InputSystem.h
    src/Core/PipelineCollection.cpp
    src/Core/PipelineCollection.h
    src/Core/SwapchainManager.cpp
    src/Core/SwapchainManager.h
    src/Core/UISystem.cpp
    src/Core/UISystem.h
    src/Core/VulkanDevice.cpp
    src/Core/VulkanDevice.h
    src/Core/ResourceManager.cpp
    src/Core/ResourceManager.h
    src/Core/EngineAuxiliary.h
    src/Core/VulkanUtils.cpp
    src/Core/VulkanUtils.h
    src/Physics/PhysicsDefines.h
    src/Physics/PhysicsSystem.cpp
    src/Physics/PhysicsSystem.h
    src/SceneManagement/Scene.cpp
    src/SceneManagement/Scene.h
    src/SceneManagement/SceneNode.cpp
    src/SceneManagement/SceneNode.h
    src/SceneManagement/Octree.h
)

set_target_properties(LaphriaEngine PROPERTIES
    CXX_STANDARD 20
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/LaphriaEngine
)

target_compile_definitions(LaphriaEngine PRIVATE
    VULKAN_HPP_DISPATCH_LOADER_DYNAMIC=1
    VULKAN_HPP_NO_STRUCT_CONSTRUCTORS=1
    VULKAN_HPP_HANDLE_ERROR_OUT_OF_DATE_AS_SUCCESS
    GLM_FORCE_DEPTH_ZERO_TO_ONE
    GLM_FORCE_RADIANS
    GLM_ENABLE_EXPERIMENTAL
    GLM_FORCE_CXX11
)

target_include_directories(LaphriaEngine PRIVATE
    ${stb_INCLUDE_DIRS}
    ${Vulkan_INCLUDE_DIR}
)

target_link_libraries(LaphriaEngine PRIVATE
    Vulkan::Vulkan
    glfw
    glm::glm
    fastgltf::fastgltf
    imgui::imgui
    KTX::ktx
    nlohmann_json::nlohmann_json
)

add_dependencies(LaphriaEngine LaphriaEngine_shaders)

if(WIN32 AND CMAKE_GENERATOR MATCHES "Visual Studio.*")
    set_target_properties(LaphriaEngine PROPERTIES
        VS_DEBUGGER_WORKING_DIRECTORY "$<TARGET_FILE_DIR:LaphriaEngine>"
    )
endif()
