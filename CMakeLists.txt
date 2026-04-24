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

set(SHADER_SOURCE_ROOT "${CMAKE_SOURCE_DIR}/src/shaders")

set(SHADER_SOURCES
        "LaphriaEngine.slang|vertMain|fragMain"
        "Compute.slang|computeMain"
        "Skinning.slang|skinningMain"
        "Shadow.slang|shadowVert|shadowFrag"
        "Physics.slang|physicsMain"
        "RT_ClosestHit.slang|main"
        "RT_AnyHit.slang|main"
        "RT_Miss.slang|main"
        "RT_Raygen.slang|main"
        "ClosestHit.slang|main"
        "AnyHit.slang|main"
        "Miss.slang|main"
        "Raygen.slang|main"
        "Reprojection.slang|reprojectionMain"
        "Denoiser.slang|atrousMain"
)

if (CMAKE_CONFIGURATION_TYPES)
    set(SHADERS_DIR "${CMAKE_BINARY_DIR}/LaphriaEngine/$<CONFIG>/Shaders")
else ()
    set(SHADERS_DIR "${CMAKE_BINARY_DIR}/LaphriaEngine/Shaders")
endif ()
set(GENERATED_SPV_FILES "")

foreach (SHADER_ENTRY ${SHADER_SOURCES})
    string(REPLACE "|" ";" SHADER_PARTS "${SHADER_ENTRY}")
    list(GET SHADER_PARTS 0 SHADER_RELATIVE_PATH)
    set(SHADER_SOURCE "${SHADER_SOURCE_ROOT}/${SHADER_RELATIVE_PATH}")
    list(REMOVE_AT SHADER_PARTS 0)

    get_filename_component(SHADER_NAME ${SHADER_RELATIVE_PATH} NAME)
    set(SHADER_OUTPUT "${SHADERS_DIR}/${SHADER_NAME}.spv")

    set(ENTRY_POINTS "")
    foreach (EP ${SHADER_PARTS})
        list(APPEND ENTRY_POINTS -entry ${EP})
    endforeach ()

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
endforeach ()

add_custom_target(LaphriaEngine_shaders DEPENDS ${GENERATED_SPV_FILES})

set(LAPHRIA_ENGINE_SOURCES
        src/Core/Camera.cpp
        src/Core/Camera.h
        src/Core/EngineAuxiliary.h
        src/Core/EngineConfig.h
        src/Core/EngineCore.cpp
        src/Core/EngineCore.h
        src/Core/EditorProject.cpp
        src/Core/EditorProject.h
        src/Core/EngineHost.cpp
        src/Core/EngineHost.h
        src/Core/FrameContext.cpp
        src/Core/FrameContext.h
        src/Core/GltfImporter.cpp
        src/Core/GltfImporter.h
        src/Core/GpuResourceRegistry.cpp
        src/Core/GpuResourceRegistry.h
        src/Core/InputSystem.cpp
        src/Core/InputSystem.h
        src/Core/PipelineCollection.cpp
        src/Core/PipelineCollection.h
        src/Core/ResourceManager.cpp
        src/Core/ResourceManager.h
        src/Core/StbImageImpl.cpp
        src/Core/SwapchainManager.cpp
        src/Core/SwapchainManager.h
        src/Core/UISystem.cpp
        src/Core/UISystem.h
        src/Core/VulkanDevice.cpp
        src/Core/VulkanDevice.h
        src/Core/VmaContext.cpp
        src/Core/VmaContext.h
        src/Core/VulkanUtils.cpp
        src/Core/VulkanUtils.h
        src/Physics/Broadphase.cpp
        src/Physics/Broadphase.h
        src/Physics/PhysicsDefines.h
        src/Physics/PhysicsSystem.cpp
        src/Physics/PhysicsSystem.h
        src/SceneManagement/Frustum.h
        src/SceneManagement/Octree.h
        src/SceneManagement/Scene.cpp
        src/SceneManagement/Scene.h
        src/SceneManagement/SceneNode.cpp
        src/SceneManagement/SceneNode.h
)

set(LAPHRIA_EDITOR_VALIDATION_SOURCES
        src/Core/EditorValidation.cpp
        src/Core/EditorValidation.h
)

add_library(LaphriaEditorValidation STATIC ${LAPHRIA_EDITOR_VALIDATION_SOURCES})
set_target_properties(LaphriaEditorValidation PROPERTIES CXX_STANDARD 20)
target_link_libraries(LaphriaEditorValidation PUBLIC nlohmann_json::nlohmann_json)

add_library(LaphriaEngine STATIC ${LAPHRIA_ENGINE_SOURCES})
set_target_properties(LaphriaEngine PROPERTIES CXX_STANDARD 20)

target_compile_definitions(LaphriaEngine PRIVATE
        VULKAN_HPP_DISPATCH_LOADER_DYNAMIC=1
        VULKAN_HPP_NO_STRUCT_CONSTRUCTORS=1
        VULKAN_HPP_HANDLE_ERROR_OUT_OF_DATE_AS_SUCCESS
)

target_compile_definitions(LaphriaEngine PUBLIC
        GLM_FORCE_DEPTH_ZERO_TO_ONE
        GLM_FORCE_RADIANS
        GLM_ENABLE_EXPERIMENTAL
        GLM_FORCE_CXX11
)

target_include_directories(LaphriaEngine PRIVATE
        ${stb_INCLUDE_DIRS}
        ${Vulkan_INCLUDE_DIR}
)

target_link_libraries(LaphriaEngine PUBLIC
        LaphriaEditorValidation
        Vulkan::Vulkan
        glfw
        glm::glm
        fastgltf::fastgltf
        imgui::imgui
        KTX::ktx
        nlohmann_json::nlohmann_json
)

add_dependencies(LaphriaEngine LaphriaEngine_shaders)

add_executable(LaphriaEditor src/main.cpp)
set_target_properties(LaphriaEditor PROPERTIES
        CXX_STANDARD 20
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/LaphriaEngine
)
target_link_libraries(LaphriaEditor PRIVATE LaphriaEngine)
add_dependencies(LaphriaEditor LaphriaEngine_shaders)

add_executable(LaphriaValidationRunner src/LaphriaValidationRunnerMain.cpp)
set_target_properties(LaphriaValidationRunner PROPERTIES
        CXX_STANDARD 20
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/LaphriaTools
)
target_link_libraries(LaphriaValidationRunner PRIVATE LaphriaEditorValidation)

if (WIN32 AND CMAKE_GENERATOR MATCHES "Visual Studio.*")
    set_target_properties(LaphriaEditor PROPERTIES
            VS_DEBUGGER_WORKING_DIRECTORY "$<TARGET_FILE_DIR:LaphriaEditor>"
    )
    set_target_properties(LaphriaValidationRunner PROPERTIES
            VS_DEBUGGER_WORKING_DIRECTORY "$<TARGET_FILE_DIR:LaphriaValidationRunner>"
    )
endif ()

enable_testing()

add_test(
        NAME LaphriaValidationProjectValid
        COMMAND LaphriaValidationRunner
        --validate-project
        --project "${CMAKE_SOURCE_DIR}/tests/fixtures/validation/valid_project.laphria_project.json"
)

add_test(
        NAME LaphriaValidationSceneValid
        COMMAND LaphriaValidationRunner
        --validate-scene
        --scene "${CMAKE_SOURCE_DIR}/tests/fixtures/validation/valid_scene.json"
)

add_test(
        NAME LaphriaValidationProjectInvalid
        COMMAND LaphriaValidationRunner
        --validate-project
        --project "${CMAKE_SOURCE_DIR}/tests/fixtures/validation/invalid_project.laphria_project.json"
)
set_tests_properties(LaphriaValidationProjectInvalid PROPERTIES WILL_FAIL TRUE)

add_test(
        NAME LaphriaValidationSceneInvalid
        COMMAND LaphriaValidationRunner
        --validate-scene
        --scene "${CMAKE_SOURCE_DIR}/tests/fixtures/validation/invalid_scene.json"
)
set_tests_properties(LaphriaValidationSceneInvalid PROPERTIES WILL_FAIL TRUE)

add_executable(LaphriaEngineUnitTests
        tests/EngineUnitTestsMain.cpp
        src/SceneManagement/SceneNode.cpp
        src/Physics/Broadphase.cpp
)
set_target_properties(LaphriaEngineUnitTests PROPERTIES CXX_STANDARD 20)
target_compile_definitions(LaphriaEngineUnitTests PRIVATE
        GLM_FORCE_DEPTH_ZERO_TO_ONE
        GLM_FORCE_RADIANS
        GLM_ENABLE_EXPERIMENTAL
        GLM_FORCE_CXX11
)
target_link_libraries(LaphriaEngineUnitTests PRIVATE
        glm::glm
        nlohmann_json::nlohmann_json
)

add_test(
        NAME LaphriaEngineUnitTests
        COMMAND LaphriaEngineUnitTests
)
