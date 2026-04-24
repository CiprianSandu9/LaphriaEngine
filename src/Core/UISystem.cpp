#include "UISystem.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <filesystem>
#include <optional>

#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_vulkan.h>

#include "EngineAuxiliary.h"
#include "ResourceManager.h"

using namespace Laphria;

namespace
{
std::string toLowerCopy(std::string value)
{
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char ch) {
        return static_cast<char>(std::tolower(ch));
    });
    return value;
}

std::optional<std::filesystem::path> resolveAssetRootPath(const std::string &root)
{
    std::error_code ec;
    std::filesystem::path rootPath(root);
    if (rootPath.empty())
    {
        return std::nullopt;
    }

    std::vector<std::filesystem::path> candidates;
    if (rootPath.is_absolute())
    {
        candidates.push_back(rootPath);
    }
    else
    {
        const auto cwd = std::filesystem::current_path(ec);
        if (!ec)
        {
            candidates.push_back(cwd / rootPath);
            // Walk up parent folders so "Assets" works when running from build output dirs.
            for (auto parent = cwd.parent_path(); !parent.empty() && parent != parent.parent_path(); parent = parent.parent_path())
            {
                candidates.push_back(parent / rootPath);
            }
        }
        candidates.push_back(rootPath);
    }

    for (const auto &candidate : candidates)
    {
        if (std::filesystem::exists(candidate, ec) && !ec)
        {
            return std::filesystem::weakly_canonical(candidate, ec);
        }
    }
    return std::nullopt;
}

bool projectWorldToScreen(const glm::vec3 &worldPos, const glm::mat4 &viewProj, const ImVec2 &displaySize, ImVec2 &screenPos)
{
    const glm::vec4 clip = viewProj * glm::vec4(worldPos, 1.0f);
    if (clip.w <= 0.0001f)
    {
        return false;
    }

    const glm::vec3 ndc = glm::vec3(clip) / clip.w;
    if (ndc.z < 0.0f || ndc.z > 1.0f)
    {
        return false;
    }

    screenPos.x = (ndc.x * 0.5f + 0.5f) * displaySize.x;
    screenPos.y = (1.0f - (ndc.y * 0.5f + 0.5f)) * displaySize.y;
    return std::isfinite(screenPos.x) && std::isfinite(screenPos.y);
}

bool screenRayToPlaneY(const ImVec2 &mousePos,
                       const ImVec2 &displaySize,
                       const glm::mat4 &invViewProj,
                       float planeY,
                       glm::vec3 &outHit)
{
    if (displaySize.x <= 0.0f || displaySize.y <= 0.0f)
    {
        return false;
    }

    const float ndcX = (2.0f * mousePos.x / displaySize.x) - 1.0f;
    const float ndcY = 1.0f - (2.0f * mousePos.y / displaySize.y);

    glm::vec4 nearPoint = invViewProj * glm::vec4(ndcX, ndcY, 0.0f, 1.0f);
    glm::vec4 farPoint = invViewProj * glm::vec4(ndcX, ndcY, 1.0f, 1.0f);
    if (std::abs(nearPoint.w) < 1e-6f || std::abs(farPoint.w) < 1e-6f)
    {
        return false;
    }

    nearPoint /= nearPoint.w;
    farPoint /= farPoint.w;

    const glm::vec3 origin = glm::vec3(nearPoint);
    const glm::vec3 direction = glm::normalize(glm::vec3(farPoint - nearPoint));
    if (std::abs(direction.y) < 1e-5f)
    {
        return false;
    }

    const float t = (planeY - origin.y) / direction.y;
    if (t < 0.0f)
    {
        return false;
    }

    outHit = origin + direction * t;
    return std::isfinite(outHit.x) && std::isfinite(outHit.y) && std::isfinite(outHit.z);
}

float worldUnitsPerPixel(float cameraDistance, float verticalFovRadians, float viewportHeightPixels)
{
    if (viewportHeightPixels <= 0.0f)
    {
        return 0.0f;
    }
    const float visibleHeight = 2.0f * std::tan(verticalFovRadians * 0.5f) * std::max(cameraDistance, 0.01f);
    return visibleHeight / viewportHeightPixels;
}
} // namespace

void UISystem::init(VulkanDevice &dev, GLFWwindow *window,
                    vk::Format colorFormat, vk::Format depthFormat) {
    std::vector<vk::DescriptorPoolSize> poolSizes = {
        {vk::DescriptorType::eSampler, 1000},
        {vk::DescriptorType::eCombinedImageSampler, 1000},
        {vk::DescriptorType::eSampledImage, 1000},
        {vk::DescriptorType::eStorageImage, 1000},
        {vk::DescriptorType::eUniformTexelBuffer, 1000},
        {vk::DescriptorType::eStorageTexelBuffer, 1000},
        {vk::DescriptorType::eUniformBuffer, 1000},
        {vk::DescriptorType::eStorageBuffer, 1000},
        {vk::DescriptorType::eUniformBufferDynamic, 1000},
        {vk::DescriptorType::eStorageBufferDynamic, 1000},
        {vk::DescriptorType::eInputAttachment, 1000}
    };

    vk::DescriptorPoolCreateInfo poolInfo = {};
    poolInfo.flags = vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet;
    poolInfo.maxSets = 1000;
    poolInfo.poolSizeCount = static_cast<uint32_t>(poolSizes.size());
    poolInfo.pPoolSizes = poolSizes.data();

    imguiDescriptorPool = vk::raii::DescriptorPool(dev.logicalDevice, poolInfo);

    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO &io = ImGui::GetIO();
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;

    ImGui::StyleColorsDark();

    ImGuiStyle &style = ImGui::GetStyle();
    style.ScaleAllSizes(static_cast<float>(WIDTH) / HEIGHT);

    ImGui_ImplGlfw_InitForVulkan(window, true);
    ImGui_ImplVulkan_InitInfo init_info = {};
    init_info.ApiVersion = VK_API_VERSION_1_4;
    init_info.Instance = *dev.instance;
    init_info.PhysicalDevice = *dev.physicalDevice;
    init_info.Device = *dev.logicalDevice;
    init_info.Queue = *dev.queue;
    init_info.DescriptorPool = *imguiDescriptorPool;
    init_info.MinImageCount = 3;
    init_info.ImageCount = 3;
    init_info.MSAASamples = VK_SAMPLE_COUNT_1_BIT;
    init_info.UseDynamicRendering = true;

    static const auto color_format = static_cast<VkFormat>(colorFormat);

    VkPipelineRenderingCreateInfoKHR pipeline_info = {};
    pipeline_info.sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO_KHR;
    pipeline_info.colorAttachmentCount = 1;
    pipeline_info.pColorAttachmentFormats = &color_format;
    pipeline_info.depthAttachmentFormat = static_cast<VkFormat>(depthFormat);
    pipeline_info.stencilAttachmentFormat = VK_FORMAT_UNDEFINED;

    init_info.PipelineRenderingCreateInfo = pipeline_info;

    ImGui_ImplVulkan_Init(&init_info);
    ImGui_ImplVulkan_CreateFontsTexture();
}

void UISystem::draw(GLFWwindow *window, Scene &scene, PhysicsSystem &physics,
                    ResourceManager &rm, vk::DescriptorSetLayout matLayout, Camera &camera) {
    if (!showEditorPanels) {
        return;
    }

    drawMainMenuBar(window);
    drawAssetBrowser(scene, rm, matLayout);
    drawValidationPanel();
    drawSceneHierarchy(scene);
    drawInspector(rm);
    drawPhysicsUI(scene, physics, rm, matLayout);
    drawSelectedNodeTransformGizmo(camera);

    ImGui::Begin("Lighting Control");
    ImGui::DragFloat3("Light Direction", glm::value_ptr(lightDirection), 0.01f, 0.0f, 0.0f);
    ImGui::Text("Dir: %.2f, %.2f, %.2f", lightDirection.x, lightDirection.y, lightDirection.z);
    ImGui::Separator();

    ImGui::Text("Camera Control");
    ImGui::SliderFloat("Speed", &camera.movementSpeed, 0.01f, 5.0f, "%.2f");
    ImGui::Separator();

    static bool freezeCulling = false;
    if (ImGui::Checkbox("Freeze Culling", &freezeCulling)) {
        scene.setFreezeCulling(freezeCulling);
    }
    if (freezeCulling) {
        ImGui::TextColored(ImVec4(1, 1, 0, 1), "Culling frustum is frozen");
    }
    ImGui::End();

    if (showModelLoadDialog) {
        ImGui::OpenPopup("Load Model");
        showModelLoadDialog = false;
    }
    if (ImGui::BeginPopupModal("Load Model", nullptr, ImGuiWindowFlags_AlwaysAutoResize)) {
        ImGui::InputText("Path", modelLoadPath, IM_ARRAYSIZE(modelLoadPath));
        if (ImGui::Button("Load", ImVec2(120, 0))) {
            try {
                scene.loadModel(modelLoadPath, rm, matLayout, selectedNode);
                LOGI("Loaded model: %s", modelLoadPath);
            } catch (const std::exception &e) {
                LOGI("Failed to load model: %s", e.what());
            }
            ImGui::CloseCurrentPopup();
        }
        ImGui::SameLine();
        if (ImGui::Button("Cancel", ImVec2(120, 0))) {
            ImGui::CloseCurrentPopup();
        }
        ImGui::EndPopup();
    }

    if (showSceneSaveDialog) {
        ImGui::OpenPopup("Save Scene");
        showSceneSaveDialog = false;
    }
    if (ImGui::BeginPopupModal("Save Scene", nullptr, ImGuiWindowFlags_AlwaysAutoResize)) {
        ImGui::InputText("Path", scenePath, IM_ARRAYSIZE(scenePath));
        if (ImGui::Button("Save", ImVec2(120, 0))) {
            scene.saveScene(scenePath, rm);
            if (hasLoadedProject) {
                project.sceneOutputPath = scenePath;
            }
            ImGui::CloseCurrentPopup();
        }
        ImGui::SameLine();
        if (ImGui::Button("Cancel", ImVec2(120, 0))) {
            ImGui::CloseCurrentPopup();
        }
        ImGui::EndPopup();
    }

    if (showSceneLoadDialog) {
        ImGui::OpenPopup("Load Scene");
        showSceneLoadDialog = false;
    }
    if (ImGui::BeginPopupModal("Load Scene", nullptr, ImGuiWindowFlags_AlwaysAutoResize)) {
        ImGui::InputText("Path", scenePath, IM_ARRAYSIZE(scenePath));
        if (ImGui::Button("Load", ImVec2(120, 0))) {
            try {
                scene.loadScene(scenePath, rm, matLayout);
                LOGI("Loaded scene: %s", scenePath);
            } catch (const std::exception &e) {
                LOGI("Failed to load scene: %s", e.what());
            }
            ImGui::CloseCurrentPopup();
        }
        ImGui::SameLine();
        if (ImGui::Button("Cancel", ImVec2(120, 0))) {
            ImGui::CloseCurrentPopup();
        }
        ImGui::EndPopup();
    }

    if (showProjectLoadDialog) {
        ImGui::OpenPopup("Load Project");
        showProjectLoadDialog = false;
    }
    if (ImGui::BeginPopupModal("Load Project", nullptr, ImGuiWindowFlags_AlwaysAutoResize)) {
        ImGui::InputText("Path", projectPath, IM_ARRAYSIZE(projectPath));
        if (ImGui::Button("Load", ImVec2(120, 0))) {
            std::string error;
            LaphriaEditor::EditorProject loadedProject;
            if (LaphriaEditor::EditorProject::loadFromFile(projectPath, loadedProject, &error)) {
                project = std::move(loadedProject);
                hasLoadedProject = true;
                strncpy_s(scenePath, project.sceneOutputPath.c_str(), IM_ARRAYSIZE(scenePath));
                assetListDirty = true;
                lastImportMessages.clear();
                LOGI("Loaded project: %s", projectPath);
            } else {
                LOGI("Failed to load project: %s", error.c_str());
            }
            ImGui::CloseCurrentPopup();
        }
        ImGui::SameLine();
        if (ImGui::Button("Cancel", ImVec2(120, 0))) {
            ImGui::CloseCurrentPopup();
        }
        ImGui::EndPopup();
    }

    if (showProjectSaveDialog) {
        ImGui::OpenPopup("Save Project");
        showProjectSaveDialog = false;
    }
    if (ImGui::BeginPopupModal("Save Project", nullptr, ImGuiWindowFlags_AlwaysAutoResize)) {
        ImGui::InputText("Path", projectPath, IM_ARRAYSIZE(projectPath));
        if (ImGui::Button("Save", ImVec2(120, 0))) {
            if (!hasLoadedProject) {
                project = LaphriaEditor::EditorProject{};
                hasLoadedProject = true;
            }
            project.sceneOutputPath = scenePath;
            if (project.assetRoots.empty()) {
                project.assetRoots.push_back("Assets");
            }
            std::string error;
            if (!LaphriaEditor::EditorProject::saveToFile(projectPath, project, &error)) {
                LOGI("Failed to save project: %s", error.c_str());
            } else {
                LOGI("Saved project: %s", projectPath);
            }
            ImGui::CloseCurrentPopup();
        }
        ImGui::SameLine();
        if (ImGui::Button("Cancel", ImVec2(120, 0))) {
            ImGui::CloseCurrentPopup();
        }
        ImGui::EndPopup();
    }
}

void UISystem::cleanup() {
    ImGui_ImplVulkan_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();
}

void UISystem::drawMainMenuBar(GLFWwindow *window) {
    if (ImGui::BeginMainMenuBar()) {
        if (ImGui::BeginMenu("File")) {
            if (ImGui::MenuItem("Load Model...")) {
                showModelLoadDialog = true;
            }
            if (ImGui::MenuItem("Save Scene...")) {
                showSceneSaveDialog = true;
            }
            if (ImGui::MenuItem("Load Scene...")) {
                showSceneLoadDialog = true;
            }
            ImGui::Separator();
            if (ImGui::MenuItem("Load Project...")) {
                showProjectLoadDialog = true;
            }
            if (ImGui::MenuItem("Save Project...")) {
                showProjectSaveDialog = true;
            }
            ImGui::Separator();
            if (ImGui::MenuItem("Exit")) {
                glfwSetWindowShouldClose(window, true);
            }
            ImGui::EndMenu();
        }
        ImGui::EndMainMenuBar();
    }
}

void UISystem::drawSceneHierarchy(Scene &scene) {
    ImGui::Begin("Scene Hierarchy");

    if (scene.getRoot()) {
        drawSceneNode(scene.getRoot(), scene);
    }

    if (!nodesPendingDeletion.empty()) {
        for (const auto &node: nodesPendingDeletion) {
            scene.deleteNode(node);
        }
        nodesPendingDeletion.clear();
    }

    ImGui::End();
}

void UISystem::drawSceneNode(const SceneNode::Ptr &node, Scene &scene) {
    ImGuiTreeNodeFlags flags = ImGuiTreeNodeFlags_OpenOnArrow | ImGuiTreeNodeFlags_OpenOnDoubleClick;
    if (selectedNode == node) {
        flags |= ImGuiTreeNodeFlags_Selected;
    }
    if (node->getChildren().empty()) {
        flags |= ImGuiTreeNodeFlags_Leaf;
    }

    const char *label = node->name.empty() ? "Node" : node->name.c_str();
    bool opened = ImGui::TreeNodeEx(reinterpret_cast<void *>(reinterpret_cast<intptr_t>(node.get())), flags, "%s", label);

    if (ImGui::IsItemClicked()) {
        selectedNode = node;
    }

    if (ImGui::BeginPopupContextItem()) {
        if (ImGui::MenuItem("Delete")) {
            if (node != scene.getRoot()) {
                nodesPendingDeletion.push_back(node);
                if (selectedNode == node)
                    selectedNode = nullptr;
            }
        }
        if (ImGui::MenuItem("Add Child")) {
            auto child = std::make_shared<SceneNode>("New Node");
            scene.addNode(child, node);
            scene.rebuildOctree();
        }
        if (ImGui::MenuItem("Duplicate")) {
            if (node != scene.getRoot()) {
                auto clone = node->clone();
                clone->name += "_Copy";
                if (node->getParent()) {
                    scene.addNode(clone, node->getParent()->shared_from_this());
                } else {
                    scene.addNode(clone);
                }
                selectedNode = clone;
                scene.rebuildOctree();
            }
        }
        if (ImGui::MenuItem("Mark For Reparent")) {
            if (node != scene.getRoot()) {
                nodePendingReparent = node;
            }
        }
        if (nodePendingReparent && nodePendingReparent != node) {
            if (ImGui::MenuItem("Reparent Marked Here")) {
                if (!isDescendant(nodePendingReparent, node)) {
                    if (SceneNode *oldParent = nodePendingReparent->getParent()) {
                        oldParent->removeChild(nodePendingReparent);
                    }
                    node->addChild(nodePendingReparent);
                    scene.rebuildOctree();
                }
                nodePendingReparent.reset();
            }
        }
        if (nodePendingReparent && ImGui::MenuItem("Cancel Pending Reparent")) {
            nodePendingReparent.reset();
        }
        ImGui::EndPopup();
    }

    if (opened) {
        for (auto &child: node->getChildren()) {
            drawSceneNode(child, scene);
        }
        ImGui::TreePop();
    }
}

void UISystem::drawInspector(ResourceManager &rm) {
    ImGui::Begin("Inspector");

    if (selectedNode) {
        char nameBuf[128];
        strncpy_s(nameBuf, selectedNode->name.c_str(), sizeof(nameBuf));
        if (ImGui::InputText("Name", nameBuf, sizeof(nameBuf))) {
            selectedNode->name = nameBuf;
        }

        char stableIdBuf[128];
        strncpy_s(stableIdBuf, selectedNode->stableId.c_str(), sizeof(stableIdBuf));
        ImGui::InputText("Stable Id", stableIdBuf, sizeof(stableIdBuf), ImGuiInputTextFlags_ReadOnly);

        ImGui::Separator();
        ImGui::Text("Transform");

        glm::vec3 pos = selectedNode->getPosition();
        if (ImGui::DragFloat3("Position", glm::value_ptr(pos), 0.1f)) {
            selectedNode->setPosition(pos);
        }

        glm::vec3 euler = selectedNode->getEulerRotation();
        if (ImGui::DragFloat3("Rotation", glm::value_ptr(euler), 0.5f)) {
            selectedNode->setEulerRotation(euler);
        }

        glm::vec3 scale = selectedNode->getScale();
        if (ImGui::DragFloat3("Scale", glm::value_ptr(scale), 0.1f)) {
            selectedNode->setScale(scale);
        }
        ImGui::TextUnformatted("Transform Gizmo");
        bool translateMode = (transformGizmoMode == TransformGizmoMode::Translate);
        bool rotateMode = (transformGizmoMode == TransformGizmoMode::Rotate);
        bool scaleMode = (transformGizmoMode == TransformGizmoMode::Scale);
        bool offMode = (transformGizmoMode == TransformGizmoMode::None);
        if (ImGui::RadioButton("Translate", translateMode)) {
            transformGizmoMode = TransformGizmoMode::Translate;
            activeTransformAxis = -1;
            transformGizmoDragging = false;
        }
        ImGui::SameLine();
        if (ImGui::RadioButton("Rotate", rotateMode)) {
            transformGizmoMode = TransformGizmoMode::Rotate;
            activeTransformAxis = -1;
            transformGizmoDragging = false;
        }
        ImGui::SameLine();
        if (ImGui::RadioButton("Scale", scaleMode)) {
            transformGizmoMode = TransformGizmoMode::Scale;
            activeTransformAxis = -1;
            transformGizmoDragging = false;
        }
        ImGui::SameLine();
        if (ImGui::RadioButton("Off", offMode)) {
            transformGizmoMode = TransformGizmoMode::None;
            activeTransformAxis = -1;
            transformGizmoDragging = false;
        }
        if (transformGizmoMode != TransformGizmoMode::None) {
            ImGui::TextUnformatted("Viewport: drag axis handles for 3D transform.");
        }

        ImGui::Separator();
        if (selectedNode->assetRef.path.empty()) {
            ImGui::TextUnformatted("Asset Ref: (none)");
        } else {
            ImGui::TextWrapped("Asset Ref: %s", selectedNode->assetRef.path.c_str());
        }

        if (ImGui::CollapsingHeader("Animation Preview", ImGuiTreeNodeFlags_DefaultOpen)) {
            ImGui::Checkbox("Enable Animation Component", &selectedNode->animation.enabled);
            if (selectedNode->animation.enabled) {
                if (auto *modelRes = rm.getModelResource(selectedNode->modelId)) {
                    if (modelRes->hasSkins && !modelRes->hasRuntimeSkinning) {
                        ImGui::TextColored(ImVec4(1.0f, 0.72f, 0.30f, 1.0f),
                                           "Skinned mesh detected: runtime skin deformation is not enabled yet.");
                        ImGui::TextColored(ImVec4(1.0f, 0.72f, 0.30f, 1.0f),
                                           "Result: imported character meshes will appear in bind pose (T-pose).");
                    } else if (modelRes->hasRuntimeSkinning) {
                        ImGui::TextColored(ImVec4(0.55f, 0.86f, 1.0f, 1.0f), "GPU skinning active for this model (raster path).");
                    }
                    if (!modelRes->animationClipNames.empty()) {
                        const char *preview = selectedNode->animation.clipId.empty() ? "(select clip)" : selectedNode->animation.clipId.c_str();
                        if (ImGui::BeginCombo("Clip", preview)) {
                            for (const auto &clipName : modelRes->animationClipNames) {
                                bool selected = (selectedNode->animation.clipId == clipName);
                                if (ImGui::Selectable(clipName.c_str(), selected)) {
                                    selectedNode->animation.clipId = clipName;
                                }
                                if (selected) {
                                    ImGui::SetItemDefaultFocus();
                                }
                            }
                            ImGui::EndCombo();
                        }
                    } else {
                        ImGui::TextUnformatted("No clip metadata on this model.");
                    }
                } else {
                    ImGui::TextUnformatted("No model resource attached.");
                }

                float clipDuration = rm.getAnimationClipDurationSeconds(selectedNode->modelId, selectedNode->animation.clipId);
                if (clipDuration <= 0.0f) {
                    clipDuration = 10.0f;
                }
                ImGui::SliderFloat("Time (s)", &selectedNode->animation.timeSeconds, 0.0f, clipDuration, "%.2f");
                ImGui::Text("Clip Duration: %.2f s", clipDuration);
                ImGui::SliderFloat("Speed", &selectedNode->animation.speed, -2.0f, 4.0f, "%.2f");
                ImGui::Checkbox("Loop", &selectedNode->animation.loop);
                ImGui::Checkbox("Autoplay", &selectedNode->animation.autoplay);
                ImGui::Checkbox("Playing", &selectedNode->animation.playing);
                if (ImGui::Button("Reset Timeline")) {
                    selectedNode->animation.timeSeconds = 0.0f;
                }
            }
        }

    } else {
        activeTransformAxis = -1;
        transformGizmoDragging = false;
        ImGui::Text("No object selected.");
    }

    ImGui::End();
}

void UISystem::drawSelectedNodeTransformGizmo(Camera &camera)
{
    if (!selectedNode || transformGizmoMode == TransformGizmoMode::None)
    {
        activeTransformAxis = -1;
        transformGizmoDragging = false;
        return;
    }

    ImGuiIO &io = ImGui::GetIO();
    if (io.DisplaySize.x <= 1.0f || io.DisplaySize.y <= 1.0f)
    {
        activeTransformAxis = -1;
        transformGizmoDragging = false;
        return;
    }

    constexpr float kFovRadians = glm::radians(45.0f);
    const float aspectRatio = io.DisplaySize.x / io.DisplaySize.y;
    const glm::mat4 view = camera.getViewMatrix();
    const glm::mat4 proj = glm::perspective(kFovRadians, aspectRatio, 0.1f, 1000.0f);
    const glm::mat4 viewProj = proj * view;

    glm::vec3 origin = selectedNode->getPosition();
    ImVec2 originScreen{};
    if (!projectWorldToScreen(origin, viewProj, io.DisplaySize, originScreen))
    {
        activeTransformAxis = -1;
        transformGizmoDragging = false;
        return;
    }

    const glm::vec3 axisWorld[3] = {
        glm::vec3(1.0f, 0.0f, 0.0f),
        glm::vec3(0.0f, 1.0f, 0.0f),
        glm::vec3(0.0f, 0.0f, 1.0f)};
    const ImU32 axisColors[3] = {
        IM_COL32(225, 70, 70, 255),
        IM_COL32(70, 210, 95, 255),
        IM_COL32(70, 140, 245, 255)};

    const float cameraDistance = std::max(glm::length(camera.position - origin), 0.75f);
    const float axisWorldLength = std::clamp(cameraDistance * 0.18f, 0.75f, 8.0f);
    const float handleRadius = 8.0f;

    ImVec2 axisScreen[3]{};
    bool axisVisible[3]{false, false, false};
    ImVec2 axisDir2D[3]{};
    float axisDirLen[3]{0.0f, 0.0f, 0.0f};
    for (int i = 0; i < 3; ++i)
    {
        axisVisible[i] = projectWorldToScreen(origin + axisWorld[i] * axisWorldLength, viewProj, io.DisplaySize, axisScreen[i]);
        const float dx = axisScreen[i].x - originScreen.x;
        const float dy = axisScreen[i].y - originScreen.y;
        const float len = std::sqrt(dx * dx + dy * dy);
        axisDirLen[i] = len;
        if (len > 1e-3f)
        {
            axisDir2D[i] = ImVec2(dx / len, dy / len);
        }
        else
        {
            axisDir2D[i] = ImVec2(1.0f, 0.0f);
        }
    }

    const ImVec2 mouse = ImGui::GetMousePos();
    if (!ImGui::IsMouseDown(ImGuiMouseButton_Left))
    {
        transformGizmoDragging = false;
        activeTransformAxis = -1;
    }

    auto distanceToSegment = [](const ImVec2 &p, const ImVec2 &a, const ImVec2 &b) {
        const float abx = b.x - a.x;
        const float aby = b.y - a.y;
        const float apx = p.x - a.x;
        const float apy = p.y - a.y;
        const float abLenSq = abx * abx + aby * aby;
        if (abLenSq <= 1e-5f)
        {
            const float dx = p.x - a.x;
            const float dy = p.y - a.y;
            return std::sqrt(dx * dx + dy * dy);
        }
        const float t = std::clamp((apx * abx + apy * aby) / abLenSq, 0.0f, 1.0f);
        const float qx = a.x + abx * t;
        const float qy = a.y + aby * t;
        const float dx = p.x - qx;
        const float dy = p.y - qy;
        return std::sqrt(dx * dx + dy * dy);
    };

    int hoveredAxis = -1;
    for (int i = 0; i < 3; ++i)
    {
        if (!axisVisible[i])
        {
            continue;
        }
        const float dx = mouse.x - axisScreen[i].x;
        const float dy = mouse.y - axisScreen[i].y;
        const float endDist = std::sqrt(dx * dx + dy * dy);
        const float lineDist = distanceToSegment(mouse, originScreen, axisScreen[i]);
        if (endDist <= handleRadius * 1.7f || lineDist <= 5.5f)
        {
            hoveredAxis = i;
            break;
        }
    }

    const bool canBeginInteraction = !ImGui::IsAnyItemHovered() && !ImGui::IsAnyItemActive() && !io.WantCaptureMouse;
    if (!transformGizmoDragging && hoveredAxis >= 0 && ImGui::IsMouseClicked(ImGuiMouseButton_Left) && canBeginInteraction)
    {
        activeTransformAxis = hoveredAxis;
        transformGizmoDragging = true;
        transformDragStartMouse = glm::vec2(mouse.x, mouse.y);
        transformDragStartPosition = selectedNode->getPosition();
        transformDragStartEuler = selectedNode->getEulerRotation();
        transformDragStartScale = selectedNode->getScale();
    }

    if (transformGizmoDragging && activeTransformAxis >= 0 && activeTransformAxis < 3 && axisVisible[activeTransformAxis])
    {
        const glm::vec2 mouseDelta = glm::vec2(mouse.x, mouse.y) - transformDragStartMouse;
        const glm::vec2 axisDir = glm::vec2(axisDir2D[activeTransformAxis].x, axisDir2D[activeTransformAxis].y);

        if (transformGizmoMode == TransformGizmoMode::Translate)
        {
            const float pixelDelta = glm::dot(mouseDelta, axisDir);
            const float unitsPerPixel = worldUnitsPerPixel(cameraDistance, kFovRadians, io.DisplaySize.y);
            const glm::vec3 newPos = transformDragStartPosition + axisWorld[activeTransformAxis] * (pixelDelta * unitsPerPixel);
            selectedNode->setPosition(newPos);
        }
        else if (transformGizmoMode == TransformGizmoMode::Rotate)
        {
            const glm::vec2 tangent(-axisDir.y, axisDir.x);
            const float pixelDelta = glm::dot(mouseDelta, tangent);
            const float degreesDelta = pixelDelta * 0.35f;
            glm::vec3 euler = transformDragStartEuler;
            euler[activeTransformAxis] += degreesDelta;
            selectedNode->setEulerRotation(euler);
        }
        else if (transformGizmoMode == TransformGizmoMode::Scale)
        {
            const float pixelDelta = glm::dot(mouseDelta, axisDir);
            glm::vec3 scale = transformDragStartScale;
            const float axisStart = transformDragStartScale[activeTransformAxis];
            if (std::abs(axisStart) < 0.01f)
            {
                scale[activeTransformAxis] = std::clamp(0.01f + pixelDelta * 0.01f, 0.01f, 512.0f);
            }
            else
            {
                const float factor = std::exp(pixelDelta * 0.01f);
                scale[activeTransformAxis] = std::clamp(axisStart * factor, 0.01f, 512.0f);
            }
            selectedNode->setScale(scale);
        }
    }

    ImDrawList *drawList = ImGui::GetForegroundDrawList();
    for (int i = 0; i < 3; ++i)
    {
        if (!axisVisible[i])
        {
            continue;
        }
        const bool highlighted = (i == hoveredAxis) || (i == activeTransformAxis && transformGizmoDragging);
        const float thickness = highlighted ? 3.0f : 2.0f;
        drawList->AddLine(originScreen, axisScreen[i], axisColors[i], thickness);
        if (transformGizmoMode == TransformGizmoMode::Scale)
        {
            const ImVec2 minP(axisScreen[i].x - handleRadius, axisScreen[i].y - handleRadius);
            const ImVec2 maxP(axisScreen[i].x + handleRadius, axisScreen[i].y + handleRadius);
            drawList->AddRectFilled(minP, maxP, axisColors[i], 1.5f);
            drawList->AddRect(minP, maxP, IM_COL32(16, 16, 16, 220), 1.5f, 0, 2.0f);
        }
        else
        {
            drawList->AddCircleFilled(axisScreen[i], handleRadius, axisColors[i]);
            drawList->AddCircle(axisScreen[i], handleRadius + 1.5f, IM_COL32(16, 16, 16, 220), 0, 2.0f);
        }
    }

    if (transformGizmoMode == TransformGizmoMode::Rotate)
    {
        for (int i = 0; i < 3; ++i)
        {
            if (!axisVisible[i])
            {
                continue;
            }
            const float radius = std::max(axisDirLen[i], 12.0f);
            const ImU32 ringColor = IM_COL32((axisColors[i] >> IM_COL32_R_SHIFT) & 0xff,
                                             (axisColors[i] >> IM_COL32_G_SHIFT) & 0xff,
                                             (axisColors[i] >> IM_COL32_B_SHIFT) & 0xff,
                                             130);
            drawList->AddCircle(originScreen, radius, ringColor, 44, 1.4f);
        }
    }

    const char *label = "Transform Gizmo";
    if (transformGizmoMode == TransformGizmoMode::Translate)
    {
        label = "Translate Gizmo";
    }
    else if (transformGizmoMode == TransformGizmoMode::Rotate)
    {
        label = "Rotate Gizmo";
    }
    else if (transformGizmoMode == TransformGizmoMode::Scale)
    {
        label = "Scale Gizmo";
    }
    drawList->AddText(ImVec2(originScreen.x + 14.0f, originScreen.y + 8.0f), IM_COL32(230, 230, 230, 230), label);
}

void UISystem::drawAssetBrowser(Scene &scene, ResourceManager &rm, vk::DescriptorSetLayout matLayout) {
    ImGui::Begin("Asset Browser");

    if (!hasLoadedProject) {
        ImGui::TextUnformatted("Project: ad-hoc");
    } else {
        ImGui::Text("Project: %s", project.name.c_str());
    }

    if (project.assetRoots.empty()) {
        project.assetRoots.push_back("Assets");
    }

    ImGui::InputText("New Asset Root", newAssetRootPath, IM_ARRAYSIZE(newAssetRootPath));
    ImGui::SameLine();
    if (ImGui::Button("Add Root")) {
        std::string rootPath = newAssetRootPath;
        if (!rootPath.empty()) {
            project.assetRoots.push_back(rootPath);
            hasLoadedProject = true;
            assetListDirty = true;
        }
    }

    ImGui::Separator();
    for (size_t rootIndex = 0; rootIndex < project.assetRoots.size(); ++rootIndex) {
        ImGui::PushID(static_cast<int>(rootIndex));
        ImGui::TextWrapped("%s", project.assetRoots[rootIndex].c_str());
        ImGui::SameLine();
        if (ImGui::Button("Remove")) {
            project.assetRoots.erase(project.assetRoots.begin() + static_cast<std::ptrdiff_t>(rootIndex));
            assetListDirty = true;
            ImGui::PopID();
            break;
        }
        ImGui::PopID();
    }

    if (ImGui::Button("Refresh Assets")) {
        assetListDirty = true;
    }
    ImGui::SameLine();
    ImGui::Text("Found: %d", static_cast<int>(cachedAssetFiles.size()));

    if (assetListDirty) {
        refreshAssetCache();
        assetListDirty = false;
    }

    ImGui::BeginChild("AssetFiles", ImVec2(0, 220), true);
    for (const auto &assetPath : cachedAssetFiles) {
        bool selected = (selectedAssetPath == assetPath);
        if (ImGui::Selectable(assetPath.c_str(), selected)) {
            selectedAssetPath = assetPath;
        }
    }
    ImGui::EndChild();

    if (ImGui::Button("Import Selected")) {
        if (!selectedAssetPath.empty()) {
            try {
                auto model = rm.loadGltfModel(selectedAssetPath, matLayout);
                model->assetRef.path = selectedAssetPath;
                model->assetRef.variant = "default";
                scene.addNode(model, selectedNode);
                selectedNode = model;
                scene.rebuildOctree();

                lastImportMessages.clear();
                if (const auto *report = rm.getLastImportReport()) {
                    for (const auto &feature : report->supportedFeatures) {
                        lastImportMessages.push_back("[feature] " + feature);
                    }
                    for (const auto &warning : report->warnings) {
                        lastImportMessages.push_back("[warning] " + warning);
                    }
                    for (const auto &error : report->errors) {
                        lastImportMessages.push_back("[error] " + error);
                    }
                }
            } catch (const std::exception &e) {
                lastImportMessages.clear();
                lastImportMessages.push_back(std::string("[error] ") + e.what());
            }
        }
    }
    ImGui::SameLine();
    if (ImGui::Button("Save Scene To Project Path")) {
        if (!project.sceneOutputPath.empty()) {
            scene.saveScene(project.sceneOutputPath, rm);
            strncpy_s(scenePath, project.sceneOutputPath.c_str(), IM_ARRAYSIZE(scenePath));
        }
    }

    if (!lastImportMessages.empty()) {
        ImGui::Separator();
        ImGui::TextUnformatted("Last Import Report");
        for (const auto &line : lastImportMessages) {
            if (line.rfind("[error]", 0) == 0) {
                ImGui::TextColored(ImVec4(1.0f, 0.42f, 0.42f, 1.0f), "%s", line.c_str());
            } else if (line.rfind("[warning]", 0) == 0) {
                ImGui::TextColored(ImVec4(1.0f, 0.76f, 0.35f, 1.0f), "%s", line.c_str());
            } else if (line.rfind("[feature]", 0) == 0) {
                ImGui::TextColored(ImVec4(0.55f, 0.86f, 1.0f, 1.0f), "%s", line.c_str());
            } else {
                ImGui::TextWrapped("%s", line.c_str());
            }
        }
    }

    ImGui::End();
}

void UISystem::drawValidationPanel()
{
    ImGui::Begin("Validation");

    ImGui::TextWrapped("Validate project and scene JSON for structural issues.");

    if (ImGui::Button("Run Project Validation"))
    {
        lastValidationReport = LaphriaEditor::EditorValidator::validateProjectFile(projectPath);
        hasValidationReport = true;
    }
    ImGui::SameLine();
    if (ImGui::Button("Run Scene Validation"))
    {
        lastValidationReport = LaphriaEditor::EditorValidator::validateSceneFile(scenePath);
        hasValidationReport = true;
    }

    if (ImGui::Button("Run Full Validation"))
    {
        const std::string sceneToValidate = hasLoadedProject ? project.sceneOutputPath : std::string(scenePath);
        lastValidationReport = LaphriaEditor::EditorValidator::validateProjectAndScene(projectPath, sceneToValidate);
        hasValidationReport = true;
    }

    if (!hasValidationReport)
    {
        ImGui::Separator();
        ImGui::TextUnformatted("No validation run yet.");
        ImGui::End();
        return;
    }

    const size_t errorCount = lastValidationReport.errorCount();
    const size_t warningCount = lastValidationReport.warningCount();
    ImGui::Separator();
    ImGui::Text("Errors: %llu | Warnings: %llu",
                static_cast<unsigned long long>(errorCount),
                static_cast<unsigned long long>(warningCount));

    if (lastValidationReport.messages.empty())
    {
        ImGui::TextColored(ImVec4(0.55f, 0.90f, 0.55f, 1.0f), "No issues detected.");
        ImGui::End();
        return;
    }

    ImGui::BeginChild("ValidationMessages", ImVec2(0, 240), true);
    for (const auto &message : lastValidationReport.messages)
    {
        const bool isError = message.severity == LaphriaEditor::ValidationSeverity::Error;
        const ImVec4 color = isError ? ImVec4(1.0f, 0.40f, 0.40f, 1.0f) : ImVec4(1.0f, 0.76f, 0.35f, 1.0f);

        ImGui::TextColored(color, "[%s] %s", LaphriaEditor::validationSeverityToString(message.severity), message.message.c_str());
        ImGui::TextWrapped("File: %s", message.file.c_str());
        ImGui::TextWrapped("Field: %s", message.fieldPath.c_str());
        ImGui::Separator();
    }
    ImGui::EndChild();

    ImGui::End();
}

void UISystem::refreshAssetCache() {
    cachedAssetFiles.clear();
    std::vector<std::string> roots = project.assetRoots;
    if (roots.empty()) {
        roots.push_back("Assets");
    }

    for (const auto &root : roots) {
        std::error_code ec;
        auto resolvedRoot = resolveAssetRootPath(root);
        if (!resolvedRoot.has_value()) {
            continue;
        }

        for (const auto &entry : std::filesystem::recursive_directory_iterator(*resolvedRoot, ec)) {
            if (ec) {
                break;
            }
            if (!entry.is_regular_file()) {
                continue;
            }
            const std::string extension = toLowerCopy(entry.path().extension().string());
            if (extension == ".gltf" || extension == ".glb") {
                cachedAssetFiles.push_back(entry.path().string());
            }
        }
    }
    std::sort(cachedAssetFiles.begin(), cachedAssetFiles.end());
    cachedAssetFiles.erase(std::unique(cachedAssetFiles.begin(), cachedAssetFiles.end()), cachedAssetFiles.end());
}

bool UISystem::isDescendant(const SceneNode::Ptr &node, const SceneNode::Ptr &candidateParent) {
    if (!node || !candidateParent) {
        return false;
    }
    for (const auto &child : node->getChildren()) {
        if (child == candidateParent || isDescendant(child, candidateParent)) {
            return true;
        }
    }
    return false;
}

void UISystem::drawPhysicsUI(Scene &scene, PhysicsSystem &physics,
                             ResourceManager &rm, vk::DescriptorSetLayout matLayout) {
    ImGui::Begin("Engine Controls");

    ImGui::Text("Rendering Backend:");
    if (ImGui::RadioButton("Rasterizer", renderMode == RenderMode::Rasterizer))
        renderMode = RenderMode::Rasterizer;
    ImGui::SameLine();
    if (ImGui::RadioButton("Ray Tracer", renderMode == RenderMode::RayTracer))
        renderMode = RenderMode::RayTracer;
    ImGui::SameLine();
    if (ImGui::RadioButton("Path Tracer##render_mode", renderMode == RenderMode::PathTracer))
        renderMode = RenderMode::PathTracer;

    if (ImGui::CollapsingHeader("Path Tracer##settings", ImGuiTreeNodeFlags_DefaultOpen)) {
        pathTracerSettings.resolutionScale = std::clamp(pathTracerSettings.resolutionScale, 0.5f, 1.0f);
        pathTracerSettings.denoiserIterations = std::clamp(pathTracerSettings.denoiserIterations, 1, 5);
        pathTracerSettings.targetFrameMs = std::clamp(pathTracerSettings.targetFrameMs, 8.0f, 40.0f);

        ImGui::SliderFloat("Resolution Scale", &pathTracerSettings.resolutionScale, 0.5f, 1.0f, "%.2f");
        ImGui::SliderInt("Denoiser Iterations", &pathTracerSettings.denoiserIterations, 1, 5);
        const char *qualityModes[] = {"Manual", "Auto Balanced", "Auto Aggressive"};
        int qualityMode = static_cast<int>(pathTracerSettings.qualityMode);
        if (ImGui::Combo("Quality Mode", &qualityMode, qualityModes, IM_ARRAYSIZE(qualityModes))) {
            pathTracerSettings.qualityMode = static_cast<PathTracerQualityMode>(qualityMode);
        }
        ImGui::Checkbox("Reduce Secondary Effects", &pathTracerSettings.reduceSecondaryEffects);
        ImGui::DragFloat("Target Frame (ms)", &pathTracerSettings.targetFrameMs, 0.1f, 8.0f, 40.0f, "%.2f");

        ImGui::Separator();
        ImGui::Text("PT Timings (GPU):");
        ImGui::Text("TLAS: %.3f ms", pathTracerPerfStats.tlasBuildMs);
        ImGui::Text("Ray Trace: %.3f ms", pathTracerPerfStats.rayTraceMs);
        ImGui::Text("Reprojection: %.3f ms", pathTracerPerfStats.reprojectionMs);
        ImGui::Text("Denoiser: %.3f ms", pathTracerPerfStats.denoiserMs);
        ImGui::Text("Total: %.3f ms", pathTracerPerfStats.totalFrameMs);
    }

    ImGui::Separator();

    ImGui::Text("Physics Backend:");
    if (ImGui::RadioButton("CPU", !useGPUPhysics))
        useGPUPhysics = false;
    ImGui::SameLine();
    if (ImGui::RadioButton("GPU", useGPUPhysics))
        useGPUPhysics = true;

    ImGui::Separator();

    if (ImGui::Button(simulationRunning ? "Pause" : "Play")) {
        simulationRunning = !simulationRunning;
    }
    ImGui::SameLine();
    if (ImGui::Button("Random Impulse")) {
        for (auto &node: scene.getAllNodes()) {
            if (node->physics.enabled && !node->physics.isStatic) {
                std::uniform_real_distribution<float> dist(0.0f, 1.0f);
                float rx = dist(rng);
                float ry = dist(rng);
                float rz = dist(rng);
                glm::vec3 randomDir = glm::normalize(glm::vec3(rx, ry, rz) * 2.0f - 1.0f);
                node->physics.velocity += randomDir * 15.0f;
            }
        }
    }
    ImGui::SameLine();
    if (ImGui::Button("Reset")) {
        simulationRunning = false;
        for (auto &node: scene.getAllNodes()) {
            node->resetToInitialState();
        }
    }

    ImGui::Separator();
    ImGui::Text("Scenarios (Predefined):");
    if (ImGui::Button("100S-250C-500CY")) {
        scene.createPhysicsScenario(1, rm, matLayout);
        for (auto &node: scene.getAllNodes())
            node->storeInitialState();
        simulationRunning = false;
    }
    if (ImGui::Button("250S-500C-1000CY")) {
        scene.createPhysicsScenario(2, rm, matLayout);
        for (auto &node: scene.getAllNodes())
            node->storeInitialState();
        simulationRunning = false;
    }
    if (ImGui::Button("500S-1000C-2500CY")) {
        scene.createPhysicsScenario(3, rm, matLayout);
        for (auto &node: scene.getAllNodes())
            node->storeInitialState();
        simulationRunning = false;
    }

    ImGui::Separator();
    ImGui::Text("Global Physics Parameters:");
    {
        glm::vec3 gravity = glm::vec3(0.f, -9.81f, 0.f);
        if (ImGui::DragFloat3("Gravity", glm::value_ptr(gravity), 0.1f, -50.f, 50.f)) {
            physics.setGravity(gravity);
        }
        static float globalFriction = 0.5f;
        if (ImGui::SliderFloat("Global Friction (GPU)", &globalFriction, 0.0f, 1.0f)) {
            physics.setGlobalFriction(globalFriction);
        }
    }

    ImGui::Separator();
    ImGui::Text("Metrics:");
    ImGui::Text("Compute Time: %.3f ms", physicsTime);
    ImGui::Text("Object Count: %d", static_cast<int>(scene.getAllNodes().size()));

    ImGui::End();
}
