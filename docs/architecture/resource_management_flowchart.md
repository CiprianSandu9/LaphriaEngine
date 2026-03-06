# Resource Management Flowchart

This flowchart breaks down the asset loading pipeline within the `ResourceManager`, tracing the journey from a `.gltf` file to fully configured Vulkan GPU buffers, textures, materials, and Acceleration Structures.

```mermaid
flowchart TD
    Start([Scene::loadModel]) --> loadGLTF[ResourceManager::loadGltfModel]
    
    subgraph FastGLTF [FastGLTF Parsing]
        loadGLTF --> parseGLTF[Parse GLTF File]
        parseGLTF --> initModelRes[Initialize ModelResource]
    end
    
    subgraph AssetLoading [Asset Processing & GPU Uploads]
        initModelRes --> loadTex[1. loadTextures<br/><i>Load KTX/PNG -> Staging -> ImageBuffer -> ImageView/Sampler</i>]
        
        loadTex --> loadMat[2. loadMaterials<br/><i>Extract PBR Params -> Upload to MaterialBuffer (SSBO)</i>]
        
        loadMat --> processNodes[3. processSceneNodes<br/><i>Parse Hierarchy -> Create SceneNode Tree<br/>Extract Vertices & Indices</i>]
        
        processNodes --> uploadBuffers[4. uploadModelBuffers<br/><i>Host -> Staging -> VertexBuffer & IndexBuffer</i>]
    end
    
    subgraph BindingData [Vulkan Descriptor & RT Setup]
        uploadBuffers --> createDesc[5. createModelDescriptorSet<br/><i>Bind Set 1: Material Buffer & Texture Arrays</i>]
        
        createDesc --> buildBLAS[6. buildBLAS<br/><i>Create Bottom-Level Acceleration Structure<br/>vkCmdBuildAccelerationStructuresKHR</i>]
        
        buildBLAS --> cache[Store in Model Cache]
    end
    
    %% Return Flow
    cache --> ret[Return SceneNode Prototype]
    ret --> endLoad([Attach Prototype to Scene Graph])
    
    %% Styling
    classDef parse fill:#2A4365,stroke:#4299E1,stroke-width:2px,color:#fff
    classDef asset fill:#744210,stroke:#D69E2E,stroke-width:2px,color:#fff
    classDef bind fill:#22543D,stroke:#48BB78,stroke-width:2px,color:#fff
    
    class FastGLTF parse
    class AssetLoading asset
    class BindingData bind
```
