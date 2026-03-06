# Ray Tracing (RT) Pipeline Flowchart

This flowchart outlines the rendering path when `RenderMode::RayTracer` is selected. It features the Vulkan hardware-accelerated Ray Tracing pipeline, utilizing the Shader Binding Table (SBT) and the Top-Level Acceleration Structure (TLAS).

```mermaid
flowchart TD
    Start([recordCommandBuffer<br/><i>RenderMode::RayTracer</i>])

    %% TLAS is built in recordCommandBuffer before this path is entered — see high_level_architecture.md
    Start --> tlasPre[TLAS Pre-Built<br/><i>Built in recordCommandBuffer for all RT/PT frames<br/>AS_Write -> RT_Read barrier already applied</i>]

    %% Ray Tracing Pass
    subgraph RTPass [Classic RT Dispatch]
        tlasPre --> transOut[Transition RT Output Image<br/><i>eGeneral</i>]
        transOut --> bindRT[Bind classicRTPipeline<br/><i>RT_Raygen, RT_CHit, RT_Miss</i>]
        bindRT --> bindDesc[Bind Descriptor Sets<br/><i>Set 0: TLAS & Output Image<br/>Set 1: Global UBO</i>]
        bindDesc --> pushConst[Push Constants<br/><i>Identity Transform</i>]
        
        pushConst --> traceRays[vkCmdTraceRaysKHR]
        
        %% Shader Execution Flow (Conceptual)
        traceRays -.-> raygen[RT_Raygen.slang]
        raygen -.-> trace[Trace Ray]
        trace -.-> miss[RT_Miss.slang]
        trace -.-> intersect{Hit Triangle?}
        intersect -- "Yes (Alpha tested via AnyHit)" --> chit[RT_ClosestHit.slang<br/><i>Compute Lighting</i>]
        intersect -- No --> miss
        
        chit -.-> payload[Return RayPayload Color]
        miss -.-> payload
        payload -.-> writeOut[Write Color to Output Image]
        
        writeOut -.-> traceRays
    end
    
    %% Presentation
    subgraph SwapchainBlit [Image Blitting & UI]
        traceRays --> transBlitSrc[Trans. Output Image <br/><i>eTransferSrcOptimal</i>]
        transBlitSrc --> transBlitDst[Trans. Swapchain Image <br/><i>eTransferDstOptimal</i>]
        transBlitDst --> blit[vkCmdBlitImage]
        
        blit --> restoreGeneral[Restore Output Image<br/><i>eGeneral</i>]
        restoreGeneral --> transColorAtt[Transition Swapchain<br/><i>eColorAttachmentOptimal</i>]
        transColorAtt --> beginRender[beginRendering]
        
        beginRender --> uiPass[ImGui_ImplVulkan_RenderDrawData]
        uiPass --> endRender[endRendering]
        endRender --> ext[Transition Swapchain<br/><i>ePresentSrcKHR</i>]
    end
    
    %% Styling
    classDef pre fill:#2F855A,stroke:#48BB78,stroke-width:1px,color:#fff,stroke-dasharray: 5 5
    classDef rtexec fill:#9B2C2C,stroke:#F56565,stroke-width:2px,color:#fff
    classDef blit fill:#2B6CB0,stroke:#4299E1,stroke-width:2px,color:#fff
    classDef shader fill:#744210,stroke:#D69E2E,stroke-width:1px,color:#fff,stroke-dasharray: 4 4

    class tlasPre pre
    class RTPass rtexec
    class SwapchainBlit blit
    class raygen,trace,miss,intersect,chit,payload,writeOut shader
```
