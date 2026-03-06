# Physically Based Rendering (PBR) Flowchart

This flowchart outlines how materials and lighting are physically evaluated within the engine's shaders. The implementation utilizes a Cook-Torrance BRDF with a GGX microfacet distribution model.

```mermaid
flowchart TD
    Start([Material Evaluation Begin]) --> fetch[Fetch Material Data<br/><i>BaseColor, Normal, Metallic, Roughness</i>]
    
    %% Base PBR Setup
    fetch --> normalMap[Apply Normal Map Tangent Space]
    normalMap --> setupVec[Calculate N, V, L vectors]
    setupVec --> f0Setup[Determine F0<br/><i>lerp(0.04, BaseColor, Metallic)</i>]
    
    %% Cook-Torrance BRDF Evaluator
    subgraph BRDF [Cook-Torrance BRDF Evaluation]
        f0Setup --> dTerm[Distribution (D)<br/><i>distributionGGX</i>]
        f0Setup --> gTerm[Geometry (G)<br/><i>geometrySmith (Schlick-GGX)</i>]
        f0Setup --> fTerm[Fresnel (F)<br/><i>fresnelSchlick</i>]
        
        %% Specular component
        dTerm --> specCalc[Specular Lobe<br/><i>(D * G * F) / (4 * NdotV * NdotL)</i>]
        gTerm --> specCalc
        fTerm --> specCalc
        
        %% Diffuse component
        fTerm --> energyConserve[Energy Conservation<br/><i>kD = (1.0 - F) * (1.0 - Metallic)</i>]
        energyConserve --> diffCalc[Diffuse Lobe<br/><i>kD * BaseColor / PI</i>]
    end
    
    %% Engine Integrations
    subgraph EngineIntegration [Pipeline Specific Integration]
        specCalc --> addLight[Combine Lighting<br/><i>(Diffuse + Specular) * Radiance * NdotL</i>]
        diffCalc --> addLight
        
        addLight --> forward[Rasterizer / Classic RT<br/><i>Accumulate across Analytical Lights</i>]
        
        %% Path Tracer integration
        f0Setup -.-> ptSample[Path Tracer Importance Sampling]
        ptSample -.-> |"randomFloat() < metallic"| specularSample[ggxSampleDirection<br/><i>Sample Specular Microfacet</i>]
        ptSample -.-> |"randomFloat() >= metallic"| diffuseSample[cosineSampleHemisphere<br/><i>Sample Diffuse Lobe</i>]
    end
    
    %% Final Output
    forward --> emissive[Add Emissive Component]
    ptSample --> emissive
    
    emissive --> tonemap[ACES Tonemapping & sRGB Conversion<br/><i>(Raster / Final Output)</i>]
    tonemap --> End([Final Pixel Color])
    
    %% Styling
    classDef brdf fill:#2D3748,stroke:#4A5568,stroke-width:2px,color:#fff
    classDef calc fill:#805AD5,stroke:#B794F4,stroke-width:2px,color:#fff
    classDef sample fill:#C53030,stroke:#FC8181,stroke-width:2px,color:#fff
    
    class BRDF brdf
    class dTerm,gTerm,fTerm,specCalc,diffCalc calc
    class ptSample,specularSample,diffuseSample sample
```
