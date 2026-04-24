#include "ShaderCommon.slang"

// Path tracer RayPayload — surface properties returned by ClosestHit to Raygen.
// hitT < 0 is the escape sentinel set by Miss to indicate a sky hit.
// Total size: 3+3+3+3+1+1+3+1+1+1 floats × 4 = 20 × 4 = 80 bytes (within 128-byte limit).
struct RayPayload {
    float3 albedo;       // Linear base color × (1 - metallic) diffuse weight
    float3 emission;     // Emissive radiance; Miss writes sky radiance here
    float3 worldNormal;  // Shading normal in world space (after normal mapping)
    float3 hitPos;       // World-space position of the surface hit
    float  roughness;    // PBR roughness [MIN_ROUGHNESS, 1]
    float  metallic;     // PBR metallic [0, 1]
    float3 F0;           // Fresnel base reflectance (precomputed from albedo + metallic)
    float  ao;           // Baked ambient occlusion factor [0, 1]
    float  hitT;         // Ray t-value; negative means ray escaped to sky (Miss fired)
    uint   instanceID;   // InstanceID() from ClosestHit; 0 on sky hit
};

// Set 0 — RT descriptor set
[[vk::binding(0, 0)]] RaytracingAccelerationStructure tlas;
[[vk::binding(1, 0)]] RWTexture2D<float4> noisyColorOutput;   // 1 SPP raw radiance
[[vk::binding(2, 0)]] RWTexture2D<float4> gBufferNormals;     // world-space normal (xyz), w unused
[[vk::binding(3, 0)]] RWTexture2D<float>  gBufferDepth;       // linear ray hit distance (negative = sky)
[[vk::binding(4, 0)]] RWTexture2D<float2> motionVectors;      // screen-space UV offset current→previous

// Set 1 — global UBO
[[vk::binding(0, 1)]] ConstantBuffer<UniformBuffer> ubo;

[shader("raygeneration")]
void main() {
    uint2  launchID   = DispatchRaysIndex().xy;
    uint2  launchSize = DispatchRaysDimensions().xy;

    // Per-pixel RNG seed: unique per pixel and per frame.
    uint rngState = pcgHash(launchID.x + launchID.y * launchSize.x + ubo.frameCount * launchSize.x * launchSize.y);

    // Sub-pixel jitter (disabled when jitter fields are zero; set to Halton values to enable TAA).
    float2 pixelCenter = float2(launchID) + float2(0.5 + ubo.jitter_x, 0.5 + ubo.jitter_y);
    float2 inUV        = pixelCenter / float2(launchSize);
    float2 d           = inUV * 2.0 - 1.0;

    float4 target = mul(ubo.projInverse, float4(d.x, -d.y, 1.0, 1.0));
    float3 rayDir = mul(ubo.viewInverse, float4(normalize(target.xyz / target.w), 0.0)).xyz;
    float3 origin = ubo.cameraPos.xyz;

    // ── Path tracing loop ──────────────────────────────────────────────────
    float3 radiance   = float3(0.0, 0.0, 0.0);
    float3 throughput = float3(1.0, 1.0, 1.0);

    RayDesc ray;
    ray.Origin    = origin;
    ray.Direction = normalize(rayDir);
    ray.TMin      = 0.001;
    ray.TMax      = 10000.0;

    static const int MAX_BOUNCES = 3;

    for (int bounce = 0; bounce < MAX_BOUNCES; ++bounce)
    {
        RayPayload payload;
        payload.hitT = -1.0;

        TraceRay(tlas, RAY_FLAG_NONE, 0xFF, 0, 0, 0, ray, payload);

        // Sky miss: accumulate background emission and terminate.
        if (payload.hitT < 0.0) {
            radiance += throughput * payload.emission;
            // Write sentinel G-Buffer values so sky pixels don't leave stale/garbage data
            // that would confuse the denoiser's edge-stopping functions.
            if (bounce == 0) {
                gBufferNormals[launchID] = float4(0.0, 0.0, 0.0, 0.0);
                gBufferDepth[launchID]   = -1.0;
                motionVectors[launchID]  = float2(0.0, 0.0);
            }
            break;
        }

        // Accumulate emissive contribution from the surface.
        radiance += throughput * payload.emission;

        float3 N2 = payload.worldNormal;
        float3 V2 = -ray.Direction;

        // ── NEE: direct lighting from directional sun light ────────────────
        // ubo.lightDir points FROM the light TOWARD the scene, so negate for surface-to-light.
        float3 Ldir  = normalize(-ubo.lightDir.xyz);
        float  NdotL = max(dot(N2, Ldir), 0.0);
        if (NdotL > 0.0)
        {
            // Cheap occlusion ray: skip ClosestHit, terminate on first accepted hit.
            RayPayload shadowPayload;
            shadowPayload.hitT        = 1.0;   // positive sentinel = "occluded"; Miss sets it to -1
            shadowPayload.albedo      = float3(0.0, 0.0, 0.0);
            shadowPayload.emission    = float3(0.0, 0.0, 0.0);
            shadowPayload.worldNormal = float3(0.0, 1.0, 0.0);
            shadowPayload.hitPos      = float3(0.0, 0.0, 0.0);
            shadowPayload.roughness   = 1.0;
            shadowPayload.metallic    = 0.0;
            shadowPayload.F0          = float3(0.0, 0.0, 0.0);
            shadowPayload.ao          = 1.0;
            shadowPayload.instanceID  = 0;

            RayDesc shadowRay;
            shadowRay.Origin    = payload.hitPos + N2 * 0.002;
            shadowRay.Direction = Ldir;
            shadowRay.TMin      = 0.001;
            shadowRay.TMax      = 10000.0;
            TraceRay(tlas,
                RAY_FLAG_SKIP_CLOSEST_HIT_SHADER,
                0xFF, 0, 0, 0, shadowRay, shadowPayload);

            // Use a shadow floor instead of fully blacked-out direct sun for readability.
            float3 H     = normalize(Ldir + V2);
            float  NdotV = max(dot(N2, V2),   0.0001);
            float  VdotH = max(dot(V2, H),    0.0001);

            float3 Fs   = fresnelSchlick(VdotH, payload.F0);
            float  D    = distributionGGX(N2, H, payload.roughness);
            float  G    = geometrySmith(N2, V2, Ldir, payload.roughness);
            float3 kD   = (float3(1.0, 1.0, 1.0) - Fs) * (1.0 - payload.metallic);

            float3 diffuse  = kD * payload.albedo / PI;
            float3 specular = (D * Fs * G) / max(4.0 * NdotV * NdotL, 0.0001);

            float3 sunRadiance = gameplaySunRadiance(ubo.gameplayVisuals);  // keep direct light aligned with raster/RT readability profile
            radiance += throughput * (diffuse + specular) * NdotL * sunRadiance * ((shadowPayload.hitT < 0.0) ? 1.0 : GAMEPLAY_SHADOW_FLOOR);
        }

        // ── Primary-ray G-Buffer write (bounce 0 only) ─────────────────────
        // Keep readability fill + ambient on the primary hit only; applying these helper
        // lights on every bounce over-amplifies color cast and drifts toward blue.
        if (bounce == 0) {
            float3 fillL = normalize(GAMEPLAY_FILL_DIR);
            float  NdotFill = max(dot(N2, fillL), 0.0);
            if (NdotFill > 0.0) {
                float3 Hfill  = normalize(fillL + V2);
                float  NdotVFill = max(dot(N2, V2), 0.0001);
                float  VdotHFill = max(dot(V2, Hfill), 0.0001);
                float3 FsFill = fresnelSchlick(VdotHFill, payload.F0);
                float  Dfill  = distributionGGX(N2, Hfill, payload.roughness);
                float  Gfill  = geometrySmith(N2, V2, fillL, payload.roughness);
                float3 kDFill = (float3(1.0, 1.0, 1.0) - FsFill) * (1.0 - payload.metallic);
                float3 diffuseFill  = kDFill * payload.albedo / PI;
                float3 specularFill = (Dfill * FsFill * Gfill) / max(4.0 * NdotVFill * NdotFill, 0.0001);
                radiance += throughput * (diffuseFill + specularFill) * NdotFill * gameplayFillRadiance(ubo.gameplayVisuals);
            }

            // Hemispherical ambient term for low-SPP readability stability.
            float3 sunDir3     = normalize(-ubo.lightDir.xyz);
            float3 skyColor    = evalSkyColor(float3(0.0, 1.0, 0.0), sunDir3);
            float3 groundColor = evalGroundColor(sunDir3);
            float3 ambientL    = lerp(groundColor, skyColor, N2.y * 0.5 + 0.5) * gameplayAmbientBoost(ubo.gameplayVisuals);
            float3 kS_amb      = fresnelSchlickRoughness(max(dot(N2, V2), 0.0), payload.F0, payload.roughness);
            float3 kD_amb      = (1.0 - kS_amb) * (1.0 - payload.metallic);
            float3 ambientContribution = (kD_amb * payload.albedo * ambientL + kS_amb * ambientL * (1.0 - payload.roughness * 0.5)) * payload.ao;
            radiance += throughput * ambientContribution;
        }

        // Primary-ray G-Buffer write (bounce 0 only).
        if (bounce == 0) {
            gBufferNormals[launchID] = float4(payload.worldNormal, 0.0);
            gBufferDepth[launchID]   = payload.hitT;

            // Motion vector: re-project hit world position with previous frame VP.
            float4 prevClip = mul(ubo.prevViewProj, float4(payload.hitPos, 1.0));
            // GLM proj is Y-up NDC: prevClip.y/w = +(1 - 2*v) where v=0 at top.
            // Convert to image UV (Y-down, v=0 at top) by negating Y, matching the
            // -d.y correction used in primary ray generation.
            float2 prevNDC  = (prevClip.xy / prevClip.w) * float2(0.5, -0.5) + 0.5;
            // Use unjittered UV so TAA jitter doesn't appear as false motion in the denoiser.
            float2 unjitteredUV = (float2(launchID) + 0.5) / float2(launchSize);
            motionVectors[launchID] = unjitteredUV - prevNDC;
        }

        // ── BSDF sampling ──────────────────────────────────────────────────
        float3 N = payload.worldNormal;
        float3 V = -ray.Direction;

        float2 xi = float2(randomFloat(rngState), randomFloat(rngState));

        // Probabilistically choose diffuse vs specular lobe based on Fresnel reflectance.
        float3 F       = fresnelSchlick(max(dot(N, V), 0.0), payload.F0);
        float  fAvg    = (F.r + F.g + F.b) / 3.0;
        float  specProb = clamp(fAvg + payload.metallic * 0.5, 0.1, 0.9);

        float3 newDir;
        float  pdf;
        float3 bsdfWeight;

        if (randomFloat(rngState) < specProb) {
            // Specular (GGX) lobe
            newDir = ggxSampleDirection(xi, N, V, payload.roughness);
            float NdotL = max(dot(N, newDir), 0.0);
            if (NdotL <= 0.0) break;

            float3 H    = normalize(V + newDir);
            float  G    = geometrySmith(N, V, newDir, payload.roughness);
            float3 Fs   = fresnelSchlick(max(dot(H, V), 0.0), payload.F0);
            float  NdotV = max(dot(N, V),    0.0001);
            float  NdotH = max(dot(N, H),    0.0001);
            float  VdotH = max(dot(V, H),    0.0001);
            bsdfWeight   = (G * Fs * VdotH) / max(NdotV * NdotH, 0.0001);
            pdf          = specProb;
        } else {
            // Diffuse (cosine hemisphere) lobe — cosine factor cancels with PDF.
            newDir = cosineSampleHemisphere(xi, N);
            if (max(dot(N, newDir), 0.0) <= 0.0) break;

            float3 kD  = (float3(1.0, 1.0, 1.0) - F) * (1.0 - payload.metallic);
            bsdfWeight = kD * payload.albedo * payload.ao;
            pdf        = 1.0 - specProb;
        }

        throughput *= bsdfWeight / max(pdf, 0.0001);

        // Russian roulette: stochastic early termination after first bounce.
        if (bounce >= 1) {
            float maxT  = max(throughput.r, max(throughput.g, throughput.b));
            float rrProb = clamp(maxT, 0.01, 0.95); // clamp away from 0 to prevent 0/0 NaN
            if (randomFloat(rngState) > rrProb) break;
            throughput /= rrProb;
        }

        // Advance ray — offset origin along geometric normal to avoid self-intersection.
        ray.Origin    = payload.hitPos + payload.worldNormal * 0.001;
        ray.Direction = normalize(newDir);
        ray.TMin      = 0.001;
        ray.TMax      = 10000.0;
    }

    // Write noisy 1-SPP result — linear HDR, tone mapping is applied in the final A-Trous pass.
    noisyColorOutput[launchID] = float4(radiance, 1.0);
}
