#include "SurfelCommon.slang"

[[vk::binding(0, 0)]] RWStructuredBuffer<SurfelGiRecord> surfelGiRecords;
[[vk::binding(3, 0)]] RWByteAddressBuffer surfelGiCounters;
[[vk::binding(4, 0)]] RWByteAddressBuffer ptAnalysisCounters;
[[vk::binding(0, 1)]] ConstantBuffer<UniformBuffer> ubo;

[shader("compute")]
[numthreads(128, 1, 1)]
void surfelIntegrateMain(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    const uint surfelIndex = dispatchThreadID.x;
    if (surfelIndex >= SURFEL_GI_MAX_SURFELS)
        return;

    SurfelGiRecord record = surfelGiRecords[surfelIndex];
    if ((record.flags & 1u) == 0u)
        return;

    record.radianceAge.w += 1.0f;
    record.normalConfidence.w = min(record.normalConfidence.w + 0.01f, 1.0f);
    surfelGiRecords[surfelIndex] = record;
}
