#include "ShaderCommon.slang"

[shader("miss")]
void main(inout RayPayload payload) {
    if (payload.isShadowed == 1) {
        payload.isShadowed = 0; // Not occluded
        return;
    }

    // Return a nice background sky color gradient using the ray direction
    float3 rayDir = normalize(WorldRayDirection());
    
    float t = 0.5 * (rayDir.y + 1.0);
    float3 white = float3(1.0, 1.0, 1.0);
    float3 blue = float3(0.3, 0.4, 0.6); // Matches regular pipeline sky ambient color
    
    payload.color = lerp(white, blue, t);
}
