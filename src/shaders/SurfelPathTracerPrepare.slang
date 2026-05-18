#include "SurfelPathTracerCommon.slang"

[shader("compute")]
[numthreads(16, 16, 1)]
void main(uint3 tid : SV_DispatchThreadID) {}
