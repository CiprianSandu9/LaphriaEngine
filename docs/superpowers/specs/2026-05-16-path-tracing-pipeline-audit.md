# Path Tracing Pipeline Audit for Radiance Cache Continuation

## Goal

Audit the current path tracing and ReSTIR GI pipeline before adding a radiance-cache-continuation prototype.

## Current Pipeline

The path tracer packs ReSTIR settings in `packPathTracerMaterialSettings()` and decodes them in `Raygen.slang`. Proposal mode uses three bits and is currently full at modes `0..7`; candidate evaluation mode uses two bits and currently uses values `0..2`, leaving value `3` available.

Primary shading calls `sampleFirstHitReservoirGiSingleFrame()` for the ReSTIR GI path. That function:

- chooses one or more local first-bounce proposal directions,
- traces a secondary surface ray,
- calls `makeLocalReservoirGiSample()` to evaluate the secondary hit suffix,
- combines local, temporal, spatial, and optional cache-reconnect candidates into one reservoir,
- validates temporal selections by reconnecting them,
- stores accepted reservoirs into the screen-space reservoir history and receiver cache.

`makeLocalReservoirGiSample()` is the narrowest useful integration point for cache continuation. It already has the secondary hit position, normal, material payload, first-leg throughput, proposal PDF, and candidate evaluation mode. It currently evaluates a suffix as emissive plus optional direct sun and then calls `evaluateReservoirGiTargetAtPrimary()` to put the sample in the current primary-hit domain.

## Existing Cache Infrastructure

The current receiver cache is a double-buffered `RWByteAddressBuffer` pair bound at ray tracing descriptor bindings 14 and 15:

- `ptReservoirGiReceiverCacheCurrent`
- `ptReservoirGiReceiverCacheHistory`

The record stores:

- candidate position,
- candidate normal,
- suffix radiance,
- target weight,
- confidence,
- frame id,
- flags,
- source pixel.

The cache was originally used as a direction proposal and then as a reconnect source. The reconnect sweep showed two lessons:

- cache as a proposal/reconnect source can easily dominate selection and cost,
- cache feedback must be blocked or carefully controlled.

## Gaps for Radiance Cache Continuation

The existing cache indexing is not spatial enough for secondary-hit continuation. It stores by a pixel/source hash, so a secondary hit cannot cheaply ask "is there cached radiance near this surface point?" The next prototype needs a spatial hash index derived from quantized world position.

The cache record has no separate "base radiance" and "cache-continued radiance" fields. To avoid feedback loops, cache-continuation candidates must not persist their combined suffix directly back into the receiver cache or temporal history. The prototype should store base direct-sun/emissive cache seeds separately, and mark final cache-continued candidates as non-persistent.

The current analysis counters distinguish cache proposal and reconnect work, but not secondary-hit cache-continuation work. Add dedicated counters so sweeps can answer whether cache continuation is actually helping:

- continuation attempts,
- hits,
- misses,
- accepted/used continuation radiance.

## Recommended First Slice

Add candidate evaluation mode `3`: shadowed sun plus receiver-cache continuation.

For each local secondary surface hit in this mode:

1. Compute the normal shadowed sun suffix as today.
2. Store a base receiver-cache seed containing only emissive plus direct-sun suffix.
3. Query the history receiver cache using a quantized world-space hash around the secondary hit.
4. Add compatible cached suffix radiance to the current suffix.
5. Evaluate the combined suffix with `evaluateReservoirGiTargetAtPrimary()`.
6. Mark the final combined sample as non-persistent to avoid cache feedback.

This is not a final DDGI/SHaRC implementation. It is a controlled prototype to test whether "cache as path continuation" improves receiver discovery and suffix value without changing the proposal model.
