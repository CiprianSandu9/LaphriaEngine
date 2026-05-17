# ReSTIR GI Receiver Cache Design

Date: 2026-05-16

## Goal

Prototype a sparse bright receiver guide for ReSTIR GI so difficult Sponza views can discover useful bounced-light receiver surfaces when local Sun Receiver proposals cannot see them often enough.

## Context

The current best Sponza validation preset is TemporalSpatial ReSTIR GI with Mixed Cosine + Sun Receiver guidance, one reservoir candidate, two spatial neighbors, temporal/spatial budget divisors of two, first-bounce environment and sun NEE, eight max bounces, and shadowed secondary sun candidate evaluation.

Recent measurements show the hard views are dominated by `localRejectNoLight`. The rejected bounce-1 sun and reservoir environment suffix experiments increased cost or accepted weak samples without solving useful receiver discovery. The new guide must therefore improve candidate direction selection, not add more suffix lighting to every candidate.

## Approach

Add an experimental receiver cache beside the existing per-pixel reservoir history buffer. Accepted high-value reservoir candidates write compact receiver records containing candidate position, normal, suffix radiance, target weight, confidence, frame id, and flags. Future local proposals can sample directions toward a cached receiver and evaluate them through the normal reservoir target path.

The first prototype is deliberately conservative:

- Keep the current Sun Receiver preset unchanged.
- Add a new proposal mode rather than changing existing modes.
- Mix cache-guided proposals with cosine and Sun Receiver guidance.
- Store only accepted, bright, surface-backed candidates.
- Treat cache-guided samples as ordinary local candidates after the ray trace.
- Add counters and source-color diagnostics to decide whether the cache should live or be killed.

## Architecture

The existing reservoir buffer remains bound at raygen bindings 12 and 13. A new receiver-cache current/history buffer pair is bound at 14 and 15. Each frame slot owns one cache buffer. The shader writes accepted high-value records into the current slot and reads candidate guides from the previous slot. Buffers are host-cleared only on explicit experiment/history reset, so cache slots can age instead of being cleared every dispatch.

The cache is a sparse fixed-size ring/hash, not a full irradiance field. The shader samples a small number of hashed slots and chooses the highest-weight valid receiver that lies in the current primary hemisphere. This keeps the prototype cheap and easy to remove if it behaves like the rejected environment suffix experiment.

## Data Flow

1. A local/temporal/spatial reservoir candidate is selected and validated.
2. If it is accepted, surface-backed, bright enough, and finite, the shader writes a receiver-cache record into the current cache.
3. A later pixel using the new proposal mode tries to load a valid previous-cache receiver.
4. If a receiver is found, the shader samples a narrow cone toward its position.
5. The resulting candidate ray is traced and evaluated by the existing local sample path.
6. Counters record attempts, hits, misses, no-light rejects, accepted guided samples, and selected guided samples.

## Diagnostics

Add UI/analysis counters for:

- `reservoirGiReceiverCacheStore`
- `reservoirGiReceiverCacheAttempt`
- `reservoirGiReceiverCacheHit`
- `reservoirGiReceiverCacheMiss`
- `reservoirGiReceiverCacheRejectNoLight`
- `reservoirGiReceiverCacheAccepted`
- `reservoirGiSelectedCache`

Extend `Reservoir GI Selected Source` so cache-selected samples show as cyan while existing local/temporal/spatial colors remain unchanged.

## Success Criteria

For Dark Courtyard, compare against the current Sun Receiver preset:

- `localRejectNoLight` falls or cache-specific no-light reject ratio is meaningfully better than generic local candidates.
- `firstHitProbeAvgLuma` and `reservoirGiAcceptedLumaSum` rise without collapsing average selected/target weight.
- `Reservoir GI Selected Source` shows useful cache contribution, especially outside tiny opening regions.
- Total frame time does not rise by more than about 25% without clear image improvement.
- No obvious flicker, ghosting, or blotchy reuse.

## Kill Criteria

Kill or redesign this prototype if cache-guided candidates mostly become no-light rejects, only improve Sunlit Wall, lower useful average contribution like the environment suffix experiment, visibly destabilize the image, or require large memory/update cost before improving Dark Courtyard.
