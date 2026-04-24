#ifndef LAPHRIAENGINE_BROADPHASE_H
#define LAPHRIAENGINE_BROADPHASE_H

#include <utility>
#include <vector>

#include <glm/glm.hpp>

namespace Laphria::Physics
{
struct AABBProxy
{
	size_t id = 0;
	glm::vec3 min{0.0f};
	glm::vec3 max{0.0f};
};

// Returns potential collision pairs from a uniform-grid spatial hash.
// Pairs are conservative candidates for narrowphase and may include false positives.
std::vector<std::pair<size_t, size_t>> buildBroadphasePairs(const std::vector<AABBProxy> &proxies, float cellSize);
} // namespace Laphria::Physics

#endif // LAPHRIAENGINE_BROADPHASE_H
