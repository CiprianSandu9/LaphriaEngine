#include "Broadphase.h"

#include <algorithm>
#include <cstdint>
#include <unordered_map>
#include <unordered_set>

namespace Laphria::Physics
{
namespace
{
struct CellCoord
{
	int x = 0;
	int y = 0;
	int z = 0;

	bool operator==(const CellCoord &other) const
	{
		return x == other.x && y == other.y && z == other.z;
	}
};

struct CellCoordHasher
{
	size_t operator()(const CellCoord &coord) const noexcept
	{
		size_t seed = 1469598103934665603ull;
		auto mix = [&seed](int value) {
			seed ^= static_cast<size_t>(value);
			seed *= 1099511628211ull;
		};
		mix(coord.x);
		mix(coord.y);
		mix(coord.z);
		return seed;
	}
};

uint64_t packPair(size_t a, size_t b)
{
	const uint32_t lo = static_cast<uint32_t>(std::min(a, b));
	const uint32_t hi = static_cast<uint32_t>(std::max(a, b));
	return (static_cast<uint64_t>(lo) << 32u) | static_cast<uint64_t>(hi);
}
} // namespace

std::vector<std::pair<size_t, size_t>> buildBroadphasePairs(const std::vector<AABBProxy> &proxies, float cellSize)
{
	std::vector<std::pair<size_t, size_t>> candidates;
	if (proxies.size() < 2 || cellSize <= 0.0f)
	{
		return candidates;
	}

	std::unordered_map<CellCoord, std::vector<size_t>, CellCoordHasher> cells;
	cells.reserve(proxies.size() * 2);

	for (size_t proxyIndex = 0; proxyIndex < proxies.size(); ++proxyIndex)
	{
		const AABBProxy &proxy = proxies[proxyIndex];
		const glm::ivec3 minCell = glm::ivec3(glm::floor(proxy.min / cellSize));
		const glm::ivec3 maxCell = glm::ivec3(glm::floor(proxy.max / cellSize));

		for (int x = minCell.x; x <= maxCell.x; ++x)
		{
			for (int y = minCell.y; y <= maxCell.y; ++y)
			{
				for (int z = minCell.z; z <= maxCell.z; ++z)
				{
					cells[CellCoord{x, y, z}].push_back(proxyIndex);
				}
			}
		}
	}

	std::unordered_set<uint64_t> uniquePairs;
	uniquePairs.reserve(proxies.size() * 4);

	for (const auto &[cell, entries] : cells)
	{
		(void)cell;
		for (size_t i = 0; i < entries.size(); ++i)
		{
			for (size_t j = i + 1; j < entries.size(); ++j)
			{
				const size_t idA = proxies[entries[i]].id;
				const size_t idB = proxies[entries[j]].id;
				if (idA == idB)
				{
					continue;
				}
				uniquePairs.insert(packPair(idA, idB));
			}
		}
	}

	candidates.reserve(uniquePairs.size());
	for (const uint64_t packed : uniquePairs)
	{
		const size_t a = static_cast<size_t>(packed >> 32u);
		const size_t b = static_cast<size_t>(packed & 0xffffffffu);
		candidates.emplace_back(a, b);
	}

	return candidates;
}
} // namespace Laphria::Physics
