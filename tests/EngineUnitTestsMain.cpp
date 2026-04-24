#include "../src/Physics/Broadphase.h"
#include "../src/SceneManagement/Frustum.h"
#include "../src/SceneManagement/SceneNode.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <unordered_set>

#include <glm/gtc/matrix_transform.hpp>

namespace
{
bool approxEq(const glm::vec3 &a, const glm::vec3 &b, float eps = 1e-4f)
{
	return std::abs(a.x - b.x) <= eps &&
	       std::abs(a.y - b.y) <= eps &&
	       std::abs(a.z - b.z) <= eps;
}

uint64_t packPair(size_t a, size_t b)
{
	const uint32_t lo = static_cast<uint32_t>(std::min(a, b));
	const uint32_t hi = static_cast<uint32_t>(std::max(a, b));
	return (static_cast<uint64_t>(lo) << 32u) | static_cast<uint64_t>(hi);
}

bool intersects(const Laphria::Physics::AABBProxy &a, const Laphria::Physics::AABBProxy &b)
{
	return (a.min.x <= b.max.x && a.max.x >= b.min.x) &&
	       (a.min.y <= b.max.y && a.max.y >= b.min.y) &&
	       (a.min.z <= b.max.z && a.max.z >= b.min.z);
}

bool testWorldTransformCaching()
{
	auto root = std::make_shared<SceneNode>("root");
	auto child = std::make_shared<SceneNode>("child");

	root->setPosition(glm::vec3(1.0f, 2.0f, 3.0f));
	child->setPosition(glm::vec3(2.0f, 0.0f, 0.0f));
	root->addChild(child);

	root->updateWorldTransformRecursive(glm::mat4(1.0f), true);
	if (!approxEq(child->getWorldPosition(), glm::vec3(3.0f, 2.0f, 3.0f)))
	{
		std::cerr << "transform cache initial propagation failed\n";
		return false;
	}

	root->setPosition(glm::vec3(5.0f, 0.0f, 0.0f));
	root->updateWorldTransformRecursive(glm::mat4(1.0f), false);
	if (!approxEq(child->getWorldPosition(), glm::vec3(7.0f, 0.0f, 0.0f)))
	{
		std::cerr << "transform cache parent dirty propagation failed\n";
		return false;
	}

	child->setPosition(glm::vec3(4.0f, 0.0f, 0.0f));
	root->updateWorldTransformRecursive(glm::mat4(1.0f), false);
	if (!approxEq(child->getWorldPosition(), glm::vec3(9.0f, 0.0f, 0.0f)))
	{
		std::cerr << "transform cache child dirty update failed\n";
		return false;
	}

	return true;
}

bool testFrustumClassification()
{
	const glm::mat4 proj = glm::perspective(glm::radians(45.0f), 1.0f, 0.1f, 10.0f);
	const glm::mat4 view = glm::mat4(1.0f);
	const Laphria::Frustum frustum = Laphria::Frustum::fromViewProjection(proj * view);

	if (!frustum.containsPoint(glm::vec3(0.0f, 0.0f, -1.0f)))
	{
		std::cerr << "frustum failed to include in-front point\n";
		return false;
	}
	if (frustum.containsPoint(glm::vec3(0.0f, 0.0f, 1.0f)))
	{
		std::cerr << "frustum failed to cull behind-camera point\n";
		return false;
	}
	return true;
}

bool testBroadphaseCoverage()
{
	std::vector<Laphria::Physics::AABBProxy> proxies;
	proxies.push_back({0, {-1.0f, -1.0f, -1.0f}, {1.0f, 1.0f, 1.0f}});
	proxies.push_back({1, {0.5f, -1.0f, -1.0f}, {2.5f, 1.0f, 1.0f}});
	proxies.push_back({2, {8.0f, 8.0f, 8.0f}, {9.0f, 9.0f, 9.0f}});
	proxies.push_back({3, {1.8f, -1.0f, -1.0f}, {3.0f, 1.0f, 1.0f}});

	std::unordered_set<uint64_t> brutePairs;
	for (size_t i = 0; i < proxies.size(); ++i)
	{
		for (size_t j = i + 1; j < proxies.size(); ++j)
		{
			if (intersects(proxies[i], proxies[j]))
			{
				brutePairs.insert(packPair(proxies[i].id, proxies[j].id));
			}
		}
	}

	const auto candidates = Laphria::Physics::buildBroadphasePairs(proxies, 2.0f);
	std::unordered_set<uint64_t> candidatePairs;
	for (const auto &[a, b] : candidates)
	{
		candidatePairs.insert(packPair(a, b));
	}

	for (const uint64_t required : brutePairs)
	{
		if (!candidatePairs.contains(required))
		{
			std::cerr << "broadphase missed a true overlap pair\n";
			return false;
		}
	}
	return true;
}
} // namespace

int main()
{
	const bool okTransform = testWorldTransformCaching();
	const bool okFrustum = testFrustumClassification();
	const bool okBroadphase = testBroadphaseCoverage();
	return (okTransform && okFrustum && okBroadphase) ? 0 : 1;
}
