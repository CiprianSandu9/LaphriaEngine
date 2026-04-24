#ifndef LAPHRIAENGINE_FRUSTUM_H
#define LAPHRIAENGINE_FRUSTUM_H

#include <array>
#include <limits>

#include <glm/glm.hpp>

#include "Octree.h"

namespace Laphria
{
// View frustum represented as six normalized planes in world space.
// Plane equation: dot(plane.xyz, point) + plane.w >= 0 means inside.
struct Frustum
{
	std::array<glm::vec4, 6> planes{};

	[[nodiscard]] bool containsPoint(const glm::vec3 &point) const
	{
		for (const glm::vec4 &plane : planes)
		{
			if (glm::dot(glm::vec3(plane), point) + plane.w < 0.0f)
			{
				return false;
			}
		}
		return true;
	}

	static Frustum fromViewProjection(const glm::mat4 &viewProjection)
	{
		Frustum frustum{};

		const glm::vec4 row0{viewProjection[0][0], viewProjection[1][0], viewProjection[2][0], viewProjection[3][0]};
		const glm::vec4 row1{viewProjection[0][1], viewProjection[1][1], viewProjection[2][1], viewProjection[3][1]};
		const glm::vec4 row2{viewProjection[0][2], viewProjection[1][2], viewProjection[2][2], viewProjection[3][2]};
		const glm::vec4 row3{viewProjection[0][3], viewProjection[1][3], viewProjection[2][3], viewProjection[3][3]};

		frustum.planes[0] = row3 + row0; // left
		frustum.planes[1] = row3 - row0; // right
		frustum.planes[2] = row3 + row1; // bottom
		frustum.planes[3] = row3 - row1; // top
		frustum.planes[4] = row3 + row2; // near
		frustum.planes[5] = row3 - row2; // far

		for (glm::vec4 &plane : frustum.planes)
		{
			const float len = glm::length(glm::vec3(plane));
			if (len > 1e-6f)
			{
				plane /= len;
			}
		}

		return frustum;
	}

	static AABB computeAABB(const glm::mat4 &inverseViewProjection)
	{
		AABB bounds{
		    .min = glm::vec3(std::numeric_limits<float>::max()),
		    .max = glm::vec3(std::numeric_limits<float>::lowest())};

		constexpr float ndcX[2] = {-1.0f, 1.0f};
		constexpr float ndcY[2] = {-1.0f, 1.0f};
		constexpr float ndcZ[2] = {0.0f, 1.0f}; // GLM_FORCE_DEPTH_ZERO_TO_ONE

		for (float x : ndcX)
		{
			for (float y : ndcY)
			{
				for (float z : ndcZ)
				{
					glm::vec4 world = inverseViewProjection * glm::vec4(x, y, z, 1.0f);
					world /= world.w;
					bounds.min = glm::min(bounds.min, glm::vec3(world));
					bounds.max = glm::max(bounds.max, glm::vec3(world));
				}
			}
		}

		// Slight expansion avoids precision clipping at plane edges.
		constexpr glm::vec3 epsilon(0.5f);
		bounds.min -= epsilon;
		bounds.max += epsilon;
		return bounds;
	}
};
} // namespace Laphria

#endif // LAPHRIAENGINE_FRUSTUM_H
