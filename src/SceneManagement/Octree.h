#ifndef LAPHRIAENGINE_OCTREE_H
#define LAPHRIAENGINE_OCTREE_H

#include <vector>
#include <array>
#include <memory>
#include <glm/glm.hpp>
#include "SceneNode.h"

namespace Laphria {
    // Axis-aligned bounding box used for octree spatial tests and frustum culling.
    struct AABB {
        glm::vec3 min;
        glm::vec3 max;

        // Returns true if point lies strictly inside or on the boundary.
        bool contains(const glm::vec3 &point) const {
            return point.x >= min.x && point.x <= max.x &&
                   point.y >= min.y && point.y <= max.y &&
                   point.z >= min.z && point.z <= max.z;
        }

        // SAT (Separating Axis Theorem) overlap test for two AABBs.
        bool intersects(const AABB &other) const {
            return (min.x <= other.max.x && max.x >= other.min.x) &&
                   (min.y <= other.max.y && max.y >= other.min.y) &&
                   (min.z <= other.max.z && max.z >= other.min.z);
        }
    };

    // Loose octree for spatial indexing of SceneNodes.
    // Subdivides a node into 8 equal children when it reaches 'capacity' entries.
    // Nodes that do not fit into any child (e.g. on a boundary) remain in the parent.
    // Usage: insert all scene nodes after each frame update, then query with a view frustum AABB.
    class Octree {
    public:
        Octree(const AABB &boundary, int capacity = 4) : boundary(boundary), capacity(capacity) {
        }

        // Inserts node if its world position falls within this node's boundary.
        // Returns false if the position is outside (caller should not retry on a parent).
        bool insert(const SceneNode::Ptr &node) {
            if (!boundary.contains(node->getWorldPosition())) {
                return false;
            }

            if (nodes.size() < capacity && children[0] == nullptr) {
                nodes.push_back(node);
                return true;
            }

            if (children[0] == nullptr) {
                subdivide();
            }

            for (auto &child: children) {
                if (child->insert(node)) {
                    return true;
                }
            }

            // Node does not fit into any child (world-position on octant boundary); keep here.
            nodes.push_back(node);
            return true;
        }

        // Appends to 'found' all nodes whose world position falls inside 'range'.
        void query(const AABB &range, std::vector<SceneNode::Ptr> &found) const {
            if (!boundary.intersects(range)) {
                return;
            }

            for (const auto &node: nodes) {
                if (range.contains(node->getWorldPosition())) {
                    // Simple point check
                    found.push_back(node);
                }
            }

            if (children[0] != nullptr) {
                for (const auto &child: children) {
                    child->query(range, found);
                }
            }
        }

        // Clear the tree
        void clear() {
            nodes.clear();
            for (auto &child: children) {
                child = nullptr;
            }
        }

        const AABB &getBounds() const { return boundary; }

    private:
        AABB boundary;
        int capacity;
        std::vector<SceneNode::Ptr> nodes;
        std::array<std::unique_ptr<Octree>, 8> children;

        void subdivide() {
            glm::vec3 min = boundary.min;
            glm::vec3 max = boundary.max;
            glm::vec3 center = (min + max) * 0.5f;

            // Create 8 children
            // Bottom (Y min)
            children[0] = std::make_unique<Octree>(AABB{min, center}, capacity);
            children[1] = std::make_unique<Octree>(AABB{glm::vec3(center.x, min.y, min.z), glm::vec3(max.x, center.y, center.z)}, capacity);
            children[2] = std::make_unique<Octree>(AABB{glm::vec3(min.x, min.y, center.z), glm::vec3(center.x, center.y, max.z)}, capacity);
            children[3] = std::make_unique<Octree>(AABB{glm::vec3(center.x, min.y, center.z), glm::vec3(max.x, center.y, max.z)}, capacity);

            // Top (Y max)
            children[4] = std::make_unique<Octree>(AABB{glm::vec3(min.x, center.y, min.z), glm::vec3(center.x, max.y, center.z)}, capacity);
            children[5] = std::make_unique<Octree>(AABB{glm::vec3(center.x, center.y, min.z), glm::vec3(max.x, max.y, center.z)}, capacity);
            children[6] = std::make_unique<Octree>(AABB{glm::vec3(min.x, center.y, center.z), glm::vec3(center.x, max.y, max.z)}, capacity);
            children[7] = std::make_unique<Octree>(AABB{center, max}, capacity);
        }
    };
} // namespace Laphria

#endif //LAPHRIAENGINE_OCTREE_H
