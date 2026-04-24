#ifndef LAPHRIAENGINE_SCENENODE_H
#define LAPHRIAENGINE_SCENENODE_H
#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>
#include <memory>
#include <string>
#include <vector>

class SceneNode : public std::enable_shared_from_this<SceneNode>
{
  public:
	using Ptr = std::shared_ptr<SceneNode>;

	SceneNode(const std::string &name = "Node");

	virtual ~SceneNode() = default;

	// Hierarchy
	[[nodiscard]] Ptr clone() const;

	void addChild(const Ptr &child);

	void removeChild(const Ptr &child);

	const std::vector<Ptr> &getChildren() const
	{
		return children;
	}

	SceneNode *getParent() const
	{
		return parent;
	}

	// Transform
	void setPosition(const glm::vec3 &pos)
	{
		position = pos;
		updateLocalTransform();
	}

	void setRotation(const glm::quat &rot)
	{
		rotation      = rot;
		eulerRotation = glm::degrees(glm::eulerAngles(rotation));        // Sync Euler from Quat
		updateLocalTransform();
	}

	void setEulerRotation(const glm::vec3 &eulerDegrees)
	{
		eulerRotation = eulerDegrees;
		rotation      = glm::quat(glm::radians(eulerRotation));        // Update Quat from Euler
		updateLocalTransform();
	}

	void setScale(const glm::vec3 &scl)
	{
		scale = scl;
		updateLocalTransform();
	}

	glm::vec3 getPosition() const
	{
		return position;
	}

	glm::quat getRotation() const
	{
		return rotation;
	}

	glm::vec3 getEulerRotation() const
	{
		return eulerRotation;
	}

	glm::vec3 getScale() const
	{
		return scale;
	}

	const glm::mat4 &getLocalTransform() const
	{
		return localTransform;
	}

	const glm::mat4 &getWorldTransform() const;

	glm::vec3 getWorldPosition() const
	{
		return glm::vec3(getWorldTransform()[3]);
	}

	// Recomputes cached world transforms in one top-down pass.
	void updateWorldTransformRecursive(const glm::mat4 &parentWorld, bool parentDirty) const;

	// Components (Simplified: just Mesh Indices for now)
	void addMeshIndex(int meshIndex)
	{
		meshIndices.push_back(meshIndex);
	}

	const std::vector<int> &getMeshIndices() const
	{
		return meshIndices;
	}

	std::string name;

	// For UI Interaction
	bool isSelected = false;

	// Index into ResourceManager::meshes
	std::vector<int> meshIndices;
	// Index into ResourceManager::models
	int modelId = -1;
	// Original glTF node index, used to map imported animation channels back to this node.
	int sourceNodeIndex = -1;

	enum class ColliderType
	{
		Sphere   = 0,
		Box      = 1,
		Cylinder = 2,
		None     = -1
	};

	struct PhysicsProperties
	{
		bool         enabled      = false;
		bool         isStatic     = false;
		ColliderType colliderType = ColliderType::None;
		float        mass         = 1.0f;
		float        friction     = 0.5f;
		float        restitution  = 0.5f;

		glm::vec3 velocity{0.0f};
		glm::vec3 acceleration{0.0f};

		// Collider Data
		float     radius = 1.0f;            // Sphere
		glm::vec3 halfExtents{0.5f};        // Box
	};

	PhysicsProperties physics;

	struct AssetReference
	{
		std::string path;
		std::string variant;
	};

	struct AnimationPlayback
	{
		bool        enabled = false;
		std::string clipId;
		float       timeSeconds = 0.0f;
		float       speed = 1.0f;
		bool        loop = true;
		bool        autoplay = true;
		bool        playing = true;
	};

	// Stable ID persisted in scene files to support deterministic references.
	std::string stableId;
	AssetReference assetRef;
	AnimationPlayback animation;

  protected:
	void updateLocalTransform();
	void markWorldTransformDirtyRecursive() const;

  public:
	// State Management
	void storeInitialState()
	{
		initialPosition  = position;
		initialRotation  = rotation;
		physics.velocity = glm::vec3(0.0f);
	}

	void resetToInitialState()
	{
		setPosition(initialPosition);
		setRotation(initialRotation);
		physics.velocity     = glm::vec3(0.0f);
		physics.acceleration = glm::vec3(0.0f);
	}

  private:
	SceneNode       *parent{nullptr};
	std::vector<Ptr> children;

	glm::vec3 position{0.0f};
	glm::quat rotation{1.0f, 0.0f, 0.0f, 0.0f};
	glm::vec3 eulerRotation{0.0f};
	glm::vec3 scale{1.0f};

	glm::vec3 initialPosition{0.0f};
	glm::quat initialRotation{1.0f, 0.0f, 0.0f, 0.0f};

	glm::mat4 localTransform{1.0f};
	mutable glm::mat4 worldTransform{1.0f};
	mutable bool worldTransformDirty{true};
};

#endif        // LAPHRIAENGINE_SCENENODE_H
