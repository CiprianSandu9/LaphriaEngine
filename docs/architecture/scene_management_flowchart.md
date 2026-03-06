# Scene Management Flowchart

This flowchart visualizes the relationship between the `Scene`, `SceneNode` hierarchy, and the `Octree` spatial partitioning system. It demonstrates how transforms propagate and how nodes are queried for rendering.

```mermaid
flowchart TD
    %% Core Entities
    subgraph Core [Entities & Hierarchy]
        S[Scene] --> |owns| Root[Root SceneNode]
        S --> |flat array| AllNodes[allNodes std::vector]
        
        Root --> ChildA[SceneNode]
        Root --> ChildB[SceneNode]
        ChildA --> ChildA1[SceneNode]
        
        ChildA -.-> |"Update Pos/Rot/Scale"| updateLocal[Update Local Transform]
    end
    
    %% Transform Calculation
    subgraph Transforms [Transform Propagation]
        updateLocal --> getWorld[getWorldTransform]
        getWorld --> parentCheck{Has Parent?}
        parentCheck -- Yes --> getParentWorld[parent->getWorldTransform]
        getParentWorld --> multiply[Parent * Local]
        parentCheck -- No --> retLocal[Return Local Transform]
        
        multiply --> retWorld[Return World Transform]
    end
    
    %% Spatial Partitioning
    subgraph Spatial [Octree & Spatial Partitioning]
        S --> |owns| Oct[Octree Root]
        
        Oct --> insert[Insert SceneNode]
        insert --> boundsCheck{Inside Boundary?}
        boundsCheck -- No --> reject[Reject/Return False]
        boundsCheck -- Yes --> capCheck{Nodes < Capacity?}
        
        capCheck -- Yes --> pushNode[Add to this Node]
        capCheck -- No --> subCheck{Has Children?}
        
        subCheck -- No --> subdivide[Subdivide into 8 Octants]
        subdivide --> insertChild
        subCheck -- Yes --> insertChild[Insert into Child 0..7]
        insertChild --> pushNodeBoundary[If on boundary -> keep in parent]
    end
    
    %% Rendering / Querying
    frameEntry([recordCommandBuffer<br/><i>Rasterizer path, per frame</i>])

    subgraph QueryLoop [Frame Execution: Frustum Culling]
        frameEntry --> sceneDraw[Scene::draw<br/><i>cullBounds AABB</i>]
        sceneDraw --> oQuery[Octree::query]
        oQuery --> oBoundsCheck{Intersects Octant?}
        
        oBoundsCheck -- No --> cullDrop[Cull Branch]
        oBoundsCheck -- Yes --> checkNodes{Node in cullBounds?}
        
        checkNodes -- Yes --> addList[Add to Found List]
        oBoundsCheck -- Yes --> recurse[Recurse Children]
        
        addList --> passDraw[Pass to Renderer<br/><i>vkCmdDrawIndexed</i>]
    end
    
    %% Styling
    classDef core fill:#2A4365,stroke:#4299E1,stroke-width:2px,color:#fff
    classDef trans fill:#744210,stroke:#D69E2E,stroke-width:2px,color:#fff
    classDef space fill:#22543D,stroke:#48BB78,stroke-width:2px,color:#fff
    classDef query fill:#4A5568,stroke:#A0AEC0,stroke-width:2px,color:#fff
    
    class Core core
    class Transforms trans
    class Spatial space
    class QueryLoop query
```
