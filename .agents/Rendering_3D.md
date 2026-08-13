# WorldOf3D Graphics Pipeline & Architecture

The `WorldOf3D` rendering pipeline relies heavily on the core `matrix` library to bridge the gap between CPU metadata and GPU execution. Everything from vertex buffers to hierarchical 3D transforms is represented as a multithreaded, GPU-accelerated `matrix` tensor.

## The Core Paradigm: Structure vs. Execution
In WorldOf3D, C++ structs (`GeoNode`, `Mesh`, `Material`) are strictly heap-allocated "folders" used for organizational scoping and UI selection. They contain absolutely no math, logic, or raw float data.
The entire engine is orchestrated by the `matrix` class—a lazy-evaluated, 1-way tensor DAG (similar to MLX or PyTorch). A `matrix` is merely a 72-byte handle pointing to an execution tape. Because `Mesh` and `Material` only contain these handles, they are extremely lightweight (approx. 216 bytes) and are passed **strictly by value**, eliminating complex C++ pointer lifecycles and heap fragmentation.

## 1. Mesh
A `Mesh` defines the physical geometry of an object. Rather than using standard arrays, `Mesh` properties are backed directly by the `matrix` class, which allows them to be seamlessly uploaded to Metal buffers (`metalBuffer`) and operated on by compute shaders.

**Key Components:**
- `vert_position`: A 2D matrix of shape `[NumVertices, 3]` containing X, Y, Z coordinates.
- `uv_coords`: A 2D matrix of shape `[NumVertices, 2]` containing texture mapping coordinates.
- `indices`: A 1D matrix of `uint32_t` dictating the order vertices are drawn to form triangles.

*Common primitives (Triangle, Quad, Cube, Circle, Cylinder, Sphere) are provided dynamically via `MeshPrimitives::` factory functions.*

## 2. Materials
`Material` defines the visual appearance of a `Mesh`. 
- `colors`: A `matrix` of RGBA values that automatically broadcasts across instances and vertices using dynamic strides (`uint2 color_strides`). It supports exactly three shapes:
  - `[4]` (1D): A single solid color. Treated as `[1, 1, 4]` (1 instance, 1 vertex) with `instance_stride = 0`, `vertex_stride = 0`.
  - `[V, 4]` (2D): Per-vertex coloring. Treated as `[1, V, 4]` with `instance_stride = 0`, `vertex_stride = strides[0]/4`.
  - `[I, V, 4]` (3D): Per-instance and/or per-vertex coloring. Uses explicit `instance_stride = strides[0]/4` and `vertex_stride = strides[1]/4`. For example, `[N, 1, 4]` applies one unique color per instance.
- `texture`: An optional `id<MTLTexture>`.
- `has_texture`: A boolean flag determining the fragment shader's sampling behavior.
- `depth_bias`: A struct allowing meshes (like decals or wireframes) to apply a custom Polygon Offset during rendering to resolve Z-fighting.

## 3. GeoNode (The DAG)
The `GeoNode` is the fundamental building block of the 3D scene graph (DAG - Directed Acyclic Graph). It inherits from `std::enable_shared_from_this` and uses `std::shared_ptr` for memory safety.

**Properties:**
- `local_transform`: A 3D tensor of shape `[Instances, 4, 4]` representing the node's translation, rotation, and scale relative to its parent.
- `world_transform`: A 3D tensor of shape `[Instances, 4, 4]` representing the node's absolute position in the world. 
- `parent` & `children`: Weak and shared pointers defining the hierarchy.

**Combinatorial Tensor Broadcasting**
Transform resolution does not use standard 1-to-1 matrix multiplication. Because every node is treated as an instanced array `[N, 4, 4]`, the engine leverages ML-style tensor broadcasting to automatically calculate nested instances (e.g., placing a 10-instance Arrow onto a 100-instance Grid).
The DAG automatically forges this link via three steps:
1. **Unsqueeze:** Parent becomes `[ParentInstances, 1, 4, 4]`, Child becomes `[1, ChildInstances, 4, 4]`.
2. **Broadcasted MatMul:** The GPU expands the singular dimensions, yielding `[ParentInstances, ChildInstances, 4, 4]`.
3. **Flatten:** Dimensions 0 and 1 are collapsed to yield the final `[ParentInstances * ChildInstances, 4, 4]` world tensor.

## 4. Controllers
Controllers (e.g., `CubeController`, `SphereController`) are lightweight C++ wrappers around a `GeoNodeImpl`. They provide syntactic sugar for instantiating a `GeoNode`, attaching a specific `Mesh` and `Material`, and returning the encapsulated node to be added to the scene.

## 5. Viewers
The `Viewer` is responsible for traversing the `GeoNode` DAG and submitting it to the GPU.

**The 2-Pass Lazy Evaluation**
The Viewer does not brute-force matrix evaluations every frame. It utilizes a 2-pass system based on a `pass_id` counter to traverse the DAG:
* **Pass 1 (Invalidation):** Walks backward up the DAG from the outputs. If a dynamic variable (like time or a UI offset) has mutated, it marks dependent nodes as "dirty" up the chain.
* **Pass 2 (Execution):** Walks forward down the DAG. It only encodes new Metal Compute operations for the paths that were marked dirty, leaving static branches (like unchanging grid topologies) completely untouched by the CPU and GPU.

**Render Loop:**
1. **Flattening**: The `gather_nodes()` function recursively traverses the DAG and flattens the hierarchy into a linear `std::vector<GeoNodeImpl>`.
2. **Evaluation**: For each node, the viewer ensures all underlying `matrix` objects (`vert_position`, `world_transform`, etc.) are uploaded to the GPU using `ensure_metal()`.
3. **Instancing**: The renderer uses Metal's Instanced Drawing. `instanceCount` is determined dynamically by querying the shape of the `world_transform` matrix (`node->world_transform.shape()[0]`).
4. **Encoding**: Vertex buffers, UVs, Colors, and Transforms are bound to `id<MTLRenderCommandEncoder>` at specific indices before issuing `drawIndexedPrimitives`.

By treating transforms as N-dimensional `matrix` instances, the engine seamlessly scales from rendering a single cube to rendering millions of instances simply by expanding the first dimension of the `local_transform` matrix.

**Scene Management & Interactivity:**
Beyond just rendering, the `Viewer` acts as the central orchestrator for user interaction and structural scene mutation:
- **Hit-Testing & Selection**: The Viewer projects screen coordinates (e.g., from a mouse click) into 3D world space. This ray-casting identifies the specific `GeoNode` and its exact instance ID that the user interacted with.
- **Handling Drag Events**: When a user drags an object, the Viewer translates 2D screen deltas into 3D world-space translation vectors. For constrained movement (e.g., dragging along the XY, YZ, or ZX planes via Gizmos), the Viewer calculates the mathematical intersection of the mouse ray against an infinite 3D plane. This perfectly maps 2D mouse movement into a precise 2D surface within the 3D world.
- **Graph Mutation (Top-Down Injection)**: Instead of modifying heavy vertex buffers, the Viewer injects the translation delta directly into the targeted `GeoNode`'s `local_transform` matrix. Because this transform is a dynamic scalar node at the top of the DAG, updating its value instantly marks it as "dirty".
- **Triggering Evaluation**: On the next frame, the 2-Pass backprop catches this dirty flag, invalidates the specific branch of the graph, and recalculates the nested instance arrays on the GPU—allowing the user to drag a single parent object and have its millions of instanced children follow perfectly in real-time.

**UI Overlays & X-Ray Rendering:**
For editor tools like Transform Gizmos, the engine separates rendering into two distinct streams: `scene_nodes` and `gizmo_nodes`. To achieve an "X-Ray" effect where the gizmo is always visible through scene geometry without losing its own internal depth sorting (self-occlusion), the Viewer applies a mathematical projection trick. Rather than relying on Metal's `DepthBias` or disabling depth testing entirely, the Viewer passes a cloned View-Projection matrix to the gizmo's shader with the Z-coordinate components scaled down by 99% (e.g., `overlayMatrix.columns[2] *= 0.01f`). This artificially compresses the gizmo into a microscopic layer against the near clipping plane, forcing it to render entirely over the scene while flawlessly preserving the relative depths of its own components.

## 6. Execution Boundary Nodes (CPU-GPU Sync)
To support interactive logic (like modifying a buffer sequentially on the CPU mid-graph), the matrix DAG supports `insert_break()` primitives. When the evaluation pass hits a boundary node, it temporarily halts the pipeline, forces a synchronous commit to the GPU, safely executes a CPU lambda on the memory-shared buffer, and then spins up a new command encoder for the rest of the downstream graph. This gives the engine TouchDesigner-like procedural flexibility without breaking the continuous mathematical tape.

## 7. Architecture: One-Way (Pull) vs. Two-Way (Push) DAG
Traditional engines (like Maya or After Effects) utilize **Bi-Directional (Push) Graphs**, where nodes know both their parents and children. When a parent updates, it immediately pushes a "dirty" flag to all its children. While evaluation is fast, maintaining bi-directional pointers in C++ is highly prone to circular dependencies, requires heavy thread-locking, and makes dynamic graph restructuring brittle.

**WorldOf3D embraces a One-Way (Pull) Model**, identical to modern lazy-evaluated systems like PyTorch and JAX. A matrix node only knows its parents. 
- **The Advantage:** Graph mutation is virtually free and thread-safe. Re-parenting or injecting nodes involves overwriting a single smart pointer.
- **The Execution:** Instead of pushing updates, the `Viewer` pulls from the end of the pipeline. The 2-Pass Backprop system elegantly solves the one-way dilemma: Pass 1 walks backward from the output to probe for dirty states, and Pass 2 executes forward. This ensures the engine can dynamically restructure itself at 60 FPS without risking cyclic memory crashes.

