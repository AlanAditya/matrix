# WorldOf3D Graphics Pipeline & Architecture

The `WorldOf3D` rendering pipeline relies heavily on the core `matrix` library to bridge the gap between CPU metadata and GPU execution. Everything from vertex buffers to hierarchical 3D transforms is represented as a multithreaded, GPU-accelerated `matrix` tensor.

> **This is the modern (current) rendering path.** For the two superseded systems it replaced (`Shape<T>` and `GeometryNode<T>`, both still physically present in `MatrixH.mm` / `Mods/GeometryNode.mm`), see [Legacy_Rendering.md](Legacy_Rendering.md). Do not model new code on them — see that doc's "Do not copy" section.

## File Map

| File | Contains |
|---|---|
| [SOPs/GeoNode.cpp](../SOPs/GeoNode.cpp) | `Mesh`, `Material`, `Topology`, `DepthBias`, `GeoNode`/`GeoNodeImpl` — the scene graph node itself |
| [SOPs/Viewer.cpp](../SOPs/Viewer.cpp) | `Viewer` — the base renderer/orchestrator: draw loop, ray hit-testing, drag/gizmo interaction |
| [SOPs/Controllers.cpp](../SOPs/Controllers.cpp) | Thin wrapper classes (`CubeController`, `LineController`, `GizmoController`, `ModelController`, `GridController`, ...) that build a `GeoNodeImpl` with a specific `Mesh`/`Material` preset |
| [SOPs/MeshPrimitives.cpp](../SOPs/MeshPrimitives.cpp) | Pure functions returning a `Mesh` (or edge-index `matrix`) for primitive shapes: `triangle`, `quad`, `cube`, `circle`, `sphere`, `cylinder`, `cone`, `quad_edges` |
| [SOPs/GraphViewer.cpp](../SOPs/GraphViewer.cpp) | `GraphViewer : Viewer` — concrete example of subclassing `Viewer` for a 2D function-plotting scene |
| [SOPs/PlotterViewer.h](../SOPs/PlotterViewer.h) | `PlotterViewer : Viewer` — concrete example for texture-based (heatmap) plotting |
| [Shaders.metal](../Shaders.metal) | `DAGNodeVertexShader` / `DAGNodeFragmentShader` — the vertex shader for pipeline_state 3 (SOA mesh) |
| [DAGShaders.metal](../DAGShaders.metal) | `vertex_line3d`, `vertex_dag_pointcloud`, `vertex_edge3d` — shaders for pipeline_states 4, 5, 6 |

## The Core Paradigm: Structure vs. Execution
In WorldOf3D, C++ structs (`GeoNode`, `Mesh`, `Material`) are strictly heap-allocated "folders" used for organizational scoping and UI selection. They contain absolutely no math, logic, or raw float data.
The entire engine is orchestrated by the `matrix` class—a lazy-evaluated, 1-way tensor DAG (similar to MLX or PyTorch). A `matrix` is merely a 72-byte handle pointing to an execution tape. Because `Mesh` and `Material` only contain these handles, they are extremely lightweight (approx. 216 bytes) and are passed **strictly by value**, eliminating complex C++ pointer lifecycles and heap fragmentation.

This is the direct opposite of the legacy `Shape`/`GeometryNode` approach, which owned raw `new[]`'d vertex/index arrays and manually-managed `simd_float4x4*` instance-matrix buffers per node (see [Legacy_Rendering.md](Legacy_Rendering.md)).

## 1. Mesh
A `Mesh` defines the physical geometry of an object. Rather than using standard arrays, `Mesh` properties are backed directly by the `matrix` class, which allows them to be seamlessly uploaded to Metal buffers (`metalBuffer`) and operated on by compute shaders.

**Key Components:**
- `vert_position`: A 2D matrix of shape `[NumVertices, 3]` containing X, Y, Z coordinates.
- `uv_coords`: A 2D matrix of shape `[NumVertices, 2]` containing texture mapping coordinates.
- `indices`: A 1D matrix of `uint32_t` dictating the order vertices are drawn to form triangles.

*Common primitives (Triangle, Quad, Cube, Circle, Cylinder, Cone, Sphere) are provided dynamically via `MeshPrimitives::` factory functions in [SOPs/MeshPrimitives.cpp](../SOPs/MeshPrimitives.cpp). Each is a pure function that returns a fully-formed `Mesh` value — no allocation ownership games, no shared state.*

## 2. Materials
`Material` defines the visual appearance of a `Mesh`. 
- `colors`: A `matrix` of RGBA values that automatically broadcasts across instances and vertices using dynamic strides (`uint2 color_strides`). It supports exactly three shapes:
  - `[4]` (1D): A single solid color. Treated as `[1, 1, 4]` (1 instance, 1 vertex) with `instance_stride = 0`, `vertex_stride = 0`.
  - `[V, 4]` (2D): Per-vertex coloring. Treated as `[1, V, 4]` with `instance_stride = 0`, `vertex_stride = strides[0]/4`.
  - `[I, V, 4]` (3D): Per-instance and/or per-vertex coloring. Uses explicit `instance_stride = strides[0]/4` and `vertex_stride = strides[1]/4`. For example, `[N, 1, 4]` applies one unique color per instance.
- `texture`: An optional `id<MTLTexture>`.
- `has_texture`: A boolean flag determining the fragment shader's sampling behavior.
- `depth_bias`: A struct allowing meshes (like decals or wireframes) to apply a custom Polygon Offset during rendering to resolve Z-fighting.
- `clip_min` / `clip_max`: An axis-aligned world-space box. `DAGNodeFragmentShader` (and the other DAG fragment shaders) `discard_fragment()` any pixel whose `WorldPos` falls outside this box — this is how `GraphViewer` clips plotted geometry to the graph's frame without a scissor rect.
- `pipeline_state`: Selects which of the 7 shared pipeline slots (see table below) renders this node.

**`Material()`'s default constructor already gives every new `GeoNode` a solid-white `[1,4]` color** — controllers rarely need to set color explicitly unless they want something else.

## 3. GeoNode (The DAG)
The `GeoNode` is the fundamental building block of the 3D scene graph (DAG - Directed Acyclic Graph), defined in [SOPs/GeoNode.cpp](../SOPs/GeoNode.cpp). It inherits from `std::enable_shared_from_this` and uses `std::shared_ptr` (aliased `GeoNodeImpl`) for memory safety. Its constructor is `private`; the only way to make one is `GeoNode::create(name)`, which guarantees every instance is heap-allocated behind a `shared_ptr`.

**Properties:**
- `local_transform`: A 3D tensor of shape `[Instances, 4, 4]` representing the node's translation, rotation, and scale relative to its parent.
- `world_transform`: A 3D tensor of shape `[Instances, 4, 4]` representing the node's absolute position in the world. 
- `mesh` / `topology` / `material`: value members, not pointers — copying a `GeoNode` copies matrix *handles*, not the underlying data.
- `draggable`: gates both ray hit-testing (`ray_hit_closest` skips non-draggable nodes) and gizmo drag targeting.
- `parent` & `children`: Weak and shared pointers defining the hierarchy.

**Structural Mutation API:**
- `add_child(child)` — detaches `child` from any existing parent, reparents it to `this`, and recursively rebuilds `child`'s (and its descendants') `world_transform` links.
- `set_local_transform(m)` — replaces `local_transform` and immediately recomputes `world_transform` down the subtree.
- `invalidate()` — call this after mutating a transform's underlying buffer *in place* (e.g. via `SIMD_MAT(i) = ...` rather than reassigning the whole matrix) so the DAG's dirty-tracking (`clear_trace_checks()`) knows to re-walk the subtree. Reassigning `local_transform` wholesale (as `set_local_transform` does) doesn't need this; mutating in place does.

**Combinatorial Tensor Broadcasting**
Transform resolution does not use standard 1-to-1 matrix multiplication. Because every node is treated as an instanced array `[N, 4, 4]`, the engine leverages ML-style tensor broadcasting to automatically calculate nested instances (e.g., placing a 10-instance Arrow onto a 100-instance Grid).
`update_world_transform_link()` forges this link via three steps:
1. **Unsqueeze:** Parent becomes `[ParentInstances, 1, 4, 4]`, Child becomes `[1, ChildInstances, 4, 4]`.
2. **Broadcasted MatMul:** `child_broad.dot(parent_broad)` — the GPU expands the singular dimensions, yielding `[ParentInstances, ChildInstances, 4, 4]`. Note the engine uses row-vector convention (`v * M`), so the child's local transform is applied first: `M_combined = M_local * M_parent`.
3. **Flatten:** Dimensions 0 and 1 are collapsed via `.flatten(0, 1)` to yield the final `[ParentInstances * ChildInstances, 4, 4]` world tensor.

A root node (no parent) just aliases `world_transform = local_transform`.

## 4. Controllers
Controllers ([SOPs/Controllers.cpp](../SOPs/Controllers.cpp), e.g. `CubeController`, `SphereController`, `LineController`, `PointCloudController`, `GizmoController`, `ModelController`, `GridController`) are lightweight C++ wrappers around a `GeoNodeImpl`. They provide syntactic sugar for instantiating a `GeoNode`, attaching a specific `Mesh` (usually from `MeshPrimitives::`) and `Material`, and returning the encapsulated node to be added to the scene. A controller is not itself part of the DAG — only the `.node` (or, for compound controllers like `GizmoController`, the several `GeoNodeImpl`s it holds and wires together with `add_child`) is.

Some controllers also expose an `update(...)` method that regenerates the node's `mesh.vert_position`/`indices`/`material` from new data at runtime (e.g. `LineController::update(matrix points)`, `PointCloudController::update(...)`). These typically dispatch the mutation onto the main queue via `dispatch_async(dispatch_get_main_queue(), ...)` to avoid racing the render thread, which reads the same `GeoNode` fields during `Viewer::draw`.

`GridController` is the most involved example: it drives two multi-instance line nodes (`major_lines`, `minor_lines`) whose *per-instance* `local_transform` entries are recomputed every frame in `update_grid()` based on camera zoom and graph pan/scale, implementing a Desmos-style infinite adaptive grid without ever touching vertex data.

## 5. Viewers
The `Viewer` ([SOPs/Viewer.cpp](../SOPs/Viewer.cpp)) is responsible for traversing the `GeoNode` DAG and submitting it to the GPU. It owns a top-level `std::vector<GeoNodeImpl> nodes` (the scene's roots), a `GizmoController transform_gizmo`, and the currently-selected `active_node`.

**Pipeline state slots**

`draw()` is handed a shared array `id<MTLRenderPipelineState> predefinedStates[7]`, indices 0-6. Slots 0-2 belong to the *legacy* `GeometryNode`/`Renderer` path (see [Legacy_Rendering.md](Legacy_Rendering.md)) and are never touched by `Viewer`. `Viewer` only ever uses slots 3-6, selected per-node via `node->material.pipeline_state`:

| `pipeline_state` | Meaning | Shader | Draw call |
|---|---|---|---|
| 3 | SOA Mesh (default) | `DAGNodeVertexShader` / `DAGNodeFragmentShader` ([Shaders.metal](../Shaders.metal)) | `drawIndexedPrimitives` (triangles, uses `mesh.indices`) |
| 4 | SOA Thick Line | `vertex_line3d` / `fragment_line3d` ([DAGShaders.metal](../DAGShaders.metal)) | `drawPrimitives` triangle-strip, `pointCount * 2` verts (camera-facing billboarded line, width from `material.line_width`) |
| 5 | SOA Point Cloud | `vertex_dag_pointcloud` / `fragment_dag_pointcloud` | `drawPrimitives` point list |
| 6 | SOA Edge Topology | `vertex_edge3d` / `fragment_edge3d` | `drawPrimitives` triangles, `num_edges * 6` verts, reads `node->topology.edges` as a second buffer — used for wireframe/edge overlays that need constant screen-space width |

`split_nodes()` buckets the flattened node list into `mesh_nodes` / `line_nodes` / `pc_nodes` / `edge_nodes` by this same value so each bucket can be drawn back-to-back without redundant `setRenderPipelineState:` calls.

**The 2-Pass Lazy Evaluation**
The Viewer does not brute-force matrix evaluations every frame. It utilizes a 2-pass system based on a `pass_id` counter (`current_pass_id`) to traverse the DAG:
* **Pass 1 (Invalidation):** `invalidate_nodes()` walks the flattened node list and calls `tape->invalidate_pass(current_pass_id)` on every underlying matrix (`vert_position`, `world_transform`, `material.colors`, `uv_coords`, `indices`). If a dynamic variable (like time or a UI offset) has mutated upstream, this marks dependent nodes as "dirty" so Pass 2 knows to re-evaluate them.
* **Pass 2 (Execution):** `draw_nodes()` walks forward, calling `.eval()` on each matrix (a no-op if already valid this pass) and lazily building/uploading the Metal buffer the first time (`ensure_metal`: `if (mat.total_size > 0 && mat.metalBuffer == nullptr) mat.buildMetalBuffer();`). Static branches (like unchanging grid topologies) are left completely untouched by the CPU and GPU.

**Render Loop (`Viewer::draw`):**
1. **Flattening**: `gather_nodes()` recursively traverses the DAG and flattens each root's hierarchy into a linear `std::vector<GeoNodeImpl>`; all roots' flattened lists are concatenated into `scene_nodes`.
2. **Bucketing**: `split_nodes()` splits `scene_nodes` into the four pipeline-state buckets described above.
3. **Invalidation**: `invalidate_nodes()` runs on every bucket (plus the gizmo's buckets, gathered separately from `transform_gizmo.node`).
4. **Evaluation + Encoding**: `draw_nodes(bucket, viewMatrix)` evaluates each node's matrices, ensures Metal buffers exist, sets the pipeline state (only if it changed since the last node — `active_state` tracks this to skip redundant `setRenderPipelineState:` calls), binds vertex buffers (`vert_position` @0, `world_transform` @1, view matrix bytes @2, `colors` @3 + `color_strides` @4, `uv_coords` @5, `local_transform` @9 + instance count @10) and fragment bytes (`isTextured` @0, `clip_min`/`clip_max` @1/@2), then issues the draw call appropriate to `active_state`.
5. **Instancing**: The renderer uses Metal's Instanced Drawing. `instanceCount` is determined dynamically by querying the shape of the `world_transform` matrix (`node->world_transform.shape()[0]`) — this is what lets a single `GeoNode` render millions of copies of the same mesh by simply expanding `local_transform`'s first dimension.

By treating transforms as N-dimensional `matrix` instances, the engine seamlessly scales from rendering a single cube to rendering millions of instances simply by expanding the first dimension of the `local_transform` matrix.

**Scene Management & Interactivity:**
Beyond just rendering, the `Viewer` acts as the central orchestrator for user interaction and structural scene mutation, all funneled through `handle_event(const ViewerEvent&)`:
- **Hit-Testing & Selection (`ray_hit_closest`)**: The Viewer projects screen coordinates (e.g., from a mouse click) into 3D world space. This ray-casting identifies the specific `GeoNode` and its exact instance ID that the user interacted with. It CPU-side reads back `vert_position`/`indices` (`.eval_metal()` / `.eval_cpu()`), transforms each triangle by that instance's `world_transform.SIMD_MAT(inst)`, and does a plane/barycentric-style inside-triangle test (`point_inside_triangle`) per triangle. Thick-line nodes (`pipeline_state == 4`) instead do a closest-point-on-segment test against `material.line_width`. Nodes with `draggable == false` or an empty mesh are skipped entirely.
- **Handling Drag Events (`DragAxis`)**: When a user drags an object, the Viewer translates 2D screen deltas into 3D world-space translation vectors. For constrained movement (e.g., dragging along the X/Y/Z axis or the XY, YZ, or ZX planes via the gizmo), the Viewer calculates the mathematical intersection of the mouse ray against an infinite 3D line/plane (`closest_point_on_lines`, `ray_plane_dir`). This perfectly maps 2D mouse movement into a precise 1D/2D constraint in 3D world space.
- **Graph Mutation (Top-Down Injection)**: Instead of modifying heavy vertex buffers, the Viewer injects the translation delta directly into the targeted `GeoNode`'s `local_transform` matrix (`active_node->local_transform.SIMD_MAT(0) = ... * Translation(delta_pos)`), then bumps `local_transform.tape->version = ++global_epoch` to mark it dirty. Because this transform is a dynamic scalar node at the top of the DAG, updating its value instantly marks it as "dirty".
- **Triggering Evaluation**: On the next frame, the 2-Pass backprop catches this dirty flag, invalidates the specific branch of the graph, and recalculates the nested instance arrays on the GPU — allowing the user to drag a single parent object and have its millions of instanced children follow perfectly in real-time.

**UI Overlays & X-Ray Rendering (Gizmo):**
For editor tools like Transform Gizmos, the engine separates rendering into two distinct streams: `scene_nodes` (gathered from `Viewer::nodes`) and `gizmo_nodes` (gathered from `transform_gizmo.node`, only when `active_node` is set). At the top of `draw()`, the gizmo is re-scaled and re-positioned every frame to sit on `active_node`'s world position and stay a constant apparent size regardless of camera distance (`Scale(0.1 * zoom_dist)`). To achieve an "X-Ray" effect where the gizmo is always visible through scene geometry without losing its own internal depth sorting (self-occlusion), the Viewer applies a mathematical projection trick when drawing the gizmo buckets: rather than relying on Metal's `DepthBias` or disabling depth testing entirely, it passes a cloned view-projection matrix (`overlayMatrix`) with the Z-coordinate components of every column scaled down by 99% (`overlayMatrix.columns[i][2] *= 0.01f`). This artificially compresses the gizmo into a microscopic layer against the near clipping plane, forcing it to render entirely over the scene while flawlessly preserving the relative depths of its own components.

## 6. Subclassing Viewer (worked examples)
`Viewer` is designed to be subclassed for scene-specific behavior by overriding `draw()` (to inject per-frame dynamic updates before calling `Viewer::draw(...)`) and/or `handle_event()` (to intercept interaction before/instead of the base gizmo logic).

- **[GraphViewer](../SOPs/GraphViewer.cpp)** — builds a 2D function-plotting scene: a clickable `graph_frame` quad (pipeline_state 6, edge topology, for the visible border), a `graph_node` that owns a `GridController` as a child (so the grid inherits the graph's pan/zoom transform) and clips to `clip_min`/`clip_max`, and axis handling. It overrides `draw()` to recompute the grid's cell sizes every frame from camera distance and the graph's current scale/offset (decomposed straight out of `graph_node->local_transform`'s columns), and overrides `handle_event()` to implement pan (`Drag`) and cursor-anchored zoom (`Zoom`) by directly composing `Translation`/`Scale` matrices onto `graph_node->local_transform`. `render_graph(x, y, color, line_width)` feeds new plotted points into `graph_node` via a `LineController`-style `pipeline_state = 4` update, dispatched onto the main queue.
- **[PlotterViewer](../SOPs/PlotterViewer.h)** — a texture-based (heatmap) plotter: three static `QuadController`s (plot image, gradient scale bar, frame) built once in the constructor. `update_plot(amplitude_map, ...)` normalizes a `[H, W]` float matrix, maps it through an Inferno-style colormap (`get_colormap_inferno`, with both a scalar-loop CPU path for small inputs and a vectorized `matrix`-based path via `apply_colormap`/`matrix::take` for large ones), and assigns the result as `nodes[0]->material.texture`. Demonstrates that a `Viewer` subclass doesn't have to touch vertex geometry at all — it can drive everything through `Material.texture`.

## 7. Execution Boundary Nodes (CPU-GPU Sync)
To support interactive logic (like modifying a buffer sequentially on the CPU mid-graph), the matrix DAG supports `insert_break()` primitives. When the evaluation pass hits a boundary node, it temporarily halts the pipeline, forces a synchronous commit to the GPU, safely executes a CPU lambda on the memory-shared buffer, and then spins up a new command encoder for the rest of the downstream graph. This gives the engine TouchDesigner-like procedural flexibility without breaking the continuous mathematical tape.

## 8. Architecture: One-Way (Pull) vs. Two-Way (Push) DAG
Traditional engines (like Maya or After Effects) utilize **Bi-Directional (Push) Graphs**, where nodes know both their parents and children. When a parent updates, it immediately pushes a "dirty" flag to all its children. While evaluation is fast, maintaining bi-directional pointers in C++ is highly prone to circular dependencies, requires heavy thread-locking, and makes dynamic graph restructuring brittle.

**WorldOf3D embraces a One-Way (Pull) Model**, identical to modern lazy-evaluated systems like PyTorch and JAX. A matrix node only knows its parents. 
- **The Advantage:** Graph mutation is virtually free and thread-safe. Re-parenting or injecting nodes involves overwriting a single smart pointer.
- **The Execution:** Instead of pushing updates, the `Viewer` pulls from the end of the pipeline. The 2-Pass Backprop system elegantly solves the one-way dilemma: Pass 1 walks backward from the output to probe for dirty states, and Pass 2 executes forward. This ensures the engine can dynamically restructure itself at 60 FPS without risking cyclic memory crashes.

This is also the fundamental structural difference from legacy `GeometryNode`, whose `BuildInstances()` is a **push**-style, eagerly-recomputed-every-call recursive walk over raw `simd_float4x4*` arrays with no dirty tracking at all (see [Legacy_Rendering.md](Legacy_Rendering.md) §3).
