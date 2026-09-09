# Legacy Rendering: `Shape<T>` and `GeometryNode<T>`

**Status: legacy / frozen.** Both systems documented here are superseded by the `GeoNode` + `Viewer` matrix-DAG pipeline described in [Rendering_3D.md](Rendering_3D.md). They are kept only because parts of `Renderer` (in `MatrixH.mm`) still construct and draw them. **Do not model new rendering code on either of these** — if you're adding a shape, a controller, or scene-graph behavior, follow the `GeoNode`/`Controllers` pattern instead (see [Rendering_3D.md](Rendering_3D.md) §4).

This doc exists so the two legacy systems can be understood, safely touched when necessary (e.g. bug fixes, migration work), and eventually retired.

## Why two legacy systems, not one

They were built in sequence, each an attempt to fix the previous one's biggest limitation, before the project moved to the `matrix`-DAG approach entirely:

1. **`Shape<T>`** (in `MatrixH.mm`) — a single flat drawable object. No hierarchy at all: every shape is its own root, transformed and drawn independently.
2. **`GeometryNode<T>`** (in [Mods/GeometryNode.mm](../Mods/GeometryNode.mm)) — adds a parent/child hierarchy and GPU instancing on top of the same "own raw arrays" philosophy, so scenes could be built compositionally. This is structurally the direct ancestor of `GeoNode`, but implements hierarchy and instancing by hand with raw `simd_float4x4*` arrays instead of `matrix` tensors.

Both share the same underlying philosophy that `GeoNode` deliberately abandoned: **a shape struct owns its raw vertex/index data and its own Metal buffers**, rather than being a lightweight handle into a shared tensor DAG.

---

## 1. `Shape<T>` (MatrixH.mm:8560)

```cpp
template<typename T>
class Shape {
public:
    Vertex3D* Verticies = nil;   int vertexCount;
    T* indices = nil;            int indexCount;
    id<MTLBuffer> vertexBuffer, indexBuffer;
    id<MTLTexture> texture;
    simd_float3 position, scale, rotation;   // Euler angles
    bool update, textured, dragable;
    MTLPrimitiveType drawType = MTLPrimitiveTypeTriangle;
    std::vector<std::function<void(simd_float3)>> transformChangeCallbacks;
};
```

- **Ownership**: `Verticies`/`indices` are raw `new[]`'d C arrays. `buildBuffers()` wraps them with `newBufferWithBytesNoCopy:...deallocator:` — the Metal buffer's deallocator block does the `delete[]`, so the `Shape`'s CPU array and its `MTLBuffer` share one lifetime, freed only once the GPU buffer is released.
- **Transform**: plain Euler `position`/`scale`/`rotation` (no hierarchy). `Transformer()` composes `Translation * RotationX * RotationY * RotationZ * Scale` fresh every call — recomputed on demand, not cached.
- **Hit-testing**: `intersectRay(...)` does its own CPU triangle-vs-ray test in clip space (project each triangle's 3 verts through `cam * transform`, then a 2D signed-area/edge test against the click point) — a self-contained, non-DAG predecessor of `Viewer::ray_hit_closest`.
- **Drag callbacks**: `transformChangeCallbacks` is a manual observer list (`triggerCallbacks()`), the ancestor of the modern gizmo-driven `ViewerEvent` model.
- **Concrete subclasses** (all `MatrixH.mm`, ~8684-10520): `Triangle`, `Quad`, `Circle`, `Pipe`, `Cone`, `Cube`, `ConvexPolygon`, `ConcavePolygon`, `Line`, `Text3D` — each just pre-fills `Verticies`/`indices` for its topology in its constructor. These are the direct ancestors of `MeshPrimitives::triangle/quad/circle/cube/cylinder/cone`, minus the "just return a value, no ownership" property.
- **`ArrayShape`** (`MatrixH.mm:10520`): bolts a `simd_float4x4* transform` array + `instanceCount` onto one `Shape`, giving basic instancing (`MakeModifierArray`, `MakeModifierArrayRadial`, `buildLineArray`, `buildDottedLine` build the transform array for grids/radial arrays/dotted lines). This is the shape-level ancestor of `GeoNode`'s instanced `local_transform`.

**Where it's drawn**: `Renderer` (`@implementation Renderer`, `MatrixH.mm:10873`) keeps `std::vector<Shape<uint16>> _objectQueue` and `_ComposedObjectQueue`, plus a parallel `_objectQueueInstanced` of `ArrayShape`. `drawInMTKView:` (`MatrixH.mm:11341`) loops each queue and, per shape: `buildBuffers()`, compute `Transformer()`, bind `vertexBuffer`/index buffer/transform bytes, `drawIndexedPrimitives`. Dragging (`MatrixH.mm:~11863`) is done by iterating `_objectQueue` by hand, calling `intersectRay` on each, and mutating `.position` directly — an O(n), no-hierarchy version of `Viewer::handle_event`.

## 2. `GeometryNode<T>` ([Mods/GeometryNode.mm](../Mods/GeometryNode.mm):97)

The hierarchical successor to `Shape`. Structurally it *is* what `GeoNode` is conceptually — a named node with a local transform, children, and a mesh — but everything is hand-rolled with raw pointers instead of `matrix` tensors, and it carries a lot of scar tissue from that.

```cpp
template<typename T>
class GeometryNode {
public:
    std::vector<GeometryNode<T>> childNodes;   // owned BY VALUE, not shared_ptr
    void* Verticies;  int vertexCount;
    T* indices;       int indexCount;
    id<MTLBuffer> vertexBuffer, indexBuffer, transformBuffer;
    simd_float3 position, scale, rotation;
    simd_float4x4 modelMatrix, globalMatrix;
    simd_float4x4* instanceMatricies;           // this node's own N instances
    simd_float4x4* parentInstanceMatricies;     // pointer INTO the parent's flattened array
    simd_float4x4* preMulparentInstanceMatricies; // this node's flattened [parent*self] result
    uint32_t instances, parentInstances;
};
```

### Key structural differences from `GeoNode`

- **Children are owned by value** (`std::vector<GeometryNode<T>>`, not `std::vector<GeoNodeImpl>`/`shared_ptr`). This is the root cause of the biggest documented footgun in the file (see below): a `GeometryNode` moved/copied into a `std::vector` is indistinguishable, to the compiler, from "please treat this as a batch of children to merge."
- **No `matrix` tensors anywhere.** Instancing is done with hand-allocated `simd_float4x4[]` arrays and manual `new[]`/`delete[]` bookkeeping in `BuildInstances()`, `buildInstanceFromModifier()`, `buildInstanceFromBuffer()`.
- **Push-style, eager, unconditional recompute.** `BuildInstances()` recursively walks the *entire* subtree and recomputes every instance matrix on every call — there is no dirty-flag/tape system like the `matrix` DAG's 2-pass invalidate/evaluate. Compare to `GeoNode::update_world_transform_recursive()`, which is structurally identical in shape but pushes `matrix` handles instead of eagerly multiplying floats, and to `Viewer`'s pass-based dirty tracking, which avoids recomputation entirely when nothing changed.
- **Draw is a per-instance big if/else state machine** (`draw()`, line 597): branches on `renderPipelineType` (`Custom` vs predefined) and then on `RenderStateNo` (`Mesh` / `PointCloud` / `Billboard`, `PredefinedRenderPipelineState` in [Mods/Utils.h](../Mods/Utils.h)) to decide which of `predefinedStates[0..2]` to bind and which draw call to issue — the direct ancestor of `Viewer::draw_nodes`'s cleaner `pipeline_state`-indexed dispatch over slots 3-6.

### The variadic-constructor footgun (documented in-file, worth preserving)

```cpp
template<typename... Children, typename = std::enable_if_t<(std::is_base_of_v<GeometryNode<T>, std::decay_t<Children>> && ...)>>
GeometryNode(Children&&... children) requires (sizeof...(Children) > 1)
```

The `sizeof...(Children) > 1` constraint exists because, without it, this "build a parent from a pack of children" constructor is a better overload match than the copy/move constructor whenever a single `GeometryNode` is passed by value — e.g. `std::vector<GeometryNode<T>>::push_back(someNode)`. The compiler would treat `someNode` as *one child to adopt* rather than *the object to copy*, silently wrapping it in a new node, discarding its original vertex/index data, and resetting `parentInstances` to 0. The in-file comment traces a real historical bug from this: appending an already-instanced Triangle/Quad to a parent Arrow node silently lost their instance count. If you ever refactor this class, that constraint is load-bearing, not incidental.

### Instance propagation (`BuildInstances`)

- `parentInstances == 0` (root): `preMulparentInstanceMatricies[j] = instanceMatricies[j] * modelMatrix` for each of this node's own `instances`, then recurse into children passing this node's flattened array + count.
- `parentInstances != 0` (has a parent): nested loop over `parentInstances × instances`, `preMulparentInstanceMatricies[i*instances+j] = parentInstanceMatricies[i] * instanceMatricies[j] * modelMatrix` — this is the manual-array equivalent of `GeoNode`'s unsqueeze→broadcast-matmul→flatten, just written out as explicit loops instead of a tensor op.
- `AddNodes(...)` reserves vector capacity up front specifically to guarantee at most one reallocation-triggered copy per call (documented inline) — another symptom of children being stored by value.

### Model loading

`GeometryNode<T>::BuildGeoNodeFromModel(path)` (line 411) loads an `.obj`/model file via ModelIO (`MDLAsset`/`MDLMesh`/`MTKMesh`), builds one child `GeometryNode` per submesh with `newBufferWithBytesNoCopy:...deallocator:nil` (buffers here are **not** auto-freed — the copied `malloc`'d vertex/index blocks are intentionally leaked per the `deallocator:nil`, unlike `Shape::buildBuffers`'s deleting deallocator). This is the direct ancestor of the modern `ModelController` in [SOPs/Controllers.cpp](../SOPs/Controllers.cpp), which does the same ModelIO extraction but writes straight into `matrix` tensors instead of owned raw buffers.

**Where it's drawn**: `Renderer` also keeps `std::vector<GeometryNode<uint16_t>> _NodesQueue` (and a `_32` / `shared_ptr` variant). `drawInMTKView:` calls `_NodesQueue[i].draw(cmdEncoder, metalDevice, predefinedRenderPipelineState, customRenderPipelineStates, &cam, active_state)` (`MatrixH.mm:11530`), which recurses into `childNodes` itself — `Renderer` doesn't flatten the hierarchy the way `Viewer::gather_nodes()` does; each `GeometryNode` walks and draws its own subtree.

## 3. Shared pipeline-state array

Both legacy systems and the modern `Viewer` are handed the *same* `id<MTLRenderPipelineState> predefinedStates[7]` array (built once in `Renderer`'s `-initWithDevice:...`, `MatrixH.mm:11064-11157`). The array is a single historical timeline, not two separate ones:

| Slot | Pipeline state | Used by |
|---|---|---|
| 0 | `nodeRenderPipelineState` (`NodeVertexShader`) | `GeometryNode::draw`, `PredefinedRenderPipelineState::Mesh` |
| 1 | `PointCloudNodeRenderPipelineState` | `GeometryNode::draw`, `PredefinedRenderPipelineState::PointCloud` |
| 2 | `BillboardNodeRenderPipelineState` | `GeometryNode::draw`, `PredefinedRenderPipelineState::Billboard` |
| 3 | `DAGRenderPipelineState` (`DAGNodeVertexShader`) | `Viewer`, `material.pipeline_state == 3` (SOA Mesh) |
| 4 | `DAGLineRenderPipelineState` (`vertex_line3d`) | `Viewer`, `pipeline_state == 4` (SOA Thick Line) |
| 5 | `DAGPointCloudRenderPipelineState` (`vertex_dag_pointcloud`) | `Viewer`, `pipeline_state == 5` (SOA Point Cloud) |
| 6 | `DAGEdgeRenderPipelineState` (`vertex_edge3d`) | `Viewer`, `pipeline_state == 6` (SOA Edge Topology) |

`Shape<T>`'s `Renderer` draw path uses yet other dedicated pipeline states (`BasicRenderPipelineState`, `instanceRenderPipelineState`, `PointCloudRenderPipelineState`, `LightingRenderPipelineState` — not part of the `predefinedStates[7]` array at all), reflecting that it predates even the `GeometryNode` convention of sharing one indexed array.

## 4. Do not copy — checklist for anyone touching this code

If you're fixing a bug in `Shape`/`GeometryNode`/legacy `Renderer` paths, it's fine to patch them in place. But when adding *new* rendering functionality, none of the following patterns from these files should be replicated in `GeoNode`/`Viewer`/`Controllers` code:

- Owning raw `new[]` vertex/index arrays instead of `matrix` handles.
- Storing children `by value` in a `std::vector<Node>` instead of `std::vector<shared_ptr<Node>>`.
- Eagerly recomputing an entire subtree's transforms on every mutation instead of relying on dirty-tracking.
- A big hand-written `if (RenderStateNo == ...)` dispatch instead of an indexed `pipeline_state` lookup.
- Per-shape/per-node hit-testing loops that don't go through the shared `Viewer::ray_hit_closest`.

For the correct modern equivalents of all of the above, see [Rendering_3D.md](Rendering_3D.md).
