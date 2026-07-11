# Memory Management in Matrix and Primitives

The core philosophy of this engine is that **Primitives are passive compute nodes (tapes), while Matrices are active, stack-allocated data handles.** 

In frameworks like MLX or JAX, arrays are heavily abstracted on the heap. Their shared buffer state is stored within the array object itself, which forces a double dereference (double indirection) that is extremely costly for performance. 

To avoid this, this engine makes `matrix` a lightweight, copyable struct. **Instances** of a matrix are literally the same conceptual matrix, just existing in different parts of memory (e.g., as temporary copies passed by value). To keep the `matrix` struct size small and avoid double indirection, the shared state is placed inside the **Primitives** as a trick. The primitive acts as a cache, allowing us to keep a cached buffer for things that are not exclusively computational-graph related (like multi-dimensional views over large buffers).

## 1. Dual Reference Counting System

Because matrices are copyable instances, memory management relies on two distinct reference counters to track instances vs. actual data views.

### `Primitive::primitive_refCount` (Instance Ownership)
- **What it is:** Tracks how many **INSTANCES** of a matrix currently exist. Because matrices are freely copied by value, multiple physical structs in memory represent the same node in the compute graph.
- **Who holds it:** All instances of the matrix that share this single primitive tape.
- **Destruction:** When a matrix instance goes out of scope, it calls `releaseTape()`, decrementing this counter. If it hits `0` (meaning no instances of this matrix exist anymore), the primitive object itself is deleted.

### `matrix::refCount` (Buffer Ownership)
- **What it is:** Tracks how many entities are actively holding a pointer to the physical data `buffer` (and `metalBuffer`). This is distinct from instances, because different matrix instances might share a buffer but have different primitive tapes (e.g. views, transposed slices).
- **Who holds it:** 
  1. Any `matrix` instance that has explicitly acquired the buffer (e.g., via `.transpose()`, `.slice()`, or by calling `update_from_trace()`).
  2. The `Primitive` itself, which holds a `+1` reference if it has cached the buffer.
- **Destruction:** When a matrix instance is destroyed, it decrements this counter. If it hits `0`, the physical buffer is deleted.

## 2. The Evaluation Lifecycle

When an unevaluated matrix instance is forced to evaluate (e.g., by calling `.at<float>()` or `ensure_evaluated()`), the following sequence occurs:

1. **Allocation:** The Primitive allocates the memory buffer, assigns it to the evaluating `matrix` instance, and calls `begin_refcount()` (setting `refCount = 1`).
2. **Caching (The Trick):** The Primitive caches the raw pointers (`out_buffer` and `out_refcount`) inside itself. It then executes `out_refcount->fetch_add(1)`. 
   - *Why +1?* The primitive must act as a co-owner of the data it caches. If the evaluating matrix is a temporary instance (e.g., an internal copy created inside `.min()`) and is destroyed immediately, the primitive's `+1` ensures the cached buffer isn't deleted, allowing other instances of the matrix to use it.
3. **Sharing:** If another instance of this matrix needs the data, it will call `update_from_trace()`. It sees the primitive is already evaluated, grabs the cached `out_buffer`, copies the `out_refcount`, and increments the `refCount` for itself.

## 3. Destruction and Cleanup Rules

Memory safety is guaranteed through strict adherence to C++ RAII (Resource Acquisition Is Initialization), perfectly balancing the instances and their shared data.

### Matrix Destruction
When `matrix::~matrix()` runs (via `destroyInstance()`):
1. It checks if it owns a `refCount`. If so, it does `fetch_sub(1)`. If it is the last owner (`== 1`), it deletes the `buffer` and the `refCount`.
2. It calls `releaseTape()`. If it is the last instance pointing to the primitive (`primitive_refCount == 1`), it deletes the `Primitive`.
3. **Crucial:** The matrix destructor does *not* look at the primitive's state to decide whether to delete the buffer. It relies purely on its own `refCount`.

### Primitive Destruction
Because the Primitive acts as a co-owner of the buffer it caches, it has its own cleanup responsibilities:
1. `~Primitive()` checks its cached `out_refcount`. It does `fetch_sub(1)`. If the primitive is the last entity holding the memory (`== 1`), the primitive deletes the `buffer` and `refCount`.
2. The primitive is also responsible for deleting any internal heap allocations it created during evaluation, such as `BroadcastDescriptor` pointers.

## 4. Multi-Compiled Primitives
For complex nodes like `MultiInputCompilePrimitive`, which output multiple matrices, the node does not store a single `out_buffer`. Instead, it creates a "sibling" node for each output. Each output matrix gets a unique sibling as its tape, and that sibling acts as the standard primitive cache for that specific buffer. When the main `MultiInputCompilePrimitive` is destroyed, it clears references to these siblings, and standard `~Primitive()` cleanup safely takes over.
