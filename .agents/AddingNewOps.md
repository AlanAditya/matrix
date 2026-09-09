# Architecture Guide: Adding New Operations and Primitives

This document outlines the architecture of the compute engine and the required steps to add new primitives or operations to the compute graph.

## 1. The 3-Phase Architecture for New Operations
Every new mathematical operation added to the engine is structurally split across 3 distinct phases to ensure a clean separation between graph topology, memory management, and hardware execution.

### Phase 1: Frontend (Graph Building & Dimension Collapsing)
- **Role:** The frontend participates in graph building. It always outputs a new matrix node and takes in inputs. **Frontend functions are mathematical graph builders and MUST NOT take execution parameters like `EvalType` or `ExecutionDevice`.**
- **Responsibilities:** Handles shape inference, broadcasting logic, stride calculations, and **Dimension Collapsing**.
- **Dimension Collapsing & Contiguity:** The frontend uses `collapse_dims` and contiguous checks to heavily reduce the operation down to the simplest possible 1D or 2D iteration space. **Crucially, we only need the operating axis to be contiguous in memory, not the entire metadata.** For operations like `cross`, check if just the operating axis is contiguous (`strides[axis] == 1`). If it is, use `collapse_dims_reduce` to handle the remaining dimensions instead of forcing an expensive full-matrix `.astype()` contiguous copy.

### Phase 2: Primitive (Memory Allocation & Backend Invocation)
- **Role:** The primitive is responsible for allocating memory, holding references to input dependencies (building the DAG), implementing auto-differentiation (JVP/VJP), and calling the backend execution function.
- **Types of Primitives:**
  - **Virtual Primitives:** (e.g., Transpose, Slice) Do not allocate memory. They simply manipulate strides and shapes to create views into existing buffers. Since they do no physical compute, **virtual primitives do not have a backend function**.
  - **Real Primitives:** Allocate fresh memory for the output buffer and invoke the backend kernels using the collapsed dimensions.
  - **Hybrid Primitives:** (e.g., Reshape) Act as a virtual primitive if the input memory is perfectly contiguous (sharing the buffer), but fallback to acting as a real primitive (allocating new memory and executing a backend copy) if the input is non-contiguous.

### Phase 3: Backend (Execution)
- **Role:** Pure execution. The backend does **not** participate in graph building or lazy node instantiation. **It is an execution backend and should only take `ExecutionDevice` (if necessary) as input, never `EvalType`.**
- **Responsibilities:** Takes basic operational parameters and executes the compute loop. **Backend functions MUST NOT accept metadata structures (e.g., `CollapsedDims_3`) as function arguments.** Instead, the backend should extract these directly from the graph node by downcasting the tape (e.g., `CrossPrimitive* primit = static_cast<CrossPrimitive*>(result.tape);`).
- **Splitting:** For massive operations, the backend function can be cleanly split into two separate files/functions: one dedicated to CPU scalar/vector execution, and one dedicated to GPU Metal encoding.
- **Thread Group Reduction (TGR):** Reduction backend operations (like `sum`, `max`, `min`) leverage specialized Thread Group Reduce (TGR) kernels. These launch highly optimized thread groups per output element that utilize shared memory and SIMD groups to perform massive parallel accumulation.

---

## 2. The 5-Step Pipeline for New Operations
When actively writing the code for a new operation, follow this 5-step implementation pipeline to integrate it across the 3 phases mentioned above:

1. **Frontend Method (Matrix Interface)**
   Define the high-level method in `matrix.h` and implement it in `Matrix.mm` (e.g., `matrix::add`, `matrix::reshape`, `matrix::conv`). This method handles shape inference, broadcasting logic, and stride calculations.

2. **Dimension Collapsing**
   Before creating any nodes or dispatching execution, use `collapse_dims` and contiguous checks to reduce the operation down to the simplest possible 1D or 2D iteration space. This significantly speeds up execution by maximizing memory contiguity.

3. **Graph Node (Primitive) Creation**
   Instantiate a new subclass of `Primitive` (e.g., `AdditionPrimitive`, `DotPrimitive`) and attach it to the output matrix's `tape`. The primitive holds references to its parent (input) matrices, effectively building the Directed Acyclic Graph (DAG).

4. **JVP / VJP Implementation (Auto-Differentiation)**
   Inside your Primitive, implement `jvp` (Jacobian-Vector Product) for forward-mode autodiff, or `vjp` (Vector-Jacobian Product) for backward-mode autodiff. This defines how gradients flow backwards through your specific operation.
   *Note: We use a modernized JAX-like functional interface. We do NOT perform VJPs using stateful `.backward()` calls or mutating `.grad` tensors. VJPs must purely return a mathematical graph of operations.*

5. **Execution Dispatching**
   Implement `eval_cpu` and `eval_metal` to perform the actual compute. 
   - `eval_cpu` should use the `dispatch_type` templates to handle the type-erased `uint8_t*` buffer.
   - `eval_metal` dispatches the logic to Metal compute pipelines for zero-overhead GPU acceleration.

---

## 2.5 Worked Examples: Legacy vs. Modern Pattern Compliance

The 5-step pipeline above is the target architecture, but the codebase currently has two styles living side by side. **New ops should be written in the Modern style (Sin/Cos), not copied from the Legacy style (Addition/Subtraction/Multiply/Divide)**, even though the legacy ops are correct and shipped. The broadcasted arithmetic ops are on the list to be ported to the modern pattern eventually — low priority since they work, but do not use them as a template.

### Legacy pattern (Addition, Subtraction, Multiply, Divide) — do not imitate

`operator+` (`Matrix.mm`) builds the graph node correctly (promote dtypes, allocate empty result, `new AdditionPrimitive(lhs, rhs)`, compute `BroadcastDescriptor`s via `broadcast_shapes`, `collapse_dims` into `collapsed_dims_3`, attach `result.tape`). So far this matches Phase 1/2.

The violation is in Phase 3. `matrix::add_cpu_brodcasted` / `matrix::add_gpu_brodcasted` are **dual-purpose**: called with `EvalType::EVAL_AUTO` they re-do the exact same graph-building work as `operator+` (allocate a *second* `AdditionPrimitive`, recompute `BroadcastDescriptor`s and `collapse_dims`), and called with any other `EvalType` they become the actual backend executor, pulling `collapsed_dims_3` off `result.tape` and running the compute loop. One function is simultaneously a frontend graph-builder and a backend executor selected by a runtime flag, which is exactly what the Phase-1/Phase-3 split above is meant to prevent — a backend function must not take `EvalType` or duplicate graph-building logic. This is legacy debt from before the phase separation existed; the ops still work correctly and are exercised in production, they're just structurally tangled and are lower priority to unwind since correctness isn't at stake.

### Modern pattern (Sin, Cos, Tan, Sqrt, Exp, ...) — copy this structure for new ops

`matrix::sin(const matrix& input)` (static, `Matrix.mm`) is **pure Phase 1**: promotes dtype, allocates the empty `output` node, runs `collapse_dims` once and caches it on `SinPrimitive::collapsed_dims`, sets `output.tape`, returns. It takes no `EvalType`/`ExecutionDevice` — it cannot execute anything, only build graph.

`SinPrimitive` (`primitives.cpp`) is **pure Phase 2**: `eval_cpu`/`eval_metal` do the standard allocate-or-adopt-`out_buffer` dance, bail on `COMPILE_TRACE`, guard on `evaluated`, then delegate to `input.sin(out, ExecutionDevice::CPU/METAL)`. `vjp` is a one-liner (`{ grad_out * matrix::cos(input) }`) — gradients are just more graph nodes, never raw buffer math.

`matrix::sin(matrix& output, ExecutionDevice exec_device)` (instance method, `Matrix.mm`) is **pure Phase 3**: takes only `ExecutionDevice`, never `EvalType`. It downcasts `output.tape` to `SinPrimitive*` to read the cached `collapsed_dims` (never recomputes it), and branches internally on `exec_device` to either encode a Metal dispatch (lazily compiling the pipeline state via `GlobalGPUManager.initSin_nd(typeCode, kernel_code)` on first use) or run the `dispatch_type` CPU loop. One function, one job, clean separation — this is the shape every new op should take.

### 2.6 The `eval_cpu`/`eval_metal` Body Structure (the "allocate-or-adopt" dance)

Every real primitive's `eval_cpu(matrix& out, EvalType eval_type)` follows the same fixed structure, in this order. Understanding *why* each step exists matters more than copying the shape:

1. **Recurse into each input, then unconditionally re-sync it:**
   ```cpp
   if (input.tape && !input.tape->evaluated) { input.tape->eval_cpu(input, eval_type); }
   input.update_from_trace();
   ```
   The `!evaluated` guard exists purely to stop recomputation when a tape is shared by multiple parents (a diamond dependency / shared subexpression) — the first parent to reach it does the real work and flips `evaluated = true`; every later parent must skip re-running the compute. But the guard also skips the *entire* function call when it's false, including whatever buffer-adoption logic lives inside it (step 2 below). Since `input` here is frequently a distinct `matrix` object per parent (its own copy from `ensure_graph_ready`) rather than the exact instance that did the computing, its own `buffer` field can still be null even though `input.tape->out_buffer` is already valid. `update_from_trace()` is called *unconditionally*, outside the guard, specifically to un-stale this instance regardless of whether the guarded call actually ran. This is not optional boilerplate — skip it and shared-subexpression graphs silently read null/stale buffers.

2. **Adopt-or-allocate the output buffer**, unconditionally, before checking `evaluated`:
   ```cpp
   if (!out.buffer) {
       if (out.tape->out_buffer) {
           // adopt: another matrix instance sharing this tape already allocated
           out.buffer = out.tape->out_buffer;
           out.metalBuffer = out.tape->out_metal_buffer;
           out.refCount = out.tape->out_refcount;
           out.refCount->fetch_add(1);
       } else {
           // allocate fresh, and publish into the tape's cache for siblings to adopt later
           out.buffer = new uint8_t[out.effectiveBufferSize() * dtype_size(out.type)];
           out.begin_refcount();
           out.buildMetalBuffer();
           out.tape->out_buffer = (uint8_t*)out.buffer;
           out.tape->out_metal_buffer = out.metalBuffer;
           out.tape->out_refcount = out.refCount;
           out.tape->out_refcount->fetch_add(1, std::memory_order_relaxed);
       }
   }
   ```
   This block is why `matrix::eval_cpu()`'s own trailing `update_from_trace()` call is a no-op for most primitives when called at the top level (no `!evaluated` guard sits above the root call, so this block always runs) — it only matters at the root for primitive types that skip this block entirely (`LeafPrimitive`, `SwapLeafPrimitive`), or after a `SwapLeafPrimitive` hot-swap leaves a stale pointer behind.

   **Current known issue (planned fix, not yet done):** the "allocate fresh" branch unconditionally calls `out.buildMetalBuffer()`, i.e. every primitive currently creates a Metal buffer even for a pure CPU-only `eval_cpu()` call that will never touch the GPU. This is wasteful and will be changed in a future commit so Metal buffer creation is lazy/only-when-needed rather than mandatory on every primitive.

3. **Bail early for `COMPILE_TRACE`:** `if (eval_type == EvalType::COMPILE_TRACE) { return; }` — compile-only passes want the buffer allocated/wired up (steps 1-2) but must not execute the backend or mark the node `evaluated`.

4. **Guard the actual compute:** `if (evaluated) { return; } else { evaluated = true; }` — this is the second, narrower `evaluated` check (compute-only, not buffer-sync), separate from the caller-side guard in step 1.

5. **Dispatch to the Phase-3 backend** (e.g. `input.sin(out, ExecutionDevice::CPU)`).

### Layer below Phase 3: the Metal kernel itself

For GPU-backed ops, Phase 3 ends by dispatching a compute pipeline state that is lazily compiled and cached in `GPUManager.h` (a `[dtype][dimSpecialization]` grid, e.g. `SinComputeState_nd` / `BrodcastedAddComputeState`), which resolves by name to a templated kernel in a dedicated `ComputeShaders/*.metal` file (e.g. `ComputeShaders/BrodcastedAdd.metal` defines `BrodcastedAddGPU_1Dgg/_2Dgg/_3Dgg/_NDgg` per collapsed-dim count, instantiated per dtype at the bottom via `instantiate_kernel(...)` macros). This layer is identical in shape for both legacy and modern ops — the split only breaks down in the C++ dispatch layer above it, not in Metal.

---

## 3. Zero-Copy Execution and Memory Management
The engine is heavily optimized to avoid deep copies. It decouples Graph Building (Topology) and Memory Allocation from actual Execution. 

### 3.1 The SwapLeafPrimitive Mechanism (Head Chopping)
When JIT-compiling a subgraph (like in `grad_graph_gpu`), we substitute the external input leaves with `SwapLeafPrimitive` nodes. 
During execution (`CompiledNodePrimitive`), the outer graph physically yields its memory buffer to the inner compiled graph (`outer_input.shareBuffer(sample_parameter)`). 
Because changing a root buffer normally breaks all attached view nodes (like Transpose/Slice) that point to it, the `SwapLeafPrimitive` uses an **Observer Pattern** (an `instances` array). It instantly notifies and updates only the specific dependent nodes with the new memory pointer. This turns an $O(N)$ graph traversal into an effectively $O(1)$ targeted pointer patch!

### 3.2 Buffer Propagation (Tail Chopping & The Aliasing Bug)
When bridging the output of an inner compiled graph back to the outer graph, we want to allocate a fresh buffer for the result so that we can reuse the compiled graph multiple times without memory aliasing (e.g., `W_grad_fn(x1) + W_grad_fn(x2)`).

Initially, we tried simply swapping the buffer of the *last node* of the compiled graph. However, if that last node was a "borrowing" view primitive (like a `TransposePrimitive`), it would just overwrite our newly allocated buffer with its parent's buffer, leaving our allocated memory uninitialized (the notorious `0xBEBEBEBE` float bug) and causing cross-invocation aliasing.

**The Fix (Buffer Propagation):** We don't blindly swap the output node's buffer. Instead, `CompiledNodePrimitive` traverses *upwards* through any borrowing primitives (using `virtual matrix* get_borrowed_input()`) until it hits the actual allocating primitive (the last *independent* output node). We inject the new buffer into *that* node. The computation safely writes directly into this isolated memory, the borrowing primitives naturally point to it, and memory aliasing is completely avoided with zero physical copies.

---

## 4. General Architecture Principles

* **Pure Lazy vs Pure Eager (The Dilemma):** 
  The engine blends the best of both worlds. It supports purely lazy evaluation for global optimizations, but avoids the DAG explosion and memory bloat of pure lazy evaluation by utilizing **Secondary Primitives** (Opaque JIT Nodes like `CompiledNodePrimitive`). These cap the graph depth and massively reuse intermediate memory buffers.
* **Primitive Interface Requirements:** 
  All ops must inherit from `Primitive` and implement `eval_cpu`, `eval_metal`, `vjp`, `jvp`, and `clear_trace_checks`. If the primitive borrows memory (like Reshape or Slice), it MUST implement `get_borrowed_input()`.
* **Raw Pointer Speed:** 
  The engine uses raw `uint8_t*` pointers for CPU buffers and raw `MTLBuffer` for GPU, paired with explicit `std::atomic<uint32_t>` refcounts. This avoids the severe indirection penalty of standard shared pointers, guaranteeing C-style execution speeds.
* **Zero-Dimensional Scalars:**
  Scalars are fully supported as zero-dimensional tensors. Their `dims` field is `0`, their shape array is empty `{}`, and their strides array is empty `{}`. When broadcasting, dimension alignment automatically treats `dims == 0` as a scalar. For hardware execution, `collapse_dims` dynamically translates 0D tensors into safe 1D (size=1, stride=1) memory layouts, avoiding boundary crashes and the need for separate scalar dispatches.

---

## 5. GPU Kernel (Metal) Conventions

When writing Metal kernels (especially N-Dimensional ones like Generic Convolutions), follow these strict conventions to prevent silent failures and `EXC_BAD_ACCESS` memory faults:

### 5.1 Struct / Type Alignment (No Vector Types for Logic)
**Never use `int2`, `int3`, `simd_int2`, etc., in standard struct definitions sent via buffers.** Metal aggressively pads and aligns these vector types differently than C++, leading to critical struct misalignment and hard crashes on device. Always map dimensional arrays to raw scalar types (e.g., `constant size_m* in_shape [[buffer(3)]]`). Avoid abstraction abstractions for indexing in favor of flat scalar loops.

### 5.2 N-Dimensional Grid Dispatching
Since Metal limits thread grids (`MTLSize`) to 3 dimensions (`x, y, z`), high-dimensional kernels must creatively flatten dimensions. A common successful convention for operations like N-Dimensional Convolution is:
- `gid.x` = Output Channels (or outermost non-batch dimension)
- `gid.y` = Last Spatial Dimension (the fastest changing index, optimizing memory locality)
- `gid.z` = Batch * Remaining Spatial Dimensions (the rest of the flattened space)

Inside the kernel, `gid.z` is dynamically un-flattened into discrete coordinates using modulo (`%`) and division (`/`) operators based on the `output_shape` buffer.

### 5.3 Dimension Naming and Ordering
Always follow the specific naming sequence when unrolling spatial dimensions in explicit dimensional functions (like 3D Conv):
- 1D: `x` = length
- 2D: `y` = height, `x` = width
- 3D: `z` = depth, `y` = height, `x` = width

### 5.4 Kernel Instantiation and Templating
Metal compute kernels for operations MUST use generic C++ templates (e.g., `template <typename T>`) rather than hardcoded types like `float`.
These templated kernels **must be explicitly instantiated** at the absolute bottom of the `.metal` file using macros (like `INSTANTIATE_FROM_TYPE(src_idx, type)`), mapped accurately to `matrix`'s internal type codes (e.g., `0` for `float`, `1` for `half`). **Do not add any other logic outside of this instantiation.**

---

## 6. Testing and Validation Workflow

We test new primitives, frontend implementations, and backwards autodiff by side-by-side comparing with **MLX** (or PyTorch). 

### 6.1 C++ Test Bed (`MatrixH.mm`)
The main sandbox for testing C++ execution resides in `MatrixH.mm`, specifically inside the function `-(void) computational_graphV2`. 
1. Append your test logic at the bottom of this function (before the closing brace). 
2. Use raw matrices (e.g., `matrix::of<float>({...})`).
3. Compile and execute. The outputs will be printed to the Xcode debugger console.

### 6.2 Python Verification Scripts (`test_mlx_*.py`)
For every major primitive test in C++, create a temporary equivalent Python file (e.g., `test_mlx_convNd.py`) in the project root.
1. Use `mlx.core` (MLX).
2. Instantiate the exact same input matrix geometries, strides, kernel weights, padding, etc.
3. Run the python script to establish the "ground truth".
4. Check that the outputs in Xcode perfectly match the MLX baseline.
5. **Clean up**: Delete the python script immediately after the test is verified so we do not pollute the workspace!
