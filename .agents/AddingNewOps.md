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
