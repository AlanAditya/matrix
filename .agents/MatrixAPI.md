# `matrix` API Reference (`matrix.h`)

This is a complete map of every public constructor, static factory, and method exposed by the `matrix` struct and the free functions/types declared alongside it in `matrix.h`. Use it to find what already exists before adding a new op — see [[AddingNewOps]] for the pipeline to follow when something here is missing. For how ownership of `buffer`/`refCount` actually works, see [[MemoryManagement]].

## Core Types

- **`dtype`** — `enum class` of supported element types: `Float`, `Float16`, `UInt8`, `Int32`, `Int16`, `UInt32`, `UInt16`.
- **`type_from_dtype<code>` / `dtype_from_type<T>()`** — compile-time mapping between a `dtype` enum value and its native C++ type.
- **`dtype_size(dtype)`** — byte size of one element of a given `dtype`.
- **`type_rules[7][7]` / `promote_types(a, b)`** — the type-promotion table used when combining two matrices of different dtypes (e.g. `Float + UInt8 -> Float`).
- **`array_descriptor`** — small-buffer-optimized union storing shape+strides inline (`inline_buffer`, for `dims <= SBO_MAX_DIMS == 3`) or a pointer to a heap-allocated `SharedArrayDescriptor` (for `dims > 3`). Access via `matrix::shape()` / `matrix::strides()`, never directly.
- **`SharedArrayDescriptor`** — refcounted heap block holding shape+strides for high-rank matrices. `create(dims)`, `retain()`, `release()`.
- **`BroadcastDescriptor`** — heap block used internally by `broadcast_shapes`/`broadcast_shapes_matmul` to describe a broadcasted view's shape+strides without mutating the source matrix.
- **`data` struct** — a small RAII helper wrapping a raw buffer + Metal buffer + refcount, used by GPU allocation paths (`data::allocate`, `data::just_allocate`).
- **`dispatch_type(dtype, buffer, lambda)`** — the standard way to get a correctly-typed pointer out of a `void* buffer`; switches on `dtype` and calls `lambda(typed_ptr)`. Used everywhere instead of hand-written switch statements.

## Constructing Matrices

| Signature | Purpose |
|---|---|
| `matrix()` | Default: rank-1, size-0, `Float`. |
| `matrix(uint32_t rank, dtype type)` | Allocates shape/strides storage for `rank` dims; no data buffer yet. |
| `matrix(uint32_t rank, size_t total_size, dtype type)` | Same, and records `total_size` up front. |
| `matrix({1,2,3})` (initializer_list ctor, 1–4 levels of nesting) | Builds a matrix directly from nested `{}` literals, inferring shape/dtype from the literal (up to 4D via the 4 explicit templated constructors + `setup_from_list`). |
| `matrix(const matrix&)` / `matrix(matrix&&)` | Copy/move constructors — copy is a *view* (shares buffer/refcount per the refcounting rules in [[MemoryManagement]]), move transfers ownership. |
| `~matrix()` | Destructor; delegates to `destroyInstance()`. |

### Static Factories

| Function | Purpose |
|---|---|
| `matrix::of<Type>({...})` (explicit template arg, 1D and 2D overloads) | Literal-based construction where you pin the element type explicitly instead of letting the compiler deduce it. Needed because a literal like `{255, 0, 0}` is ambiguous — it could be meant as `UInt8`, `Int16`, `Int32`, etc — and the bare `matrix{...}` constructor would deduce `int` (`Int32`) by default. Use `matrix::of<uint8_t>({255, 0, 0})` to force the intended dtype. |
| `matrix::scalar(T val, dtype type = ...)` | Rank-0 matrix holding a single value. |
| `matrix::withShape({shape...}, dtype)` | Allocates a matrix of the given shape with **uninitialized** data. |
| `matrix::zeros({shape...}, dtype)` (both `initializer_list` and `std::vector` overloads) | Zero-filled matrix. |
| `matrix::ones({shape...}, dtype)` | One-filled matrix. |
| `matrix::gaussian({shape...}, std_dev=1.0, normalize=true)` | 1D/2D/3D Gaussian kernel (`Float` only). See [[Generators]] for the generator pattern this follows. |
| `matrix::randint(low, high, {shape...}, dtype=Int32)` | Uniform random integers; runs on GPU compute shader when `total_size > 10`, else CPU loop. |
| `matrix::rand({shape...}, dtype=Float)` | Uniform random floats in `[0, 1)`; GPU/CPU split same as `randint`. |
| `matrix::randn({shape...}, dtype=Float)` | Standard-normal random floats (Box–Muller); GPU/CPU split same as `randint`. |
| `matrix::eye(m, n, k, dtype=Float)` / `matrix::eye(m, dtype=Float)` | Identity / diagonal-offset identity matrix. |
| `matrix::leaf({shape...}, dtype=Float)` (both `initializer_list` and `std::vector` overloads) | Creates an unevaluated "leaf" graph node (no data yet, but a valid tape placeholder) — see `make_leaf()`. |
| `matrix::linespace(a, b, n)` (matrix endpoints) / `matrix::linespace<T>(start, end, n, dtype=...)` (scalar endpoints) | Evenly spaced values between two endpoints. |
| `matrix::repeating({shape...}, pattern)` | Tiles `pattern` to fill a larger shape (CPU `PatternFill`). |
| `matrix::repeatingGPU({shape...}, pattern)` | Same but builds a zero-stride broadcast view of `pattern` and materializes it via `copyGPUinplace` on the GPU. |
| `matrix::fromImage(path, meta_out=nullptr)` | Loads an image (HEIC/PNG/etc, or `UIImage` on iOS) into a `[H, W, 4]` matrix (`Float` if the source is float-component, else `UInt8`). |
| `matrix::stack(mats, axis)` | Static overload returning a new matrix; the `void` overload writes into a caller-provided `output` matrix instead. |
| `matrix::concat(mats, axis=0)` | Static overload returning a new matrix; the `void` overload writes into a caller-provided `output`. |
| `matrix::meshgrid(x, y, sparse=false)` / `meshgrid(x, y, z, sparse=false)` | Coordinate grids from 1D axis matrices (2D and 3D overloads). |
| `matrix::cross(a, b, axis=-1)` | Cross product (static free-function style entry point; see also instance `cross_impl`). |
| `matrix::sin/cos/tan/sqrt/exp/abs/log(input)` | Static elementwise math ops, each also has an instance overload writing into an explicit `output` (see Elementwise Math below). |
| `matrix::max(a, b)` / `matrix::min(a, b)` | Static elementwise max/min between two matrices (broadcastable). |
| `matrix::conv / conv1d / conv2d / conv3d(...)` | Static convolution entry points (see Convolution below). |

## Shape / Layout Introspection & Mutation

| Method | Purpose |
|---|---|
| `shape()` / `strides()` (const and non-const) | Pointer to the shape/stride array (inline SBO buffer or heap `SharedArrayDescriptor`, transparently). |
| `calcStrides()` | Recomputes `strides()` from the current `shape()` (row-major contiguous). |
| `accumul(start, end) const` | Product of `shape()[start..end)` — used to compute `total_size` or sub-block sizes. |
| `effectiveBufferSize() const` | Actual number of elements backing the buffer (accounts for broadcasted/non-contiguous views). |
| `detach_shape()` | Copy-on-write for the shared shape descriptor: clones it if another matrix still shares it (`dims > SBO_MAX_DIMS` only). |
| `set_array_desc(...)` (2 overloads) | Replaces `array_desc` (and optionally `dims`) wholesale. |
| `printShape(verbose=true) const` / `printStrides(verbose=true) const` | Debug printing to stdout. |
| `print() const` | Full debug dump of the matrix contents. |

## Reshaping, Views & Indexing

| Method | Purpose |
|---|---|
| `reshape(Args... newShape)` (variadic) / `reshape(array_descriptor, dims)` | Returns a new matrix with a different shape over the same logical data. |
| `reshape_eval(output, reshape_desc, dims, execDev=AUTO)` | Backend that materializes a reshape into an explicit `output`. |
| `unsqueeze(axis)` / `unsqueeze(axes[], num_axes)` / `unsqueeze(insertion_axis, num)` | Inserts one or more size-1 axes. |
| `squeeze(axis)` / `squeeze()` | Removes a specific size-1 axis, or all of them. |
| `flatten(start_dim=0, end_dim=-1) const` | Collapses a contiguous range of axes into one. |
| `transpose({new_axis_order...})` / `transpose(vector<size_m>)` / `transpose(array_descriptor)` | Permutes axes. |
| `T() const` | Shorthand full transpose (reverses all axes). |
| `broadcast_to(target_shape, target_dims) const` / `broadcast_toV2(...)` (2 overloads) / `broadcast_to_impl(...)` | Produces a zero-stride broadcasted view to a larger shape. |
| `broadcast_shapes(...)` / `broadcast_shapes_matmul(...)` (free functions) | Computes the common broadcast shape + `BroadcastDescriptor`s for two operands, with a matmul-specific variant that leaves the last two axes alone. |
| `unbroadcast_shape(target_shape, target_dims) const` / `unbrodcast(output, target)` | Reduces a broadcasted matrix back down to a smaller target shape (used in the backward pass for gradients). |
| `slice(...)` (4 overloads: optional-pair list, `AxisRange` list, single `AxisRange` + axis, low-level descriptor form) | Extracts a sub-view; `R`/`AxisRange` supports NumPy-style `a[3, R(), R(2,4)]` slicing (aliased as `R`/`r` at the top of the file). |
| `slice_assign(...)` (matching 4 overloads) | In-place assignment into a sliced region. |
| `operator[](AxisRange)` / `operator[](AxisRange, AxisRange)` / `operator[](AxisRange, AxisRange, AxisRange)` | Sugar over `slice()` for 1–3 axis indexing. |
| `take(index, axis) const` / `take_backend(index, output, axis, exec_device)` | Gather elements along an axis by an index matrix. |
| `pad(...)` (4 overloads: initializer_list of pairs + value, vector + output + value, single-axis left/right + value, normalized-range + target-shape) | Pads a matrix with a constant value along one or more axes. |
| `at<T>(Args... indices)` | Typed, bounds-checked scalar element access (forces evaluation first via `ensure_evaluated()`). |
| `SIMD_MAT(int i)` | Reinterprets row `i` of the buffer as a `simd_float4x4` (forces `eval()` first). |

## Reductions

| Method | Purpose |
|---|---|
| `sum(axis, keepdims=false) const` / `sum(output, axis, keepdims, exec_device=AUTO)` / `sum(start, end, keepdims=false) const` / `sum() const` | Sum along one axis, into an explicit output, over an axis range, or globally. |
| `sum_legacy(axis, keepdims=false)` / `sum_legacy(output, axis, keepdims, eval_type=AUTO)` / `SumNoRed(output, axis, eval_type=AUTO)` | Older/alternate sum backends kept alongside the current implementation. |
| `mean(axis, keepdims=false) const` / `mean(start, end=-1, keepdims=false) const` / `mean() const` | Mean along an axis, over a range, or globally. |
| `rms(axis, keepdims=false) const` / `rms(start, end=-1, keepdims=false) const` / `rms() const` | Root-mean-square, same shape of overloads as `mean`. |
| `max(axis, keepdims=false) const` / `max(output, axis, keepdims, exec_device=AUTO)` / `max(start, end, keepdims=false) const` / `max() const` | Max along an axis, into an output, over a range, or globally. |
| `min(...)` | Same overload set as `max`, for minimum. |

Note: `matrix::max(a, b)` / `matrix::min(a, b)` (static, two-matrix elementwise) are a *different* overload set from the axis-reduction `max`/`min` above — see the Core factories table.

## Arithmetic & Elementwise Math

| Method | Purpose |
|---|---|
| `operator+`, `operator-`, `operator*`, `operator/` (free functions, `matrix ⊗ matrix`) | Standard elementwise arithmetic, broadcasting-aware. |
| `operator+/-/* //` (templated, `matrix ⊗ scalar` and `scalar ⊗ matrix`) | Convenience overloads that wrap the scalar in `matrix::scalar(...)` and reuse the matrix⊗matrix operators. |
| `add/multiply/subtract/divide(other, result, evalType=AUTO)` | Instance-level entry points backing the operators above; dispatch to the `_cpu`/`_gpu` (and `_brodcasted` variants) below based on shape/device. |
| `add_cpu / multiply_cpu / subtract_cpu / divide_cpu(other, result, evalType=AUTO)` | Non-broadcasting CPU backends. |
| `add_gpu / multiply_gpu / subtract_gpu / divide_gpu(other, result, evalType=AUTO)` | Non-broadcasting GPU backends. |
| `add_cpu_brodcasted / multiply_cpu_brodcasted / subtract_cpu_brodcasted / divide_cpu_brodcasted(other, result, evalType=AUTO)` | Broadcasting CPU backends. |
| `add_gpu_brodcasted / multiply_gpu_brodcasted / subtract_gpu_brodcasted / divide_gpu_brodcasted(other, result, evalType=AUTO)` | Broadcasting GPU backends. |
| `matrix::sin/cos/tan/sqrt/exp/abs/log(input)` (static) + instance `sin/cos/tan/sqrt/exp/abs/log(output, exec_device)` | Elementwise transcendental/algebraic functions. `abs`/`log` currently only expose the static-return form plus the instance-with-output form (no separate static-declared `abs`/`log`-into-output split beyond what's listed). |
| `clamp(min_val, max_val) const` / `clamp(output, min_val, max_val, exec_device) const` | Elementwise clamp to `[min_val, max_val]`. |
| `matrix::max(a, b)` / `max(other, output, exec_device) const` / `matrix::min(a, b)` / `min(other, output, exec_device) const` | Elementwise (broadcastable) max/min between two matrices — distinct from the axis-reduction overloads above. |
| `matrix::cross(a, b, axis=-1)` / `cross_impl(other, result)` / `cross_cpu_brodcasted(...)` / `cross_gpu_brodcasted(...)` | Vector cross product, with broadcasting CPU/GPU backends. |

## Linear Algebra

| Method | Purpose |
|---|---|
| `dot(b, transposeB=false)` | Matrix multiply / batched matmul entry point. |
| `dot_cpu(b_transposed, result)` / `batched_dot_cpu(b_transposed, result)` | CPU backends (single and batched). |
| `dot_gpu(b_transposed, result)` / `batched_dot_gpu(b_transposed, result)` | GPU backends (single and batched). |

## Convolution

| Method | Purpose |
|---|---|
| `matrix::conv(input, kernel, padding, stride, dilation, groups=1)` | Generic N-D convolution entry point taking `std::vector<int>` params. |
| `matrix::conv1d(input, kernel, padding=0, stride=1, dilation=1, groups=1)` | 1D convenience wrapper. |
| `matrix::conv2d(input, kernel, pad_h=0, pad_w=0, stride_h=1, stride_w=1, dilation_h=1, dilation_w=1, groups=1)` | 2D convenience wrapper. |
| `matrix::conv3d(input, kernel, pad_d=0, pad_h=0, pad_w=0, stride_d=1, stride_h=1, stride_w=1, dilation_d=1, dilation_h=1, dilation_w=1, groups=1)` | 3D convenience wrapper. |
| `conv1d_gpu / conv2d_gpu / conv3d_gpu(kernel, output)` / `conv_gpu(kernel, output)` | GPU backends invoked by the static wrappers. |

## Type Conversion

| Method | Purpose |
|---|---|
| `astype(dtype, make_contig=false) const` | Returns a new matrix cast to `dtype` (optionally forcing contiguity). |
| `astype(output, dtype, eval_type=AUTO, exec_device=AUTO) const` | Writes the cast result into an explicit `output`. |

## Graph Evaluation / Autodiff / JIT

Matrices carry an optional `Primitive* tape` describing how they were computed (lazy compute graph). See [[MemoryManagement]] for how `tape`/`refCount` interact.

| Method | Purpose |
|---|---|
| `eval()` / `eval_cpu()` / `eval_metal()` | Forces the compute graph to materialize into `buffer` (device-general, CPU-only, or Metal-only). |
| `ensure_evaluated() const` | Evaluates only if not already evaluated (used by `at<T>()`). |
| `update_from_trace()` | Pulls the cached buffer/refcount from `tape` if this instance hasn't grabbed it yet. |
| `compile_cpu()` / `compile_metal()` | Builds a reusable compiled execution plan for the graph (CPU/Metal) without running it. |
| `execute_cpu()` / `execute_metal()` | Runs a previously compiled plan. |
| `clear_trace_checks()` | Resets internal flags used to detect stale/re-traced graphs (used by the `jit_gpu` family after each call). |
| `make_leaf()` | Marks a matrix as a graph leaf (input placeholder) — pairs with `matrix::leaf(...)`. |
| `insert_break(lambda, exec_device=METAL) const` | Inserts a debug/inspection breakpoint node into the graph, invoking `lambda` when that point evaluates. |
| `matrix::jit_gpu(func, sample)` | Compiles `func` against a `sample` input into a fast-path callable `matrix(matrix&)`. |
| `matrix::grad_gpu(func, sample)` | Same, but returns the gradient function. |
| `matrix::build_grad_graph(output_node, sample_input_node)` | Builds the backward graph for a single output/input pair. |
| `matrix::jit_graph_gpu(func, sample)` / `matrix::grad_graph_gpu(func, sample)` | Graph-based (vs. flat) JIT/grad variants for a single input. |
| `matrix::multi_jit_graph_gpu(func, sample_inputs)` / `matrix::grad_graph_gpu(func, sample_inputs)` (vector overload) / `matrix::build_grad_graph(output_nodes, sample_input_nodes)` (vector overload) | Multi-input/multi-output variants of the above. |

## Buffer & Reference-Count Management

See [[MemoryManagement]] for the full ownership model this section implements.

| Method | Purpose |
|---|---|
| `beginReferenceCounting()` / `begin_refcount()` | Initializes `refCount = 1` for a freshly allocated buffer. |
| `shareBuffer(matrix&) const` | Makes another matrix instance share this one's `buffer`/`metalBuffer`/`refCount` (view semantics). |
| `buildMetalBuffer()` | Wraps the current CPU `buffer` in an `MTLBuffer` (`newBufferWithBytesNoCopy`). |
| `releaseBuffer()` | Decrements/releases the data buffer's refcount. |
| `releaseTape()` | Decrements/releases the `Primitive` tape's instance refcount. |
| `destroyInstance()` | Full teardown called from `~matrix()` (releases buffer, then tape). |
| `matrix::copyGPUinplace(outMat, inMat, offset, exec=EncodeAndExecute)` | Copies `inMat` into `outMat` on the GPU (blit fast path when both are contiguous, else a compute-shader gather/scatter); dispatches to the type-casted variant when dtypes differ. |
| `matrix::copyGPUinplaceTypeCasted(outMat, inMat, offset, exec=EncodeAndExecute)` | GPU copy with an on-the-fly dtype cast. |
| `matrix::copyCPUinplace(outMat, inMat, offset)` | CPU equivalent of `copyGPUinplace`, with small-size fast paths (1/4/8-byte direct stores) and a `memcpy`-per-row path for the contiguous case. |
| `matrix::copyCPUinplaceTypeCasted(outMat, inMat, offset)` | CPU copy with an on-the-fly dtype cast. |

## Assignment & Misc Operators

| Method | Purpose |
|---|---|
| `operator=(matrix&&)` / `operator=(const matrix&)` | Move/copy assignment. |
| `operator=(Type value)` (templated, arithmetic types only) | Assigns a scalar value into (presumably a rank-0/element) matrix. |
| `compare_shapes(a, b)` (free function) | Returns whether two matrices have identical shapes. |

## Rendering / Media Interop

| Method | Purpose |
|---|---|
| `CopyToTexture(texture, exec=EncodeAndExecute)` | Blits the matrix's buffer into an existing `MTLTexture`. |
| `ToMTLTexture(exec=EncodeAndExecute)` | Creates and returns a new `MTLTexture` from the matrix. |
| `save_as_image(path, img_type)` | Writes the matrix out as an image file. |
| `matrix::fromImage(path, meta_out=nullptr)` | See Core factories above — the inverse operation. |

## Friend / Free Helper Functions Declared Here

- `setBufferOrBytes(commandEncoder, tensor, index)` — binds a matrix's buffer (or inlines small scalars via `setBytes`) to a Metal compute encoder argument slot.
- `broadcast_shapes(...)` / `broadcast_shapes_matmul(...)` — see Reshaping section above.
- `dispatch_type(...)` — see Core Types above.
