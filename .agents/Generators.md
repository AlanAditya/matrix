# Adding New Matrix Generators

A **Generator** is a static function that creates a new matrix populated with values based on specific mathematical or randomized logic (e.g., `zeros`, `ones`, `gaussian`, `linespace`), or loaded from outside data (e.g., `fromImage`, `pointsFromPLY`).

There are two kinds:

| Kind | When the buffer is filled | Used by |
|---|---|---|
| **Lazy** (the default for shape-driven generators) | On the first `.eval()` / `.eval_cpu()` / `.eval_metal()` | `zeros`, `ones`, `gaussian`, `perlin`, `randint`, `rand`, `randn` |
| **Eager** | Immediately, inside the call | `linespace`, `eye`, `leaf`, `fromImage`, `pointsFromPLY`, instance `zeros()`/`ones()` |

Write a new shape-driven generator as **lazy**. Use eager only when the output shape or data comes from outside (a file, an image) or when the generator is composed from other ops. See [[MatrixAPI]] for the full list, [[MemoryManagement]] for how `buffer`/`refCount` ownership works.

## 1. Lazy Generators: The Pattern

A lazy generator sets up shape metadata only and attaches a `GeneratorPrimitive` (defined in `primitives.cpp`) as its tape. No buffer is allocated and no data is written in the call itself.

- **Declare** it as `static` in `matrix.h`.
- **Define** it out-of-line in `Matrix.mm`. `GeneratorPrimitive` is only visible there, because `Matrix.mm` is the file that `#include`s `primitives.cpp`.

```cpp
matrix matrix::my_generator(std::initializer_list<size_m> shapeI, float param, dtype type) {
    // 1. Shape metadata only - no buffer, no Metal buffer.
    matrix output((uint32_t)shapeI.size(), type);
    memcpy(output.shape(), shapeI.begin(), output.dims * sizeof(size_m));
    output.calcStrides();
    output.total_size = output.accumul(0, output.dims);

    // 2. CPU fill: out.buffer is already allocated when this runs.
    auto cpu_gen = [param](matrix& out) {
        dispatch_type(out.type, out.buffer, [&](auto* data) {
            using T = std::decay_t<decltype(*data)>;
            for (size_t i = 0; i < out.total_size; ++i) data[i] = (T)/* logic using param */;
        });
    };
    // 3. GPU fill: out.metalBuffer is already built when this runs (see section 2).
    auto gpu_gen = [param](matrix& out) { /* encode a compute kernel or a blit */ };

    output.tape = new GeneratorPrimitive(cpu_gen, gpu_gen);
    return output;
}
```

What `GeneratorPrimitive` does for you on the first eval:

- allocates `out.buffer` (`effectiveBufferSize() * dtype_size`) and starts its refcount
- on `eval_metal`, also calls `buildMetalBuffer()`
- runs `cpu_gen` or `gpu_gen` **once**: an `evaluated` flag guards against re-running, and `COMPILE_TRACE` evals only allocate
- `vjp` returns nothing and `jvp` returns zero: a generator has no inputs and is constant

Capture parameters **by value** in the lambdas. They run later, after the generator call has returned.

## 2. The GPU Side

Look at `ones()` and `gaussian()` in `Matrix.mm`, `ComputeShaders/Fill.metal` and `Mods/GPUManager.h` for the full wiring.

1. **Kernel:** write it in `ComputeShaders/` (fills live in `Fill.metal`) as a template, instantiated once per dtype with a macro. Names end in the dtype tag from `GPUManager::kDTypeTag`, in `dtype` enum order: `f32, f16, u8, i32, i16, u32, u16` (e.g. `fill_ones_f32`).
2. **Pipeline cache in `GPUManager.h`:**
   - add `bool MyInit[7]` and `id<MTLComputePipelineState> MyComputeState[7]`
   - reset `MyInit[i] = false` in the constructor loop alongside `FillOnesInit`
   - add `initMy(int i)`, which loads `@"my_kernel_%s"` with `kDTypeTag[i]`
3. **`gpu_gen`:**
   - `getCommandEncoder()`
   - init the pipeline on first use (`if (!MyInit[typeCode]) initMy(typeCode)`), with `typeCode = (int)out.type`
   - bind the output with `setBufferOrBytes(commandEncoder, out, 0)` and scalars with `setBytes`
   - `dispatchThreads(size, min(size, 256))`
4. **Pure byte-pattern fills** don't need a kernel. `zeros()` calls `GlobalGPUManager.endCommandEncoding()`, then does a `blitCommandEncoder` `fillBuffer` on the command buffer.

Don't pass `int2`/`simd_int2`-style vector types inside structs to Metal. Pass flat scalars or pre-padded `simd_uint3`, as `gaussian()` does. See [[AddingNewOps]].

## 3. Key Requirements and Pitfalls

- **Nothing exists before eval.** `output.buffer` is null until the matrix is evaluated. Code that writes into a lazy generator's buffer directly must `eval()` first. `eye()` does `zeros(...)` then `output.eval()` before writing the diagonal. `leaf()` does `eval()`, then `buildMetalBuffer()` if missing, then `releaseTape()` before attaching its `LeafPrimitive`.
- **Type flexibility:** take a `dtype type = dtype::Float` parameter and fill via `dispatch_type`. If only some dtypes make sense, validate up front and throw. `gaussian()` throws `std::invalid_argument` for anything but `Float`/`Float16`.
- **CPU and GPU must produce the same values.** Mirror the CPU loop in the kernel. `gaussian()`'s kernel sums in `float` even for `Float16` output, as the CPU path does.
- **Eager generators** allocate like `withShape()` in `matrix.h`: `new uint8_t[total_size * dtype_size(type)]`, then `buildMetalBuffer()` when `total_size > 10`, then fill `buffer` directly. The instance methods `zeros()`/`ones()` (shape copied from `this`) still work this way.

## 4. Example: `ones`

This is the smallest complete lazy generator, from `Matrix.mm`:

```cpp
matrix matrix::ones(std::initializer_list<size_m> shapeI, dtype type) {
    matrix output((uint32_t)shapeI.size(), type);
    memcpy(output.shape(), shapeI.begin(), output.dims * sizeof(size_m));
    output.calcStrides();
    output.total_size = output.accumul(0, output.dims);

    // Unlike zero, "one" is a different bit pattern per dtype, so there's no single-byte
    // blit fill available.
    auto fill_ones_cpu = [](matrix& out) {
        dispatch_type(out.type, out.buffer, [&](auto* data) {
            std::fill(data, data + out.total_size, static_cast<std::decay_t<decltype(*data)>>(1));
        });
    };
    auto fill_ones_gpu = [](matrix& out) {
        id<MTLComputeCommandEncoder> commandEncoder = GlobalGPUManager.getCommandEncoder();
        int typeCode = (int)out.type;
        if (!GlobalGPUManager.FillOnesInit[typeCode]) {
            GlobalGPUManager.initFillOnes(typeCode);
        }
        [commandEncoder setComputePipelineState:GlobalGPUManager.FillOnesComputeState[typeCode]];
        setBufferOrBytes(commandEncoder, out, 0);
        uint size = (uint)out.total_size;
        [commandEncoder setBytes:&size length:sizeof(uint) atIndex:1];

        auto dispatchSize = MTLSizeMake(size, 1, 1);
        auto threadsPerGroup = MTLSizeMake((size < 256 ? size : 256), 1, 1);
        [commandEncoder dispatchThreads:dispatchSize threadsPerThreadgroup:threadsPerGroup];
    };
    output.tape = new GeneratorPrimitive(fill_ones_cpu, fill_ones_gpu);
    return output;
}
```

For a generator with parameters, a CPU loop that depends on rank, and a two-pass GPU kernel (fill, then normalize by an atomic sum), see `gaussian()` in `Matrix.mm` and `fill_gaussian_*` / `normalize_by_sum_*` in `ComputeShaders/Fill.metal`.

## 5. File-Loading Generator: `pointsFromPLY`

`matrix::pointsFromPLY(path, groups)` is an **eager** generator that loads a point cloud from a `.ply` file. It is eager and does not follow the lazy pattern above: the output shape comes from the file, and it returns one matrix per requested group. Declared in `matrix.h` next to `save_as_npz`; implemented in `Matrix.mm` (search `PLY point-cloud loading`).

```cpp
auto out = matrix::pointsFromPLY(path, {
    {"x", "y", "z"},                     // out[0] -> [N, 3] Float
    {"red", "green", "blue", "alpha"},   // out[1] -> [N, 4] UInt8
});
```

### Contract

- **Groups:** each inner list is one output matrix `[N, group.size()]`. `out[i]` is always group `i`. Columns follow the requested order, so `{"z","x"}` is a free swizzle.
- **Names:** exact PLY header property names (`red`, not `r`). No aliases.
- **Only the requested properties are read.** Everything else is skipped by byte offset.
- **No casting, ever.** A group's dtype is the source type (`float` -> `Float`, `uchar` -> `UInt8`, `short` -> `Int16`, `ushort` -> `UInt16`, `int` -> `Int32`, `uint` -> `UInt32`). Converting is the caller's job via the separate cast op.
- **Throws `std::runtime_error`** when:
  - a group mixes PLY types (e.g. `{"x","y","red"}`)
  - a property is missing (the message lists the available ones)
  - a property is a list
  - a property's type has no dtype: `char`, `int64`/`uint64` (non-standard), `double`
  - a group is empty
  - the file is truncated, not a PLY, or can't be opened
- **Unsupported types are only a problem when requested.** `char`, `int64`, `uint64` and `double` properties you don't ask for are skipped fine. When `Float64`/`Int64` dtypes are added, supporting them is one line each in `ply_dtype_for`.
- **Formats:** `ascii`, `binary_little_endian`, `binary_big_endian`. Elements before `vertex` (e.g. `face` with list properties) are skipped. The vertex count must fit in `size_m` (uint32).

### How the data gets from the file into the matrices

A PLY body is an array of structs, packed with no padding. The struct is whatever the header declares, so there is no fixed layout: a missing property takes zero bytes, and offsets must always come from the parsed header, never be hardcoded.

The file is `mmap`'d read-only. Then:

1. **Byte-view path** (little endian, fixed-size vertex records, which is the common case). Types don't matter at the byte level: a float is just 4 bytes. The vertex block is viewed as `UInt8 [N, stride]`, each output is re-viewed as `UInt8 [N, cols * size]`, and `copyCPUinplace` does strided byte copies:
   - properties adjacent in the file and in request order (xyz, rgba, `f_rest_0..44`) -> **one copy per group**
   - otherwise -> one copy per column
2. **Gather loop fallback** for big endian (bytes must be reversed), ascii (text must be parsed with `strtof`/`strtoll` as the declared type), and vertex records containing lists (no fixed stride).

Benchmark (2M points, page cache warm): xyz + rgba from a 28-byte record takes 12.6 ms (the gather loop took 88.5 ms). xyz from an xyz-only file takes 1.5 ms (46.6 ms before), because the copy collapses to a single `memcpy`.

### Pitfalls learned building it

- **Don't `slice()` a matrix that wraps foreign memory.** `slice()` puts its input on the compute graph via `ensure_graph_ready` (`primitives.cpp`). That starts refcounting the buffer (which would later `delete[]` it) and wraps it in a Metal buffer (which needs a page-aligned pointer). For a non-owning matrix it throws `cannot begin refcount on non-owning matrix`. Build strided views directly instead: set `shape`/`strides`/`buffer` and `NON_OWNERSHIP_FLAG` (plus `NON_CONTIGUOUS_FLAG` if the row stride != row width), with no tape.
- **`copyCPUinplace` casts values, not bytes, when dtypes differ.** To move raw bytes into a `Float` output, view both sides as `UInt8`.
- A matrix with `NON_OWNERSHIP_FLAG` and no `refCount` is never freed by the destructor or by copies, so wrapping mmap'd memory this way is safe.

## 6. Generators With Tracked Inputs

A lazy generator can take **inputs**: 1-element matrices whose values it reads when it runs, instead of constants captured at build time. That's how a UI slider drives `perlin()`'s `scale` without rebuilding the graph (see `UI/MatrixSliders.h`).

- `GeneratorPrimitive(cpu_gen, gpu_gen, inputs)`: the closures take `(matrix& out, std::vector<matrix>& ins)`. The inputs are ordinary graph edges: evaluated first in the DFS, checked in `invalidate_pass` (so `leaf.mark_changed()` re-runs the generator into its **same** cached buffer), included in `clear_trace_checks`/`get_inputs`, and given a zero gradient each in `vjp` (generator parameters aren't differentiated). The old two-closure constructor still works and means "no inputs".
- **Public signature:** take `const ScalarArg&` (in `matrix.h`) for each tweakable parameter. It's implicit from `float`/`double`/`int` (a constant, baked into the closures as before) or from a `matrix` (tracked, pushed into `inputs`). Keep the plain-float overload and make it delegate, as `perlin()` does. A `ScalarArg` overload can't have default arguments (`ScalarArg` is incomplete inside the class), so all its parameters are required.
- **Validate** each tracked input: exactly 1 element, and the dtype the kernel reads. A `Float` matrix bound as `constant int&` is garbage, so `perlin()` throws unless `octaves` is `Int32`.
- **CPU closure:** read the value from `ins[i].buffer` (already evaluated on the CPU).
- **GPU closure:** bind the input with `setBufferOrBytes(encoder, ins[i], index)`. **Don't read it on the host.** An input computed earlier in the same command buffer (e.g. `slider * 2`) only has its value once its kernel has run: the GPU orders that, the host can't see it. A `constant float&` kernel parameter works with both `setBytes` and `setBuffer`, so the kernel doesn't change.

Verified standalone (`perlin`): constant `ScalarArg`s match the float overload bit for bit; changing a tracked `Float`/`Int32` leaf propagates through a downstream graph on CPU and Metal; a CPU pass followed by a Metal pass after a change; a computed input (`slider * 2`) on Metal; reruns reuse the same buffers with a flat buffer count.


## 7. Custom Metal Kernels: `gpu_func` + `host_op`

To put your own kernel in a graph, pass `gpu_func(name, grid)` to `host_op` in place of a lambda:

```cpp
matrix y = matrix::host_op({H, W}, dtype::Float, gpu_func("my_kernel", {W, H, 1}), {x, slider});
```

- **In-pass, no flush.** The kernel is encoded on the thread's compute encoder like a built-in op, so it reads the outputs of earlier nodes in the same pass without a commit. The lambda `host_op` is different: it commits and waits first, because its host code reads the inputs' memory.
- **`gpu_func(name, grid)`** registers the compute state through `GPUManager::create_custom(name)` (it reuses `shaderNameToIndex` / `customComputeShader`, guarded by `customComputeMutex`) and throws if the default library has no `name`, so typos fail when the graph is built. `grid` (`DispatchGrid {x, y, z}`, unset axes 1) is passed to `dispatchThreads` as is, one thread per cell; the threadgroup comes from the pipeline's `threadExecutionWidth` / `maxTotalThreadsPerThreadgroup`.
- **Binding layout.** Each tensor is a block of 4 slots: buffer, shape (`size_m*`), strides (`size_m*`, in elements), dims (`uint`), at `4b .. 4b+3`. The output is block 0 and input `i` is block `i + 1`, so at most **6 inputs** (Metal has 31 buffer slots); more throws. A rank-0 tensor binds shape and strides `{1}`. Buffers go through `setBufferOrBytes`, so views (slices, transposes) bind at their offset and inputs ≤ 4096 bytes are inlined. Declare blocks with `TENSOR_ARG(name, T, b)` from `ComputeShaders/Utils.h` (`const T` for inputs), which also gives `name_shape`, `name_strides`, `name_dims`; index strided data with `elem_to_loc`.
- **Slider parameters** are 1-element inputs (`leaf_scalar`) with their own block; `mark_changed()` re-runs the kernel into the same output buffer.
- **GPU only.** The CPU path throws. `eval()` picks the CPU for outputs of ≤ 10 elements, so evaluate those with `eval_metal()`.
- The kernel's dtype isn't checked: bind matching types. No gradient (`GeneratorPrimitive`'s zero VJP).
- Example kernel: `gpu_func_scale` at the bottom of `ComputeShaders/custom_kernels.metal`. (`custom_kernel1..4` above it use the legacy `create_custom(name, MatrixH, MatrixH, size)` layout, which is unchanged.)

Verified standalone (`matrix-sync/debug/tests/scratch_gpu_func.mm.bak`), also under `MTL_DEBUG_LAYER=1 MTL_SHADER_VALIDATION=1`: values on 2D and rank 4, offset slice and transposed inputs, a chain `sin(gpu_func(x + 1, s))` in one pass, a slider change re-runs into the same buffer with a flat refcount and no change doesn't re-run, and the CPU path / unknown name / zero grid axis / 7 inputs throw.
