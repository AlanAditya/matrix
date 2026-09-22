# Memory Architecture: `matrix`, `Primitive` and Metal buffers

This is the ownership model for every byte the engine allocates. It's deliberately manual: no `shared_ptr`, no heap-allocated array object, no double indirection. We give that up to get performance, and in return **every rule below has to be followed exactly**. Nothing enforces them at compile time. A broken rule shows up later as a leak, a crash or silently wrong numbers, often far from the line that caused it.

Read this before touching `primitives.cpp`, `releaseBuffer`/`releaseTape`/`update_cache`/`update_from_trace`/`shareBuffer`, or any constructor, destructor or assignment on `matrix`. See [[AddingNewOps]] for the op pipeline, [[MatrixAPI]] for the public surface, and [[RunningMatrixStandalone]] for how to reproduce memory bugs outside Xcode.

## Philosophy: why it's built this way

The core philosophy of this engine is that **Primitives are passive compute nodes (tapes), while Matrices are active, stack-allocated data handles.**

In frameworks like MLX or JAX, arrays are heavily abstracted on the heap. Their shared buffer state is stored within the array object itself, which forces a double dereference (double indirection) that is extremely costly for performance.

To avoid this, this engine makes `matrix` a lightweight, copyable struct. **Instances** of a matrix are literally the same conceptual matrix, just existing in different parts of memory (e.g., as temporary copies passed by value). To keep the `matrix` struct size small and avoid double indirection, the shared state is placed inside the **Primitives** as a trick. The primitive acts as a cache, allowing us to keep a cached buffer for things that are not exclusively computational-graph related (like multi-dimensional views over large buffers).

A `shared_ptr` would make most of these bugs impossible, but it would bring back exactly the indirection and atomic traffic this design exists to avoid. Memory safety comes instead from strict adherence to C++ RAII: every handle and every cache balances its own holds. The rest of this document is the set of rules that makes that balancing work.

---

## 0. The laws (quick reference)

1. **Two counts, never mixed.** The *instance count* (`Primitive::primitive_refCount`) counts matrix handles that are the same graph node. The *buffer count* (`*refCount`) counts everyone using a buffer. `releaseTape()` drops the first, `releaseBuffer()` drops the second. → §2, §3
2. **`releaseBuffer()` never touches the tape.** Rebinding a matrix to other data is `releaseBuffer()` only. Ending a matrix's life is both. → §3
3. **Never write the tape cache by hand once it's filled.** Direct writes to `out_buffer`/`out_refcount`/`out_metal_buffer` are allowed only when the cache is still empty (the first allocation). After that, always use `update_cache()`. → §3.4
4. **COMPILE does all the setup, EXECUTE never allocates.** Host memory, refcounts and Metal wrappers are all created before the `COMPILE_TRACE` early-return. After it, a primitive only runs its kernel. No second `COMPILE_TRACE` return may skip setup code. → §4
5. **The CPU path never touches Metal.** `eval_cpu` never calls `buildMetalBuffer()`. → §5
6. **Owning nodes build their own output wrapper, during COMPILE.** Every owning primitive's `eval_metal` has `if (!out_metal_buffer) buildMetalBuffer()`, because a graph last run on the CPU has memory but no wrapper. → §6
7. **Views (transient primitives) own nothing, not even a wrapper.** A view borrows the parent's buffer at an offset and inherits the parent's Metal wrapper. It never builds one. → §7
8. **A view's rebind check compares against `parent + offset`**, and its rebind updates the cache through `update_cache()`. → §7.3
9. **Rank > 3 shape descriptors are counted too.** Never overwrite `shared_arr_desc` without releasing it, and every `retain(dims)` needs a `release(dims)`. → §3.3c

---

## 1. The pieces

### `matrix` (a stack handle, copied freely)
| Field | Meaning |
|---|---|
| `buffer` | Host pointer to the data. For a view, it points *into* the parent's allocation (`parent + offset`). |
| `metalBuffer` | `id<MTLBuffer>` wrapper around host memory (no copy). For a view, this is the **parent's** wrapper, which starts at the parent's base. |
| `refCount` | `std::atomic<uint32_t>*`, the buffer count shared by every user of this allocation. `nullptr` = not refcounted. |
| `tape` | `Primitive*`, the graph node this matrix *is*. `nullptr` = plain data (a leaf with no graph). |
| `flags` | `NON_OWNERSHIP_FLAG` (bit 0): this handle must never free `buffer`. `NON_CONTIGUOUS_FLAG` (bit 1): strides aren't packed. |
| `array_desc` | Shape and strides. Inline up to `SBO_MAX_DIMS`; above that, a heap `SharedArrayDescriptor` with **its own** separate refcount (a third, independent count, covering only shape and stride storage). |

### `Primitive` (a heap graph node, shared by all instances of one matrix)
| Field | Meaning |
|---|---|
| `primitive_refCount` | Instance count. Starts at 1. |
| `out_buffer`, `out_refcount`, `out_metal_buffer` | **The tape cache.** The node's evaluated output, kept so every instance (and every re-execution) finds the same memory. The cache is a **user of the buffer**: it holds exactly **+1** on the buffer count while it's filled. |
| `evaluated` | Set when the kernel has run in this pass. `clear_trace_checks()` resets it for a re-run. |
| `version`, `last_visited_pass_id` | Dirty tracking for `invalidate_pass()`. |

The cache lives on the primitive rather than in `matrix` for the reason given in the Philosophy section: the primitive is the one shared object every instance already points to.

---

## 2. The two counts

### Instance count: `Primitive::primitive_refCount`
- **Counts:** how many **instances** of a matrix currently exist. Because matrices are freely copied by value, multiple physical structs in memory represent the same node in the compute graph, and all of them share this one primitive tape.
- **+1:** copy construct / copy assign of a matrix with a tape.
- **Moved, not counted:** move construct / move assign take over the source's hold, and the source's `tape` is set to null.
- **−1:** `releaseTape()`. When it hits 0, the primitive is `delete`d, and `~Primitive()` then gives back the cache's buffer hold.

### Buffer count: `*refCount`
- **Counts:** every entity actively holding a pointer to the physical data `buffer` (and `metalBuffer`). That includes all instances, **plus** things that aren't instances, because matrices with *different* tapes can share one buffer: views (slice/transpose/broadcast/reshape), tape caches, JIT placeholders.
- So **buffer count ≥ instance count**, always.
- **Who holds +1:**
  1. every `matrix` handle whose `buffer` points into the allocation and has `refCount` set
  2. every primitive whose tape cache (`out_refcount`) points at it, including `LeafPrimitive`/`SwapLeafPrimitive`, which take +1 in their constructor
- **Freed when it hits 0** by whoever makes the last `fetch_sub` return 1: `releaseBuffer()`, `~matrix()`, `destroyInstance()`, `~Primitive()` or `update_cache()`. That caller does `delete[] buffer; delete refCount;`.

### The evaluation lifecycle (why the cache holds +1)
When an unevaluated matrix instance is forced to evaluate (e.g., by calling `.at<float>()` or `ensure_evaluated()`), this happens:

1. **Allocation:** the primitive allocates the memory buffer, assigns it to the evaluating `matrix` instance, and calls `begin_refcount()` (`refCount = 1`, the instance's hold).
2. **Caching (the trick):** the primitive caches the raw pointers (`out_buffer`, `out_refcount`, and on Metal `out_metal_buffer`) inside itself and does `out_refcount->fetch_add(1)` (the cache's hold).
   - *Why +1?* The primitive must act as a co-owner of the data it caches. If the evaluating matrix is a temporary instance (e.g., an internal copy created inside `.min()`) and is destroyed immediately, the primitive's `+1` ensures the cached buffer isn't deleted, so other instances of the matrix can still use it.
3. **Sharing:** when another instance of this matrix needs the data, it calls `update_from_trace()`. It sees the primitive's cache, takes `out_buffer`/`out_metal_buffer`, copies `out_refcount`, and increments `refCount` for itself.

**The invariant:** buffer count == number of *current* users. Every `fetch_add` must be paired with exactly one future `fetch_sub`. A holder that adds twice while only ever releasing once is a leak (§9, bug 4). A holder that releases without having added frees memory still in use.

---

## 3. Function contracts

### 3.1 `releaseBuffer()`: "I stop pointing at this data"
Drops **this handle's** buffer hold. If `refCount` is set, it does `fetch_sub` and frees on the last one; otherwise it frees `buffer` if owning, or does nothing if `NON_OWNERSHIP_FLAG`. It then sets `buffer`, `refCount` and `metalBuffer` to null. **It never touches `tape`.**

Every caller uses it to *rebind* the handle while it stays the same node, and then keeps using `tape`:
- `update_from_trace()`: `releaseBuffer()`, then reads `tape->out_buffer`
- view rebind blocks: `out.releaseBuffer()`, then `out.tape->update_cache(...)`
- `shareBuffer()`: plugs new data into a JIT placeholder, which must stay its `SwapLeafPrimitive` node
- JIT builders: `sample.releaseBuffer()` after compiling; `sample` keeps its `SwapLeafPrimitive` as the graph's input slot

If `releaseBuffer()` also released the tape, every one of those would crash or lose the node.

### 3.2 `releaseTape()`: "I stop being this node"
`primitive_refCount.fetch_sub`, deletes the primitive on the last one, sets `tape` to null. Doesn't touch the buffer.

### 3.3 End of life releases **both**
- `~matrix()`: releases the descriptor, drops the buffer hold (`fetch_sub`, or frees an owning un-refcounted buffer), then `releaseTape()`. It does **not** call `destroyInstance()`.
- `destroyInstance()`: same two releases. Its only caller is **move assignment**, which then takes over the source's buffer, `refCount` and tape without incrementing, and sets them to null on the source. (Move assignment sends a non-owning `this` to copy assignment before it gets here.)
- copy assignment (owning target): `releaseBuffer(); releaseTape();`, then takes +1 on the source's tape and buffer.
- **Crucial:** the matrix destructor does *not* look at the primitive's state to decide whether to delete the buffer. It relies purely on its own `refCount`. Only the buffer count decides when memory dies.

### 3.3b Primitive destruction
Because the primitive co-owns the buffer it caches, it has its own cleanup responsibilities:
1. `~Primitive()` gives back the cache's hold: `out_refcount->fetch_sub(1)`, and if it was the last holder it deletes `out_buffer` and the counter. (`~SlicePrimitive` is the exception, §7.5.)
2. A primitive also frees any heap allocation it made itself. For example, the binary ops (Add/Sub/Mul/Div, Max/Min, Dot, Cross) destroy their `BroadcastDescriptor`s (`desc_a`/`desc_b`) in their destructors. Anything a primitive `retain()`s, it must also release (§3.3c).

### 3.3c The descriptor count (`SharedArrayDescriptor`, rank > 3 only)
Above `SBO_MAX_DIMS` (3) dims, shape and strides live in a heap `SharedArrayDescriptor` with its own refcount, independent of both counts in §2. The rules:
- `matrix(rank, type)` (and the other rank constructors) **create** one for rank > 3. The matrix's destructor `release()`s it.
- **Don't overwrite a matrix's `shared_arr_desc` without releasing the old one.** To get a private copy you can write into, reuse it if you're the only holder, otherwise create a fresh one and release the shared one. That's `matrix::detach_shape()`, and `ensure_exclusive_out_desc()` in `Matrix.mm` does the same for broadcast outputs.
- `array_descriptor` is a **union with no destructor**. A primitive that stores one and calls `retain(dims)` must call `release(dims)` in its own destructor with the same `dims` (`BrodcastPrimitive`, `ReshapePrimitive`, `TransposePrimitive`).

### 3.4 `update_cache(buf, metal, rc)`: the only way to change a filled cache
```cpp
if (out_refcount == rc) return;                 // same allocation: the cache already holds its 1
if (out_refcount && out_refcount->fetch_sub(1) == 1) { delete[] out_buffer; delete out_refcount; }
out_buffer = buf; out_metal_buffer = metal; out_refcount = rc;
out_refcount->fetch_add(1);                     // take 1 on the new allocation
```
This keeps the cache at exactly one hold. Two sharp edges:
- Because of the early return, calling it with the *same* `rc` but a different `metal` does **not** update `out_metal_buffer`. The view hand-off (§7.2) writes `out.tape->out_metal_buffer` directly for that reason. That's allowed because the wrapper isn't counted.
- **Never pass `rc == nullptr`** while the cache is filled: it releases the old reference and then calls `fetch_add` on null. The legacy `grad_gpu` crashes exactly like this (§10).

**Direct writes to the cache fields** (`out.tape->out_buffer = ...; out_refcount = ...; fetch_add(1)`) are correct **only in the first-allocation branch**, where the cache is still empty. Anywhere else they add a second hold without releasing the first.

### 3.5 `update_from_trace()`
If `tape->out_buffer != buffer`, it does `releaseBuffer()`, then takes the cache's `buffer`, `metalBuffer` and `refCount` with +1. Only the buffer count changes; the instance count never does. Because it compares only `buffer`, a wrapper built later on a handle survives it as long as the buffer pointer is unchanged.

### 3.6 `shareBuffer(mat)`: copies **this** into **`mat`**
Updates `mat.tape`'s cache with `update_cache(this…)`, then `mat.releaseBuffer()`, then points `mat` at this `buffer`/`metalBuffer` with +1 (or marks `mat` non-owning if this isn't refcounted). The direction matters: `a.shareBuffer(b)` puts **a's** data into **b**.

### 3.7 Copy construction
- **Shallow (a view share):** used when the source has a `refCount`, is `NON_OWNERSHIP`, or has a tape. It shares `buffer`/`metalBuffer`/`refCount` (+1) and the descriptor, and takes +1 on the tape.
- **Deep:** used otherwise. It allocates fresh contiguous memory, **builds a Metal wrapper** and copies the data (GPU copy if `total_size > 10`). It leaves `refCount` null.

---

## 4. Execution modes and the compile/execute rule

`EvalType` (Mods/Utils.h):

| Mode | Entry point | What it does |
|---|---|---|
| `COMPILE_TRACE` | `compile_metal()` / `compile_cpu()` | DFS over the graph. Each node prepares inputs, allocates its output and builds wrappers, then **returns before running anything**. |
| `EXEC_TRACE` | `execute_metal()` / `execute_cpu()` | DFS again. Setup is already in place (every guard is a no-op), so each node only runs its kernel, once per pass (`evaluated`). |
| `EVAL_INSTANTLY` | `eval_metal()` / `eval_cpu()` | Compile and execute in one pass: set up, then run. |
| `EVAL_AUTO` | legacy methods | Kept for old non-graph paths. |

`eval()` / `ensure_evaluated()` / `at()` choose the device by size: `eval_metal()` if `total_size > GPU_EXECUTION_THRESHOLD` (10), else `eval_cpu()`.

**The rule:** everything that allocates or wraps goes **before** the `if (eval_type == EvalType::COMPILE_TRACE) return;` in each primitive. After it there's only `if (evaluated) return; else evaluated = true;` and the kernel call. EXECUTE must never allocate, build a wrapper or grow a refcount.

- **No early `COMPILE_TRACE` return on a sub-path.** An extra `if (COMPILE_TRACE) return;` inside the cache-hit branch skipped the wrapper build during compile and pushed it into the first execute (§9, bug 2). There is exactly one `COMPILE_TRACE` return per `eval_metal`.
- **Re-running** a graph: execute sets `evaluated = true`, so call `clear_trace_checks()` before the next `execute_*` (`CompiledNodePrimitive` does `execute_*` then `clear_trace_checks()`).

---

## 5. Anatomy of a primitive's `eval_metal` (owning node)

```cpp
void eval_metal(matrix& out, EvalType eval_type) override {
    // (1) Inputs, DFS first: recurse into unevaluated inputs, otherwise give an already-evaluated
    //     (e.g. CPU-evaluated) input a wrapper if it's big enough to need one.
    if (a.tape && !a.tape->evaluated) { a.tape->eval_metal(a, eval_type); a.update_from_trace(); }
    else { a.update_from_trace(); ensure_metal_buffer(a); }

    // (2) Output memory: take it from the tape cache, or allocate once and fill the (empty) cache.
    if (!out.buffer) {
        if (out.tape->out_buffer) { /* take cache: buffer, metalBuffer, refCount (+1) */ }
        else { /* new uint8_t[]; begin_refcount(); buildMetalBuffer(); fill cache directly (+1) */ }
    }
    // (3) Output wrapper: the CPU -> Metal rerun guard (§6).
    if (!out_metal_buffer) { out.buildMetalBuffer(); out_metal_buffer = out.metalBuffer; }

    if (eval_type == EvalType::COMPILE_TRACE) return;      // ---- everything above is setup ----
    if (evaluated) return; else evaluated = true;
    a.add_gpu_brodcasted(b, out, eval_type);               // (4) kernel only
}
```
`eval_cpu` has the same shape without anything Metal: no `ensure_metal_buffer`, no `buildMetalBuffer`, no step (3).

---

## 6. Metal wrappers and the CPU → Metal rerun

### 6.1 What a wrapper is
`buildMetalBuffer()` = `newBufferWithBytesNoCopy:buffer length:effectiveBufferSize()*dtype_size options:Shared deallocator:{}`. It wraps existing host memory and **doesn't own it**: the empty deallocator means the memory is still freed only through the buffer count. Building one isn't free, which is why the CPU path never does it (law 5).

### 6.2 Binding: `setBufferOrBytes(encoder, m, index)`
- **Has a wrapper:** `setBuffer:m.metalBuffer offset:(m.buffer - [m.metalBuffer contents])`. The offset is worked out at bind time, which is how a view sharing its parent's wrapper binds at the right place.
- **No wrapper:** `setBytes:m.buffer length:effectiveBufferSize()*dtype_size`, inlining the data into the command. Metal caps this at **4096 bytes**. Past that, Metal aborts (`AGX setBytes:length:atIndex:` → `abort`).
- All compute bindings go through `setBufferOrBytes`. Raw `setBuffer:x.metalBuffer` is not allowed.
- **Blit encoders have no `setBytes`**: the blit copy fast path in `matrix.h` builds any missing wrapper on both sides first.
- **Outputs always get a real wrapper** (step (3)), since a kernel can't write into `setBytes` data.

### 6.3 `ensure_metal_buffer(m)` (inputs)
Builds a wrapper for an input that has data but no wrapper, **only if it's larger than 4096 bytes**. Smaller inputs are left without one on purpose and get inlined. It runs in the `else` branch of step (1), during COMPILE, on the primitive's **own copy** of the input handle. It never writes the input node's tape cache.

### 6.4 The CPU → Metal rerun (why step (3) exists)
Graphs are mathematical, not tied to a device: the same graph can run on the CPU and later on Metal. That happens in ordinary use too, because `eval()`/`at()` pick the CPU for ≤ 10 elements.

1. The CPU run allocates every node's memory and fills the tape caches, **with no wrappers** (law 5).
2. The Metal run is a DFS: each node recurses into its inputs first, so the deepest nodes are prepared first and results flow back up.
3. At each owning node, step (2) is **skipped**, because `out.buffer` is already set from the CPU run, and that's where wrappers normally get built.
4. So in the DFS, **step (3) is the only place left** to give the output a wrapper: `if (!out_metal_buffer) buildMetalBuffer()`, during COMPILE. On EXECUTE it's already set, so it does nothing.

Without step (3), a CPU-then-Metal rerun would reach the kernel with a nil output wrapper. Every owning primitive has it, with a comment pointing here. `MultiInputCompilePrimitive` does the same per output, on each output's own `out_i.tape->out_metal_buffer` (siblings, §8.3). Reshape does it on its owning branch.

---

## 7. Transient (view) primitives

**Views:** `SlicePrimitive`, `TransposePrimitive`, `BrodcastPrimitive` (used by `broadcast_toV2`), and `ReshapePrimitive` when `!REQUIRES_NEW_BUFFER`. They declare it with `get_borrowed_input()` returning `&input`, which is also what `CompiledNodePrimitive` / `MultiInputCompilePrimitive` follow to find the owning node behind a chain of views.

### 7.1 The law: a view owns nothing
- `out.buffer = input.buffer + offset * dtype_size` (offset 0 for transpose/broadcast/reshape)
- `out.refCount = input.refCount` (+1): the view is a user of the parent's allocation
- `out.metalBuffer = input.metalBuffer`: the **parent's** wrapper, starting at the parent's base. Only `buffer` carries the offset; `setBufferOrBytes` applies it at bind time (§6.2).
- **A view never calls `buildMetalBuffer()`** on its output.

### 7.2 Hand-off: inherit, don't build
In the CPU → Metal rerun, a view's step (2) is skipped too, but a view must not build. Instead, after step (2):
```cpp
if (out.metalBuffer != input.metalBuffer) {        // parent just built one earlier in this DFS
    out.metalBuffer = input.metalBuffer;           // take the parent's
    out.tape->out_metal_buffer = out.metalBuffer;  // and cache it (wrapper isn't counted: direct write is fine)
}
```
COMPILE does the hand-off; on EXECUTE the pointers match, so it does nothing. **This matters most for a zero-offset slice**: its rebind check (§7.3) never fires, so without the hand-off it kept a nil wrapper, and its consumer inlined the whole slice with `setBytes` and aborted past 4096 bytes (§9, bug 3).

If the parent is ≤ 4096 bytes and has no wrapper, the view has none either. That's correct: the view's bytes are at most the parent's, so they're inlined.

### 7.3 Rebind: compare against where the view starts, and use `update_cache`
Each view has a rebind block for when the parent's memory actually changed:
```cpp
if (out.buffer != (uint8_t*)input.buffer + offset * dtype_size(out.type)) {   // NOT input.buffer
    out.releaseBuffer();
    out.buffer = (uint8_t*)input.buffer + offset * dtype_size(out.type);
    out.metalBuffer = input.metalBuffer;
    out.refCount = input.refCount; out.refCount->fetch_add(1);
    out.tape->update_cache((uint8_t*)out.buffer, out.metalBuffer, out.refCount); // NOT direct writes
}
```
- Compare with `input.buffer + offset`. Comparing with `input.buffer` fires on **every pass** for any non-zero offset.
- Update the cache with `update_cache()`. Direct writes add a second hold every time the block runs (§9, bug 4).

### 7.4 Reshape is pseudo-transient
`REQUIRES_NEW_BUFFER` (non-contiguous input) → Reshape **owns** a fresh buffer: it allocates, builds its own wrapper with the step (3) guard, and runs `reshape_eval` to pack the data. Otherwise it's a view and follows §7.1–7.3. `get_borrowed_input()` returns `nullptr` on the owning branch.

### 7.5 `~SlicePrimitive`
It releases the cache's hold with a plain `fetch_sub` and sets `out_refcount` to null so `~Primitive` does nothing. The final free happens through the `input` member's own destructor, which also holds the parent's buffer.

---

## 8. Graph construction and JIT

### 8.1 `ensure_graph_ready(m)`: entering the graph
Most primitive constructors wrap their inputs with it. A **tapeless** input with data gets `begin_refcount()` (if not refcounted) and a **Metal wrapper**. Leaves therefore get a wrapper at graph-build time on any backend; this is a known exception to law 5 (§11).

### 8.2 Compiled subgraphs: `jit_graph_gpu` / `grad_graph_gpu` → `CompiledNodePrimitive`
The traced graph's inputs are placeholder leaves with a `SwapLeafPrimitive`. Each call:
1. `outer_input.shareBuffer(sample_parameter)` plugs the real input into the placeholder.
2. It walks `get_borrowed_input()` from the inner output down to the **owning** node and does `out.shareBuffer(*owning_node)`, so the inner graph writes straight into this node's output (a placeholder there is a "head–tail collision").
3. `output_graph.execute_*()`, then `clear_trace_checks()`.

Checked standalone: correct values, and live buffers stay flat across calls.

### 8.3 `multi_jit_graph_gpu` → `MultiInputCompilePrimitive` (siblings)
There's one `MultiInputCompilePrimitive` **per output**, and each one is the tape of its own output. Every one of them holds copies of all `outer_outputs` with tapes pointing to their siblings, plus `siblings` (the full list, **including itself**). Those in-primitive tape pointers are **raw and not counted**, so `~MultiInputCompilePrimitive` sets them to null (in its own copies and in the siblings') before the member matrices are destroyed, and removes itself from the siblings' lists. Running any sibling marks all of them `evaluated`.

### 8.4 Legacy: `jit_gpu` / `grad_gpu`
These are **phased out and scheduled for removal.** Don't model anything on them, and don't fix them. Both are broken (reproduced standalone):
- `jit_gpu`: sets its tape cache to null by hand (+1 per call, so every result's buffer leaks), never sets `total_size` on the result, and has a debug `printf` on each call.
- `grad_gpu`: calls `shareBuffer` in the wrong direction, reaching `update_cache(nullptr, …)` → null `fetch_add` → crash on the first call.

---

## 9. Bugs these rules came from

Bugs 1–4 came from **copy-pasting owning-node code into view primitives, or across paths, without adapting it.** Bugs 5–6 came from breaking the descriptor count's pairing (§3.3c).

1. **The CPU path built Metal wrappers** (every `eval_cpu` output allocation called `buildMetalBuffer()`). That was wasted time on every CPU op. Fixed in `844928d`: wrappers only on the Metal path (law 5, §6.4).
2. **A second `COMPILE_TRACE` return in the cache-hit branch** of Stack/Concat/Padding/AsType/Slice/Broadcast/Transpose. It skipped the step (3) wrapper build during compile, so the first EXECUTE built it. Removed in `844928d` (§4).
3. **Views lost their parent's wrapper on a CPU → Metal rerun.** A zero-offset slice kept a nil wrapper, and its consumer aborted on `setBytes` above 4096 bytes. The other views built a second wrapper of their own. Fixed in `c50f28a` with the inherit hand-off (§7.2).
4. **An offset slice leaked a reference to its parent every pass.** The rebind check compared against `input.buffer` (so it always fired), and the rebind wrote the cache by hand (+1 each time). The count went 7 → 16 over 10 runs, and the parent could never be freed. Transpose had the same by-hand write. Fixed in `c50f28a` (§7.3).
5. **Every 4D+ binary op and matmul leaked a shape descriptor.** `broadcast_shapes` / `broadcast_shapes_matmul` unconditionally `create()`d `out_shape.shared_arr_desc`, overwriting the one the result's constructor had just created. That's one leak per Add/Sub/Mul/Div/Max/Min/Cross/matmul on rank > 3. It's fixed by reusing the existing descriptor via `ensure_exclusive_out_desc()` (§3.3c).
6. **4D+ views leaked their shape descriptor.** Broadcast, Reshape and Transpose `retain()`ed their `array_descriptor` in the constructor and never released it. Fixed with destructors that `release()` it (§3.3c).

Bugs 5 and 6 were verified standalone with a live-descriptor counter: before the fix it grew by 2–4 per iteration of a 4D graph, and after it, it stays flat for transpose, broadcast_toV2, reshape (both branches) and 4D matmul, on CPU and Metal. A control run with only the matmul fix reverted leaked again.

---

## 10. Checklist for writing or changing a primitive

- [ ] Constructor wraps inputs in `ensure_graph_ready()`.
- [ ] `eval_metal` inputs: recurse if unevaluated, else `update_from_trace()` + `ensure_metal_buffer()`.
- [ ] Output setup (allocation, refcount, wrapper) is all **before** the single `COMPILE_TRACE` return, with no early return on a sub-path.
- [ ] **Owning:** allocate once, fill the empty cache directly, and include the `if (!out_metal_buffer)` guard with its comment.
- [ ] **View:** borrow the parent's buffer at the offset, `get_borrowed_input()` returns `&input`, use the inherit hand-off, never `buildMetalBuffer()`, rebind check against `parent + offset`, rebind cache via `update_cache()`.
- [ ] `eval_cpu` never touches Metal.
- [ ] All kernel bindings go through `setBufferOrBytes`.
- [ ] Every `fetch_add` has exactly one matching future `fetch_sub`. No direct cache writes once the cache is filled.
- [ ] The destructor frees every heap allocation the primitive made (`BroadcastDescriptor`s etc.) and `release()`s every descriptor it `retain()`ed.
- [ ] Verify standalone ([[RunningMatrixStandalone]]): CPU vs Metal values, rerun the same graph on Metal after a CPU run, count `buildMetalBuffer` calls during EXECUTE (must be 0), and watch the parent's buffer count across repeated runs (must stay flat).

---

## 11. Known exceptions and open issues

**Wrappers built outside `eval_metal`** (intentional or legacy; none of these run on EXECUTE):
- `ensure_graph_ready()` wraps tapeless leaves at graph construction (§8.1).
- `matrix(rank, total_size, type)` builds a wrapper when `total_size > 10`, and so do the eager *instance* methods `ones()`/`zeros()` (`matrix::ones() const`, not the lazy static generators) and `matrix::leaf()`.
- Deep copy construction and deep copy assignment build a wrapper (§3.7).

**Open issues found by reading the code (not fixed yet):**
- `MultiInputCompilePrimitive::eval_metal` does `sib->evaluated = true` without a null check. `eval_cpu` checks, and the destructor sets entries in the siblings' lists to null, so running a surviving sibling after one is destroyed dereferences null.
- `ArgMax`/`ArgMin` `eval_metal` run the CPU kernel in the middle of a Metal pass without waiting for queued GPU work, so they can read stale input.
- `ConvolvePrimitive::eval_cpu` has no CPU implementation (it's commented out), so a CPU conv returns uninitialized memory.
- `PaddingPrimitive` never evaluates its `value` input and leaves it out of `clear_trace_checks`/`invalidate_pass`.
- Sum/Max/Min reductions, ArgMax/ArgMin, Take, Padding and ExecutionBoundary skip `ensure_graph_ready()`, so a tapeless input isn't refcounted.
- Copy assignment's deep-copy path runs `copyGPUinplace` **and then** `copyCPUinplace` for `total_size > 10`, and leaves `refCount` null.
