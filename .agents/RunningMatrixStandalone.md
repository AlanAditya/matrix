# Running/Debugging `matrix` Outside Xcode

`matrix.h`/`Matrix.mm` don't need the full WorldOf3D Xcode project to build and run — they compile standalone via CMake (Objective-C++, `-fmodules` for `@import`). This is faster than opening Xcode for isolated repro/debugging of matrix bugs. See [[MatrixAPI]] for what's on the class, [[AddingNewOps]] before adding a new op, [[MemoryManagement]] for refcounting semantics.

## The exact file set needed to build

Only these files are required — not the rest of the project:

- `matrix.h`, `Matrix.mm`, `primitives.cpp` (primitives.cpp is `#include`d by `Matrix.mm`, not compiled separately)
- `Mods/GPUManager.h`, `Mods/Utils.h`, `Mods/Utils.mm` — `matrix.h` does `@import GPUManager;`, which does `@import Utils;`. **`Mods/Utils.mm` is easy to miss**: it defines `collapse_dims`/`collapse_dims_matmul`/`collapse_dims_reduce` (declared in `Utils.h`) but is never `#include`d anywhere — Xcode just compiles every `.mm` in the target. Leaving it out compiles fine and fails at *link* time.
- `module.modulemap` — wires up the `@import GPUManager;` / `@import Utils;` clang modules
- `ComputeShaders/*.metal` — loaded at runtime by `GPUManager` (`NSBundle mainBundle pathsForResourcesOfType:@"metal"`), not compiled in; needed for GPU-path ops to actually dispatch, not needed to link

Requires C++23 (the `a[3, R(), R(2,4)]` multi-arg `operator[]` slicing syntax needs it) and macOS/Apple Silicon (ARM NEON intrinsics, Metal).

## Two standing setups on this machine

**1. Standalone published mirror** — `https://github.com/AlanAditya/matrix.git`. A history-filtered, one-way mirror of just the file set above, kept in sync by `/Users/adityadude/Documents/C++XCODE/matrix-sync/sync-matrix.sh` (re-derives history from WorldOf3D and force-pushes — only picks up **committed** changes). Has a working `CMakeLists.txt` already, for consuming as a library from other CMake/CLion projects via `add_subdirectory`.

**2. Local debug sandbox** — `/Users/adityadude/Documents/C++XCODE/matrix-sync/debug/`. For debugging against the **live working tree**, uncommitted edits included:

```bash
/Users/adityadude/Documents/C++XCODE/matrix-sync/refresh-debug.sh   # copies current files from WorldOf3D, incl. uncommitted edits
cd /Users/adityadude/Documents/C++XCODE/matrix-sync/debug
cmake -S . -B build && cmake --build build
./build/scratch    # runs debug/tests/scratch.mm, if present
```

Put a repro in `debug/tests/scratch.mm` (`#include "matrix.h"`, needs a `main()`), then refresh+build+run. The scratch target only exists in the CMake config if that file is present — it's appended by `refresh-debug.sh`, not part of the published repo's `CMakeLists.txt`.

Both scripts only ever read from WorldOf3D via copies/clones — neither writes back to it.

### GPU-path ops (compute shaders) work standalone too, not just CPU

In the real app, `GPUManager` loads compute shaders via `[metalDevice newDefaultLibrary]` (`Mods/GPUManager.h`), which only works once Xcode has compiled `ComputeShaders/*.metal` into a metallib embedded in the app bundle — a bare CMake CLI binary has none, so any GPU-path op used to crash with `computeFunction must not be nil`.

`refresh-debug.sh` and the tooling `CMakeLists.txt` now handle this automatically, every refresh:

- `CMakeLists.txt` adds a `matrix_metallib` target that compiles every `ComputeShaders/*.metal` via the real `xcrun metal`/`xcrun metallib` toolchain into `ComputeShaders/default.metallib`, and makes the `matrix` library depend on it — so it's rebuilt whenever a shader changes, with no manual `xcrun` invocations needed.
- `refresh-debug.sh` patches the **copied** `Mods/GPUManager.h` (never the real source in WorldOf3D) to load that `default.metallib` by absolute path instead of calling `newDefaultLibrary`. It re-patches from a fresh copy every run, so it can't double-patch or drift; if `GPUManager.h`'s structure changes upstream (the `class GPUManager {` marker or the `newDefaultLibrary` line), the script fails loudly with a clear message rather than silently skipping the patch.

Net effect: `eval_metal()`, `jit_graph_gpu`, and every broadcasted/Take/etc. GPU kernel work in `debug/tests/scratch.mm` exactly as they would on-device — this is how the `add_gpu_brodcasted` collapsed-dims bug and the `resample()`/`jit_graph_gpu` dtype-mismatch bug were both actually reproduced and fixed, not just reasoned about.
