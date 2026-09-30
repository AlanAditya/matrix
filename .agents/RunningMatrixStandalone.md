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

## 3. CLion project with a plain `.cpp` entry point (optimised)

`/Users/adityadude/Documents/C++XCODE/matrix-sync/cpp/` — open this folder in CLion. `main.cpp` is compiled as Objective-C++ (`set_source_files_properties(... LANGUAGE OBJCXX)`) because `matrix.h` uses `@import`/Metal. Its `CMakeLists.txt` runs `refresh-debug.sh` on every CMake configure (turn off with `-DMATRIX_REFRESH_ON_CONFIGURE=OFF`), then `add_subdirectory(../debug ... EXCLUDE_FROM_ALL)` for `libmatrix` + `default.metallib`. Release builds use `-O3 -DNDEBUG -mcpu=native -fno-math-errno` + LTO, applied to `libmatrix` too. CLion's default profile is Debug; add a Release profile to get the fast flags.

### Renderer window (`matrix-sync/cpp/renderer/`)

`renderer.h` / `renderer.mm` put the modern `Viewer`/`GeoNode`/`Controllers` path in an OpenCV-style window. The window is laid out in typed tiles, and each tile is a real `Viewer`:

```cpp
auto r = Renderer::init(1500, 800, "matrix");
auto [graph, scene, heat] = r.subplots<GraphViewer, Viewer, PlotterViewer>(1, 3);   // rows, cols
LineController wave = graph->plot(x, matrix::sin(x), {1, .35f, .2f, 1});   // any number of curves per graph
PointCloudController cloud = scene->points(pts, colors, 3.0f);
heat->update_plot(values);
while (r.is_open()) { wave.update(x, y2); if (r.poll() != -1) break; }
r.wait();
```

- Drawing methods live on the viewers in `SOPs/`, so the app has them too: `Viewer::points` / `mesh` / `add(controller)`, `GraphViewer::plot` / `set_bounds`. They return the controller that owns the new node: call `update()` / `clear()` on it, or reach the node through `.node`. `GraphViewer::plot` fits the graph to the first curve's data unless `auto_fit` is off.
- `subplots<Ts...>(rows, cols)` replaces the window's layout; `r.add(viewer, {x, y, w, h})` adds one over a rectangle (fractions, top-left origin). Every tile gets its own `Camera3D`. `GraphViewer`/`PlotterViewer` are framed head-on and set `show_grid = false`; other viewers look down on the infinite grid. Background is black (`set_background`).
- Nothing owns `main()`: the window only draws and takes input inside `r.poll()` (non-blocking, draws at most every 1/120 s), `r.show()`, `r.wait_key(ms)` (like `cv::waitKey`) and `r.wait()` (until closed). These also drain the main dispatch queue, which is where the controllers' `update()` swaps land. `r.save("x.png")` writes a frame. Everything runs on the main thread.
- `renderer.mm` ports `drawInMTKView` (viewer/viewport/grid part only, pipelines 3-7 plus the infinite grid) and `MyMetalView`'s mouse/scroll/magnify/arrow-key handlers from `MatrixH.mm`, routing each event to the tile under the cursor with that tile's camera and viewport.
- CMake copies `SOPs/`, `Mods/Camera3D.h`, `cpu_rasteriser.hpp`, `Shaders.metal` and `DAGShaders.metal` into `debug/` at configure time, and builds `render.metallib` from the two shader files. `libmatrix` and the renderer are built with `-fobjc-arc` to match the app.
- `renderer.h` includes the SOPs, so they end up in several translation units: every free function or out-of-class member definition in `SOPs/` must be `inline`, or the link fails with duplicate symbols.
