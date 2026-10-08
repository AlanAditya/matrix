# Running/Debugging `matrix` Outside Xcode

`matrix.h`/`Matrix.mm` don't need the full WorldOf3D Xcode project to build and run — they compile standalone via CMake (Objective-C++, `-fmodules` for `@import`). This is faster than opening Xcode for isolated repro/debugging of matrix bugs, and the CLion project (§3) adds the app's renderer in an OpenCV-style window. See [[MatrixAPI]] for what's on the class, [[AddingNewOps]] before adding a new op, [[MemoryManagement]] for refcounting semantics, [[Rendering_3D]] for the scene graph.

## 1. The exact file set needed to build

Only these files are required — not the rest of the project:

- `matrix.h`, `Matrix.mm`, `primitives.cpp` (primitives.cpp is `#include`d by `Matrix.mm`, not compiled separately)
- `Mods/GPUManager.h`, `Mods/Utils.h`, `Mods/Utils.mm` — `matrix.h` does `@import GPUManager;`, which does `@import Utils;`. **`Mods/Utils.mm` is easy to miss**: it defines `collapse_dims`/`collapse_dims_matmul`/`collapse_dims_reduce` (declared in `Utils.h`) but is never `#include`d anywhere — Xcode just compiles every `.mm` in the target. Leaving it out compiles fine and fails at *link* time.
- `module.modulemap` — wires up the `@import GPUManager;` / `@import Utils;` clang modules
- `ComputeShaders/*.metal` — compiled into `default.metallib` (§2); needed for GPU-path ops to dispatch, not needed to link

Requires C++23 (the `a[3, R(), R(2,4)]` multi-arg `operator[]` slicing syntax needs it) and macOS/Apple Silicon (ARM NEON intrinsics, Metal).

## 2. Three standing setups on this machine

All live in `/Users/adityadude/Documents/C++XCODE/matrix-sync/` and only ever **read** WorldOf3D (copies/clones); none write back to it.

**(a) Published mirror** — `https://github.com/AlanAditya/matrix.git`. A history-filtered, one-way mirror of the §1 file set, kept in sync by `matrix-sync/sync-matrix.sh` (re-derives history from WorldOf3D and force-pushes — only picks up **committed** changes). Has a working `CMakeLists.txt` for consuming as a library via `add_subdirectory`.

**(b) Debug sandbox** — `matrix-sync/debug/`, built from the **live working tree**, uncommitted edits included:

```bash
/Users/adityadude/Documents/C++XCODE/matrix-sync/refresh-debug.sh   # copies current files from WorldOf3D
cd /Users/adityadude/Documents/C++XCODE/matrix-sync/debug
cmake -S . -B build && cmake --build build
./build/scratch    # runs debug/tests/scratch.mm, if present
```

Put a repro in `debug/tests/scratch.mm` (`#include "matrix.h"`, needs a `main()`). The `scratch` target only exists when that file does (appended by `refresh-debug.sh`).

What `refresh-debug.sh` does, every run:
- copies `matrix.h`, `Matrix.mm`, `primitives.cpp`, `Mods/{GPUManager.h,Utils.h,Utils.mm}`, `module.modulemap`;
- syncs `ComputeShaders/` with `rsync -a --delete --exclude default.metallib` (in place, so the built metallib survives a refresh);
- patches the **copied** `GPUManager.h` to load `debug/ComputeShaders/default.metallib` by absolute path instead of `newDefaultLibrary` (a bare CLI binary has no app bundle). It re-patches from a fresh copy every run and fails loudly if the `class GPUManager {` marker or the `newDefaultLibrary` line changes upstream;
- writes `debug/CMakeLists.txt` only when its content changed (rewriting it every time made CMake reconfigure on every build, which re-ran the script).

`debug/CMakeLists.txt` (from `matrix-sync/CMakeLists.txt`) builds `libmatrix` plus a `matrix_metallib` target: every `ComputeShaders/*.metal` → `xcrun metal` → `.air` → `xcrun metallib` → `default.metallib`. So `eval_metal()`, `jit_graph_gpu` and every GPU kernel work standalone exactly as on-device (that's how the `add_gpu_brodcasted` collapsed-dims bug and the `resample()`/`jit_graph_gpu` dtype bug were reproduced and fixed).

**(c) CLion project** — `matrix-sync/cpp/`, §3.

## 3. The CLion project (`matrix-sync/cpp/`)

Open the folder in CLion. Its `CMakeLists.txt`:
- runs `refresh-debug.sh` on every CMake configure (`-DMATRIX_REFRESH_ON_CONFIGURE=OFF` to skip), then `add_subdirectory(../debug matrix EXCLUDE_FROM_ALL)` for `libmatrix` + `default.metallib`;
- syncs the renderer's sources into `debug/` **in place** (`rsync -a --delete` for `SOPs/`, `cmake -E copy_if_different` for `Mods/Camera3D.h`, `cpu_rasteriser.hpp`, and `Shaders.metal`/`DAGShaders.metal` → `debug/RenderShaders/`). The SOPs' relative includes (`./../matrix.h`) need them next to `matrix.h`;
- builds `render.metallib` (in `build/`) from the two render shaders, and a `renderer` static library (`renderer/renderer.{h,mm}`);
- Release flags `-O3 -DNDEBUG -mcpu=native -fno-math-errno` + LTO, applied to `libmatrix` too. **CLion's default profile is Debug**: add a Release profile for the fast flags;
- builds `libmatrix`, the renderer and every executable with `-fobjc-arc`, to match the app.

Targets: `main` (demo: graph + Lorenz swarm + heatmap), `easings` (every `Ease` plotted and animated), `bounce` (a ball driven by `Curve`s), `comp` (a `Comp` with lights, shadows, PBR, fog), plus your own experiments (e.g. `random_exp1`). To add one, copy a block:

```cmake
add_executable(my_exp my_exp.cpp)
set_source_files_properties(my_exp.cpp PROPERTIES LANGUAGE OBJCXX)   # matrix.h is Objective-C++ (@import, Metal)
target_compile_options(my_exp PRIVATE -fobjc-arc)
target_link_libraries(my_exp PRIVATE renderer)                       # renderer pulls in matrix
```

and `#include "renderer.h"` (which includes `matrix.h`, `Camera3D.h`, `cpu_rasteriser.hpp` and the SOPs) — plus `"SOPs/Comp.cpp"` / `"SOPs/Keys_and_Curves.cpp"` for animation.

**Workflow after editing WorldOf3D sources** (`matrix.h`, `SOPs/`, `Shaders.metal`, ...): the copies are made at **configure** time, so **Tools → CMake → Reload CMake Project**, then **Build → Build All**, then run. Rebuilding without a reload compiles the *old* copies (that's how a fixed shadow bug kept showing up). Build All also guarantees `default.metallib` exists — if a GPU op dies with `computeFunction must not be nil`, build all, and look for iCloud duplicates (§6).

CLion builds into its own `cmake-build-debug/`; a terminal build into `build/` doesn't update CLion's binaries and vice versa.

## 4. The renderer window (`renderer/renderer.{h,mm}`)

```cpp
auto r = Renderer::init(1500, 800, "matrix");
auto [graph, scene, heat] = r.subplots<GraphViewer, Viewer, PlotterViewer>(1, 3);   // rows, cols; each tile IS that viewer
LineController wave = graph->plot(x, matrix::sin(x), {1, .35f, .2f, 1});         // any number of curves per graph
PointCloudController cloud = scene->points(pts, colors, 3.0f);
heat->update_plot(values);
while (r.is_open()) { wave.update(x, y2); if (r.poll() != -1) break; }
r.wait();
```

- **Event loop (OpenCV-style, nothing owns `main()`):** the window only draws and takes input inside `r.poll()` (non-blocking, draws at most every 1/120 s), `r.show()`, `r.wait_key(ms)` (like `cv::waitKey`) and `r.wait()` (until closed). They also drain the main dispatch queue, where the controllers' `update()` swaps land. Everything runs on the main thread. `r.save("x.png")` writes the current frame (alpha ignored — the window is opaque).
- **Layout:** `subplots<Ts...>(rows, cols)` replaces the layout; `r.add(viewer, {x, y, w, h})` adds one over a rectangle (fractions, top-left origin). Every tile gets its own `Camera3D`; `GraphViewer`/`PlotterViewer` are framed head-on (`show_grid = false`), others look down on the infinite grid. Background black (`set_background`).
- **Drawing methods live on the viewers in `SOPs/`** (the app has them too): `Viewer::points` / `mesh` / `add(controller)`, `GraphViewer::plot` / `set_bounds`. They return the **controller** that owns the new node: `update()` / `clear()` it, or reach the node through `.node`.
- **Per frame:** `renderer.mm` calls `Viewer::prepare(cmdBuffer, device, shadow_state)` on every visible viewer (shadow pass), then ports `drawInMTKView`'s viewer part: per-tile viewport/scissor/camera, the grid for `show_grid` viewers, `Viewer::draw`, `frame_refs` released when the command buffer completes. Input handlers are ported from `MyMetalView`, routed to the tile under the cursor with that tile's camera.
- **Lighting** (all in `SOPs/GeoNode.cpp`, `SOPs/Viewer.cpp`, `Shaders.metal`; the app gets it too):
  - `Mesh::normals` (`MeshPrimitives` generates exact ones; the cube has 24 vertices, the cylinder/cone split caps and sides), `material.shading`: `Unlit` (default), `Lit` (Blinn-Phong: `specular`, `shininess`), `Standard` (PBR GGX: `metallic`, `roughness`, as three.js `MeshStandardMaterial`; light intensity means "a white Lambert surface shows that brightness", i.e. three.js's / π).
  - `viewer->lights` (up to 8 `Light::directional` / `Light::point`), `viewer->ambient`.
  - Shadows: `Light::casts_shadows` on a directional light (the first one gets a 2048² shadow map fitted to the casters each frame, 3×3 PCF), `material.casts_shadows` / `receives_shadows`.
  - `viewer->tone_mapping` (`None`, `ACES`, `Neutral` — Khronos PBR Neutral's highlight compression **without its dark toe**, which turned dark near-greys blue), `viewer->exposure`, `material.tone_mapped`.
  - `viewer->fog` (`Linear` near/far smoothstep, `Exp2` density; applied last, every shading mode), `material.fog`.
  - Alpha: pipeline 3 blends (`material.colors` alpha = opacity); fully transparent fragments are discarded. Translucent meshes aren't sorted yet (self-overlap banding).
- **Animation** (`SOPs/Keys_and_Curves.cpp`, `SOPs/Comp.cpp`): `Curve<T>` (keys `{time, value, Ease}`, holds before/after) and 13 `Ease`s; `Comp`/`Layer<Controller>` (in/out, local time, `pos`/`rotation`/`scale` as `Curve3`, `opacity`, `parent`, `on_frame`, settled layers skipped), `comp.light(...)` → `LightLayer` (animated lights, `casts_shadows`), `comp.frame(t)` (pure — seek anywhere), `comp.play(window)`. Transforms go in with `transform_of(simd_float4x4)` (fresh refcounted `[1,4,4]` buffer every frame).

## 5. CLion code insight (Nova) — what it can't read, and fixes

CLion here uses the **Nova** engine (ReSharper C++; `clion.use.radler = true`). It parses C++ only — **no Objective-C** — and compiles nothing; the real compiler (AppleClang via CMake) is unaffected by what it flags.

- **`matrix` resolved to a ModelIO property** (red `matrix` everywhere): `SOPs/Controllers.cpp`'s `#import <ModelIO/ModelIO.h>` brings in `MDLTransform.h`'s `@property (nonatomic, assign) matrix_float4x4 matrix;`, which Nova reads as a global variable `matrix`. Fixed by hiding that import from IDE parsers only: `#if !defined(__RESHARPER__) && !defined(__CLION_IDE__)`. Apply the same guard to any Objective-C-heavy header that starts shadowing names.
- **"Ambiguous symbol" with a `__resharper_unknown_type` candidate:** Nova invented a declaration of that name while recovering from code it couldn't parse. Cmd+click the symbol, pick the bogus candidate, and it jumps to the line it misread. Seen so far:
  - a stray `uint32_t` on its own line before `auto r = ...` — a real compile error in that target (`uint32_t auto r`); fix the code.
  - **`uint32_t` everywhere, from `Foundation/NSLocale.h:226`**: `+ (uint32_t)windowsLocaleCodeFromLocaleIdentifier:...` is an Objective-C class method; Nova reads `+ (uint32_t)` as a global variable named `uint32_t` of unknown type. It can't simply be hidden (it comes through the `Foundation.h` umbrella that Metal needs), so the CLion project **quarantines** it for Nova only, without touching `matrix.h` or the sync script: `matrix-sync/cpp/ide_prelude.h` is force-included into every target (`add_compile_options(-include .../ide_prelude.h)`), and under `#if defined(__RESHARPER__)` it imports NSLocale.h's own dependencies at global scope, then `#import <Foundation/NSLocale.h>` inside `namespace ide_quarantine { }`. The bogus declaration lands in that namespace, and `#import`'s include-once skips the file when Foundation.h asks for it. Compilers never define `__RESHARPER__`, so the prelude is empty for them (real clang would reject the block anyway: under `-fmodules` an `#import` inside a namespace is an error). Put any further Nova-only workarounds in that prelude, not in WorldOf3D sources. It's the only Foundation/Metal/CoreFoundation header with an Objective-C method returning `(uint32_t)`. If a future SDK adds another, quarantine it the same way.
  - ModelIO's `matrix`, above (that one *was* guardable: a single optional `#import`).
- **Classic engine (clangd) is worse here:** CLion's bundled clangd (clang 22) can't build the `std` module from the Xcode SDK under `-fmodules` ("could not build module 'std'") on any SDK version tried; Xcode's own `clangd` reads the same compile commands with 0 errors. So stay on Nova.
- `.mm`/`renderer.mm` and `@import` lines are opaque to Nova; that's expected.

## 6. Testing and pitfalls

- **Metal validation:** plain CMake runs don't enable it; Xcode's scheme does. Run with `MTL_DEBUG_LAYER=1 ./build/<target>` before trusting a renderer change (a zero-length Metal buffer passed every plain run and asserted in the app).
- **Snapshots:** `comp.frame(t)` + `r.show()` (a few times, to drain queued updates) + `r.save(path)` checks exact moments; measure pixels rather than eyeballing subtle changes. Opacity, shadows and tone mapping were all verified this way.
- **Don't edit files CLion has open from scripts** for temporary tests: copy to a throwaway `zz_probe.cpp` target instead. Restoring a file with `mv`/`cp` keeps its **old timestamp**, so `make` skips the rebuild and the binary keeps the test code — `touch` restored files, and `strings build/<target> | grep <marker>` to check.
- **iCloud:** `~/Documents` is iCloud-synced. Delete + recreate leaves `name 2.ext` conflict copies (duplicate `.metal`s once broke `metallib` with "multiple symbols"). Tooling must update in place (rsync / copy-if-different); if something odd happens, `find matrix-sync -name "* [0-9].*" -not -path "*/build*"` first.
- **SOPs in several translation units:** `renderer.h` includes the SOPs, so every free function / out-of-class member definition in `SOPs/` must be `inline` (duplicate symbols at link otherwise).
- **Xcode side:** new header-style `.cpp` files in `SOPs/` (e.g. `Comp.cpp`, `Keys_and_Curves.cpp`) must be **unticked from the AdityaIntelligenceProMax target** (the folder is synchronized, so new files join the target and get compiled as plain C++, failing on `matrix.h`'s Objective-C). Include them from `MatrixH.mm` instead. The app's `drawInMTKView` calls `Viewer::prepare` before its main encoder (shadows); hosts that don't still render (a 1×1 dummy shadow map is bound).
- **Invariants learned the hard way:** an empty `matrix` has `buffer == nullptr` (`total_size` always describes the buffer); `frame_refs` must share buffers, so plain owned matrices are made refcounted before being kept; a `local_transform` set from host data has no tape, so in-place edits call `GeoNode::invalidate()` instead of bumping a version.
