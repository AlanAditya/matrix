# matrix

A lazy-evaluated, GPU-accelerated matrix/tensor library for Objective-C++ on Apple platforms, built around a persistent compiled computational graph (closer to TensorFlow 1.x sessions than eager frameworks like MLX/JAX, though inspired by both).

This is a standalone mirror of the `matrix` subsystem extracted from a larger project ([WorldOf3D](https://github.com/AlanAditya/WorldOf3D)), tracking just the files this library needs to build and run on their own.

## Layout

| File | Role |
|---|---|
| `matrix.h` | Core `matrix` class — a node-handle in the computational graph, refcounted buffer, CPU + Metal storage, dtype/shape/broadcast metadata |
| `Matrix.mm` | `matrix` class implementation |
| `primitives.cpp` | Graph node ops (`Primitive` subclasses): CPU/Metal eval, trace building/execution, forward/reverse-mode autodiff (`jvp`/`vjp`) |
| `Mods/GPUManager.h` | Metal device/command-queue/pipeline-state management — compiles and dispatches the compute kernels |
| `Mods/Utils.h` / `Mods/Utils.mm` | Shared utility types + the `collapse_dims`/`collapse_dims_matmul`/`collapse_dims_reduce` helpers `matrix`'s broadcasting, matmul and reduction ops are built on |
| `module.modulemap` | Clang module map wiring up `@import GPUManager;` / `@import Utils;` |
| `ComputeShaders/` | Metal compute kernels (elementwise ops, broadcasting, reductions, matmul, convolution, RNG, etc.) — loaded into the app bundle at runtime by `GPUManager` |

## Building

A `CMakeLists.txt` is included, producing a static `matrix` target (Objective-C++, requires C++23 and Apple Clang's `-fmodules` for the `@import` module syntax `matrix.h`/`GPUManager.h` use). It links `Metal`, `Foundation`, `CoreGraphics`, `ImageIO`. macOS only (Apple Silicon; uses ARM NEON intrinsics).

```bash
cmake -S . -B build
cmake --build build
```

To consume from another CMake project (e.g. in CLion), `add_subdirectory(path/to/matrix)` and `target_link_libraries(your_target PRIVATE matrix)`.

`ComputeShaders/*.metal` are loaded at runtime from the app/executable bundle, not compiled into the static library — make sure they ship alongside whatever links against `matrix`.

## Design

- A `matrix` is an edge in the graph: it owns or shares a data buffer (CPU + optional Metal buffer) via an atomic refcount, and holds a pointer to the `Primitive` that produced it (`nullptr` for leaf/input nodes).
- A `Primitive` is a node: it stores its input matrices (sharing their underlying buffers), and implements both a CPU and a Metal (GPU) evaluation path, plus trace build/execute for the compiled-graph model and `vjp`/`jvp` for autodiff.
- The graph is built once, compiled once, and can be executed many times with minimal per-run overhead.

## Syncing

This repo is kept in sync with the source project via a one-way mirror script (`git filter-repo` + force-push), re-derived fresh from `WorldOf3D` each time — so history here reflects only commits that touched these specific files. It is not meant to be edited directly; changes should be made in the source project and re-synced.
