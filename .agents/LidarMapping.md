# PS 26053: adaptive variable-resolution 2.5D LiDAR mapping

The project most recent work serves: a DRDO / Smart India Hackathon problem statement (SIH26053). See [[Applets]] for the applets that run it, [[Concurrency]] for the threading, [[Generators]] §6 / [[MatrixAPI]] for `ScalarArg`, `host_op` and `leaf_scalar`.

## 1. The brief

Turn raw LiDAR point clouds into a **variable-resolution 2.5D grid** (an elevation map with semantic layers), in real time:
1. **Terrain analysis**: drivable vs non-drivable.
2. **Object detection**: static obstacles (walls, poles) vs dynamic objects (pedestrians, vehicles).
3. **Adaptive spatial representation**: cell size grows with distance (e.g. 5 cm within 10 m, 50 cm out to 100 m), with no alignment errors or data loss.

Expected: a deep learning model (e.g. PointNet++ or a sparse CNN), the grid engine, a real-time dashboard with colour-coding showing a big memory saving versus a uniform high-resolution 3D map, and metrics (low latency / high FPS, accuracy across distances). The team's deck (`~/Downloads/PS26053_LiDAR_Mapping.pptx`) plans: scan → segmentation (terrain / static / dynamic) → variable-res grid → costmap fusion → Hybrid A* → control. A cell holds min height, max height, class by priority, point count.

**"2.5D" is the output map, not the input.** Model input = one scan as a panoramic range image (a depth map, sensor view). Model output = a class per point. The 2.5D thing = a top-down grid where each ground cell stores heights + class.

## 2. Pipeline and status

```
LiDAR scan (x forward, y left, z up, metres)                 kitti::Sequence (AI/Kitti.h)  or phone stream
  -> range image [1, 64, 2048, 5]                            ai::project (AI/RangeProjection.h)
  -> SalsaNext (MLX, Metal)                                  ai::SalsaNext (AI/SalsaNext.h)
  -> class per point (20 SemanticKITTI classes -> 3 groups)  ai::labels_for_points, kitti::group_of
  -> foveated 2.5D grid (heights + class per cell)           foveated::Grid (AI/FoveatedGrid.h)
  -> tiles for the renderer                                  foveated::tiles -> TriangleMeshController
  -> a box per car: BEV clustering + L-shape fit             bbox::fit_boxes (AI/BoundingBoxes.h) -> WireBoxController
```

| Piece | Status |
|---|---|
| SalsaNext in MLX C++ | done, matches PyTorch |
| KITTI loader, projection, back-projection | done, matches SalsaNext's own code |
| Foveated 2.5D grid | done |
| Replay dashboard (`kitti_replay`) | done (accuracy/timing to the console) |
| Benchmark (`tools/salsanext/benchmark_kitti.cpp`) | done |
| Car bounding boxes (`AI/BoundingBoxes.h`: 0.3 m BEV 8-connected clustering + L-shape closeness fit, <1 ms/scan). On predictions: pixel owners only (`ai::owns_pixel`: points that lost their range-image pixel inherit the front point's label, which made ghost cars behind real ones) + mean car probability >= 0.9 + width >= 0.3 m. Seq 08, every 5th of 500 scans: precision 46% -> 96.5%, GT cars (>= 30 pts) found 96% -> 92% | done |
| Pedestrian boxes (class 6: 0.2 m cells, >= 8 points, no min width; GT labels find every person with >= 10 points, SalsaNext's ~75% precision / ~54% recall: network-limited, 71% of person points labelled person) | done |
| Speed per box (`AI/Tracking.h`: world frame via poses, 1.5 s least-squares fit; parked cars 90% < 1.0 m/s, moving 9.7-10.4 m/s; standing people 90% < 0.35 m/s) | done |
| KNN post-processing (SalsaNext's optional edge clean-up) | not done |
| Moving vs parked (ego-motion from poses + tracking) | not done (`FoveaCluster.h` has a tracker without ego-motion) |
| Grid accumulated over time, free space by ray casting, costmap, Hybrid A* | not done (deck's Phase 3) |
| On-screen numbers in the panel (needs a text label control) | not done |

MLX is linked for **macOS / visionOS only** (`/usr/local/lib/libmlx.dylib`, MLX 0.26.5; the project's link filter excludes iOS). The `AI/` headers that use MLX compile to nothing on iOS (`#if !TARGET_OS_IPHONE`). `Kitti.h` and `FoveatedGrid.h` are plain C++ and work everywhere. Inference therefore runs on the Mac (phone → Mac streaming fits this).

## 3. The model: SalsaNext

Chosen over PointNet++ / sparse CNNs: a range-image CNN (only 2D convs, which MLX runs well), fast, pretrained on SemanticKITTI, MIT licence. The iPhone depth map is also an image, so the same idea transfers.

- **Weights:** `/Users/adityadude/Documents/C++XCODE/pretrained/` holds `SalsaNext` (the original 107 MB PyTorch checkpoint, Google Drive id `1utfzooTDAlV5M6XGvCE0-L-vbdLe_2rD`), `arch_cfg.yaml`, `data_cfg.yaml`, and the converted `salsanext.safetensors` (27 MB) + `salsanext_golden.safetensors`.
- **Conversion:** `tools/salsanext/convert_salsanext.py`, run with **`/opt/miniconda3/bin/python`** (base env has torch 2.9.1 + mlx 0.29.4; Homebrew Python has neither).
  - The checkpoint is a *legacy (non-zip) pickle* that also stores the SGD optimiser and CyclicLR scheduler, so `torch.load(weights_only=True)` refuses it. It was audited without executing (pickletools): only torch / numpy / collections globals plus `getattr(scheduler, "_triangular_scale_fn")`. The script re-checks that allowlist before every load and refuses anything else.
  - Layout: NCHW → NHWC; conv weights `[out, in, kh, kw]` → `[out, kh, kw, in]`; each BatchNorm becomes a per-channel `scale` / `shift` (it can't be folded into the conv before it: SalsaNext applies LeakyReLU in between).
  - Metadata in the safetensors: input size, fov, normalisation means/stds, `learning_map_inv`, class names, class colours (RGB; data_cfg's `color_map` is BGR).
  - Before writing anything it runs a Python MLX port (same ops as the C++) against PyTorch.
- **Architecture** (`AI/SalsaNext.h`): 6.7M params. ResContextBlock ×3 → ResBlock ×5 (avg-pool 3×3/2, pad 1, count_include_pad, done as a depthwise conv with a 1/9 kernel) → UpBlock ×4 (PixelShuffle(2) as reshape + transpose) → 1×1 conv logits → softmax. Order in every block: conv → LeakyReLU(0.01) → BatchNorm. Forward wrapped in `mx::compile`.
- **Input:** 64 × 2048 × 5 (range, x, y, z, remission), HDL-64: `fov_up` 3°, `fov_down` -25°, each channel `(v - mean) / std` with means `12.12, 10.88, 0.23, -1.04, 0.21`, stds `12.32, 11.47, 6.91, 0.86, 0.16`, then **0 where no point**.
- **Output:** 20 classes: 0 unlabelled, 1 car, 2 bicycle, 3 motorcycle, 4 truck, 5 other-vehicle, 6 person, 7 bicyclist, 8 motorcyclist, 9 road, 10 parking, 11 sidewalk, 12 other-ground, 13 building, 14 fence, 15 vegetation, 16 trunk, 17 terrain, 18 pole, 19 traffic-sign.
- **Groups** (`kitti::group_of`): terrain = road, parking, sidewalk, other-ground, terrain; static = building, fence, vegetation, trunk, pole, traffic-sign; dynamic = the vehicles, persons and riders (things that *can* move; whether they *are* moving needs tracking with ego-motion).

## 4. Range projection (`AI/RangeProjection.h`)

Reproduces SalsaNext's `laserscan.py do_range_projection` + its parser's normalisation exactly: `yaw = -atan2(y, x)`, `pitch = asin(z / r)`, `col = floor(0.5 (yaw/π + 1) W)`, `row = floor((1 - (pitch + |fov_down|)/fov) H)`, clamped; closest point wins a pixel; the original's mask is `proj_idx > 0`, so a pixel held by point #0 counts as empty (kept for bit-exactness). Back-projection: each point takes its own pixel's argmax (what SalsaNext does without KNN).

## 5. KITTI data (`AI/Kitti.h`, `tools/salsanext/fetch_kitti_subset.py`)

SemanticKITTI = pre-recorded Velodyne HDL-64 drives (Karlsruhe) + per-point labels; licence CC BY-NC-SA. Sequence 08 is the **validation** split (not trained on). Layout per sequence: `velodyne/NNNNNN.bin` (float32 x, y, z, remission), `labels/NNNNNN.label` (uint32: lower 16 bits class id such as 40 road / 10 car / 252 moving-car, upper 16 instance id), `poses.txt` (3×4, **left camera frame**), `calib.txt` (`Tr`: LiDAR → camera). `Sequence::pose(i)` converts to the LiDAR frame: `Tr⁻¹ · P · Tr`.

**Downloading without the 80 GB zip.** The KITTI (S3) and SemanticKITTI servers support HTTP range requests, and the scans are *stored* (uncompressed) in the zip, so `fetch_kitti_subset.py` reads each zip's central directory (~8 MB) and fetches only the requested members; adjacent members are batched into one request (the 7 KB label files would otherwise cost one slow round-trip each); CRCs are checked; re-running skips finished files (resume after switching Wi-Fi). Sequence 08 = 4,071 scans × ~2 MB = 8 GB; all its labels 32 MB; poses + calib ~0.2 MB. The first 500 scans (~1 GB) are at:

```
/Users/adityadude/Documents/C++XCODE/datasets/SemanticKITTI/dataset/sequences/08
/opt/miniconda3/bin/python tools/salsanext/fetch_kitti_subset.py <dataset dir> 08 <first> <count>
```

## 6. The foveated 2.5D grid (`AI/FoveatedGrid.h`)

Nested **square** rings, cell size doubling outward; every ring edge is a whole number of cells of both rings that meet there (checked in the constructor), so the rings tile the area with no gaps or overlaps and each point inside 100 m lands in exactly one cell ("no alignment errors or data loss"). Square cells suit a Cartesian planner (Hybrid A*); the deck proposed log-polar, which was weighed and not used for that reason.

| Ring | Cell | Covers | Cells |
|---|---|---|---|
| 0 | 5 cm | \|x\|,\|y\| < 10 m | 160,000 |
| 1 | 10 cm | 10–20 m | 120,000 |
| 2 | 20 cm | 20–40 m | 120,000 |
| 3 | 40 cm | 40–100 m | 210,000 |

610,000 cells vs 16,000,000 for uniform 5 cm over 200 m × 200 m. Each ring stores only its band (four strips around the finer ring's hole), so `bytes()` is real. `Cell` = 12 bytes: `zmin`, `zmax`, `count` (uint16), `group` (highest priority seen: dynamic > static > terrain > unknown), `cls` (fine class of that point). A point exactly on the inner hole's edge goes into the strip on that edge's side (a bug the tests caught: it used to go sideways). `tiles()` builds one flat quad per occupied cell at `zmax`, slightly inset so the cell sizes stay visible.

Empty cell = **unknown**, not free. Free space needs ray casting from the sensor (not done). The grid is rebuilt per scan, centred on the sensor (not accumulated with poses yet).

## 7. Measured results (500 scans of sequence 08, this Mac)

| Stage | ms / scan |
|---|---|
| load | 3.24 |
| range image | 1.18 |
| SalsaNext | 17.32 |
| labels to points | 1.17 |
| 2.5D grid binning | 1.44 |
| tiles | 2.03 |
| **total** | **26.4 (37.9 FPS; the LiDAR runs at 10 Hz)** |

- ~124k points/scan → ~57.6k occupied cells/scan.
- Memory: grid **7.32 MB** vs uniform 5 cm 2.5D grid 192 MB (**26×**) vs uniform 5 cm 3D voxels, 10 m tall, 1 byte each, 3.2 GB (**437×**).
- Accuracy (58.9M labelled points): 3 groups **88.31%** (IoU terrain 82.06, static 73.51, dynamic 88.41); 20 classes 84.03% accuracy, **mIoU 46.25%** over 19 present (road 95.98, sidewalk 86.70, car 83.29, building 81.59; truck and motorcyclist 0: barely present in these scans).
- By distance (3 groups): 0–10 m 87.47%, 10–20 m 88.63%, 20–40 m 90.84%, 40–100 m 92.35%. Near is *lower* mainly from grass (terrain) vs low bushes (vegetation = static), SemanticKITTI's hardest pair.
- Below SalsaNext's published ~59 mIoU because: only 500 scans, rare classes, and no KNN post-processing. The network itself is verified identical to PyTorch.

## 8. Phone LiDAR vs the model

The grid works on phone data (it only needs metric points; everything within 5 m lands in the 5 cm ring). SalsaNext runs on phone data but gives mostly arbitrary labels: it learned a 360° 64-beam street scan at 5–80 m (normalisation mean range ~12 m) with remission; the phone gives a narrow forward cone (anything above +3° is squashed into the top row), 0.3–5 m, no remission, indoor scenes. `salsa_reciver` exists to experiment with this (floor height, distance scale, fake remission sliders). The meaningful phone demo is classical labels (RANSAC floor = terrain, planes/clusters = obstacles) into the grid. Fine-tuning on labelled phone scans would be the way to make the model work there.

**`CapReader` delivered disparity, not depth, until this was fixed.** Its init took `supportedDepthDataFormats[0]` (typically `DisparityFloat16`, 1/metres): far objects squashed, near stretched, range sliders behaving backwards. Now `initWithCam:disparity:` (default `initWithCam:` = NO) picks the format by media subtype, and the depth callback converts each frame to the requested type; `useDisparity` can be flipped at runtime. Debug builds log the delivered/handed-out fourcc once (`hdep`/`fdep` = depth, `hdis`/`fdis` = disparity). Applets tuned by eye on disparity (`depth_inplace`'s `* -5`) may need their scale revisited.

**Intrinsics orientation.** `CapReader` rotates depth to portrait (180 × 320) but the calibration is in the sensor's landscape frame: `unproject_depth` swaps fx/fy and cx/cy when the orientations differ, then rescales by depth size / reference size (the same scaling Apple's TrueDepth and LiDAR samples do as `f /= reference / depth`). `fovea::points_from_depth_intrinsics` doesn't handle the rotation and likely skews portrait depth.

## 9. Tests and tools (`tools/salsanext/`)

| File | Checks |
|---|---|
| `convert_salsanext.py` | converts, Python MLX port vs PyTorch (max diff 3.2e-6, argmax 100%) |
| `test_salsanext.cpp` | C++ MLX vs the PyTorch golden output (3.2e-6, 100%), 16.5 ms / 64×2048 scan |
| `make_test_sequence.py` + `test_kitti_projection.cpp` | synthetic KITTI sequence (road, wall, pole, box in HDL-64 geometry) built with SalsaNext's own `laserscan.py`; loader, label bits, pose conversion, class mapping, projection (exact match), back-projection |
| `test_foveated_grid.cpp` | tiling (area, cell ↔ index round trip), 1M random + ring-edge points each in a containing cell, cell sizes by distance, >100 m dropped, cell contents/priority, real scans |
| `benchmark_kitti.cpp` | §7's numbers for any sequence |
| `fetch_kitti_subset.py` | partial KITTI download (§5) |

Build line (each file's header has it): `xcrun clang++ -std=c++20 -O2 -I../../AdityaIntelligenceProMax/AI -I/usr/local/include <file>.cpp -L/usr/local/lib -lmlx -Wl,-rpath,/usr/local/lib -framework Metal -framework Foundation -framework Accelerate -framework QuartzCore -o /tmp/<name>`. The standalone `matrix` sandbox (`matrix-sync/debug`, [[RunningMatrixStandalone]]) holds the matrix-side tests (`tests/*.mm.bak`: sliders/ScalarArg, depth sender pipeline, UDP end-to-end, planes with `host_op`, receiver planes, unprojection).

## 10. Next steps
1. KNN post-processing (accuracy at object edges).
2. Moving vs parked: ego-motion from `poses.txt` (or ARKit on the phone) + the tracker.
3. On-screen metrics: a text label control in `AppletPanel`, FPS / latency / memory / accuracy in the panel.
4. Grid over time: accumulate with poses (a scrolling map), free space by ray casting, costmap (height + class + count → traversal cost), Hybrid A*.
5. Optional: fine-tune on labelled phone scans to make the model useful on the iPhone.
