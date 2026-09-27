# Applets: the `Intelligence` class, the control panel, and every applet so far

See [[Concurrency]] for the threading rules every applet follows, [[LidarMapping]] for the PS 26053 LiDAR project most applets here serve, [[Generators]] §6 and [[versioning]] for the graph mechanics sliders rely on.

## 1. Philosophy: functions as applets

`Intelligence` (in `MatrixH.mm`) is instantiated by the SwiftUI app (`_3D_Editor` in `3D_Editor.swift`), and one of its methods is called from `onAppear`. **Each method is a self-contained applet**: a prototype that uses the existing frameworks (the `matrix` engine, the renderer, cameras, networking) instead of a new Xcode project. To switch applet, change the `onAppear` call (the macOS one and the iOS `metalView` one are separate).

An applet typically:
1. builds its inputs (camera, LiDAR, a file, a UDP receiver),
2. creates its UI (sliders, toggles, buttons) through `SliderGroup`,
3. builds a `matrix` graph **once**,
4. runs it: event-driven (`ui.on_change`) or loop-driven (`attachToGCDQueue:withFPS:onQueue:`),
5. publishes results to the renderer (`updateBaseImageV2:`, `PointCloudController`, `TriangleMeshController`).

## 2. Parameters are graph leaves (why sliders don't need callbacks)

The old approach: `__block` floats, one `UIComponent` callback per slider, each callback recomputing everything. Problems: duplicated recompute code, heavy work on the main thread at 60+ Hz, no coalescing, races with the render thread.

The model now: **a slider owns a 1-element leaf matrix** (`matrix::leaf_scalar`). Moving it writes the value into the leaf's buffer in place and calls `mark_changed()` (`version = ++global_epoch`, `evaluated = false`). Any graph built on that leaf, arithmetic or a generator parameter, re-runs **only the dirty path** on its next pass, into the same buffers (compile once, execute never allocates, see [[MemoryManagement]]).

- Constants in graphs (`x * 2.0f`) stay untracked `matrix::scalar`s; only slider leaves are tracked, so tracking costs nothing extra.
- Generator parameters (e.g. `perlin`'s scale) become slider-drivable through `ScalarArg` (a constant *or* a tracked matrix) and generator **inputs** (`GeneratorPrimitive(cpu, gpu, inputs)`), see [[Generators]] §6.
- Algorithms that aren't graph ops (RANSAC, normals) become lazy nodes with `matrix::host_op(shape, dtype, fn, inputs)`, parameters passed via `HostArg`. They re-run only when one of their own inputs changes.

**Push vs pull.** A slider *pushes* a change signal (a version bump); the consumer *pulls* the values when it runs a pass. Ten slider ticks between two passes become one recompute. That's `SliderGroup::on_change` (a `DISPATCH_SOURCE_TYPE_DATA_OR` source coalesces the ticks) or a loop that calls `SliderGroup::run_graph_pass(out)`.

## 3. `SliderGroup` (`UI/MatrixSliders.h`)

```objc
SliderGroup ui(self);                      // owner thread = main (static-input applets)
SliderGroup ui(self, queue);               // owner = the applet's own serial queue (loop applets)
ui.section(@"Noise");
matrix scale = ui.slider(@"Scale", 0.5f, 20.0f, 5.0f);      // Float leaf
matrix oct   = ui.slider_int(@"Octaves", 1, 8, 4);          // Int32 leaf (rounded)
matrix on    = ui.toggle(@"Colour planes", true);           // Float leaf, 1 or 0: blend with on*a + (1-on)*b
ui.button(@"Invert", ^{ /* runs on the owner queue; may write leaves then mark_changed() */ });
ui.set(scale, 7.5f);                       // set from code: writes the leaf AND moves the slider on screen
ui.on_change(out, ^(matrix& o) { ... });   // re-run `out` on the owner queue when anything changes, then publish
SliderGroup::run_graph_pass(out);          // one pass (invalidate with ++global_epoch, eval if dirty) for loop applets
```

- The UI calls arrive on the main thread; each write is sent to the **owner queue** with `dispatch_async` (inline if the owner is main). The value is written on the same thread that reads it: no torn reads, no stale "new version, old value" read on ARM, no atomics in the graph. See [[Concurrency]].
- `set()` is `const` (it only touches shared state behind a pointer), so it can be called from a button block's captured copy of `ui`.
- `on_change`'s `publish` receives the graph's **live** output buffer. It's rewritten by the next pass, so consume it right away (`updateBaseImageV2` copies into a texture) or hand the renderer a plain copy.

## 4. `AppletPanel` (`UI/AppletPanel.h/.mm`): the native panel

The old SwiftUI `ControlPanel` over `UIComponent`s is commented out in `3D_Editor.swift`. `UIComponent` / `Intelligence.components` still exist only because `RGBCam` and `GeometryNodeTesterV2` read their values; they are no longer displayed. Don't use them in new code.

`AppletPanel` is plain Objective-C (MatrixH.mm is the Swift bridging header, so it's visible to Swift) with an AppKit implementation (macOS) and a UIKit one (iOS). `Intelligence` creates it in `init` as `appletPanel`; Swift shows `appletPanel.view` through the `ControlPanel` representable (same name and call site as the old SwiftUI panel).

- Settings-style layout: small section headers over rounded cards, one row per control, hairline separators.
- Sliders: value readout (editable on macOS: type + Return), right-click "Reset to Default" (macOS) / double-tap the value (iOS), integer sliders snap and show ticks.
- Buttons: push buttons (macOS), Liquid Glass buttons on iOS 26 (tinted before).
- Toggles: `NSSwitch` / `UISwitch`.
- `setValue:forSliderTitled:` moves a slider without calling its `onChange` (used by `SliderGroup::set`).
- Every method is safe from any thread (UI work hops to main); callbacks arrive on main.

### iOS panel modes (`3D_Editor.swift`)
`@State var panelMode: IOSPanelMode = .floating` (default) or `.docked`.
- `.floating`: a Maps / Find My style sheet over the renderer. Detents `.height(110)`, `.medium`, `.large`; **resized only by dragging the grabber** (`presentationContentInteraction(.scrolls)`: scrolling scrolls the controls); background interaction up to `.medium`; can't be swiped away; Liquid Glass background on iOS 26 (`PanelSheetBackground`: no `presentationBackground`), `.regularMaterial` before. `appletPanel.translucent = YES` (clear background, see-through cards).
- `.docked`: the original opaque panel, 280 pt tall, under the renderer.
- The renderer is edge to edge (`.ignoresSafeArea()`) with `.defersSystemGestures(on: .all)` (the first edge swipe goes to the app).

### iOS gestures on the renderer (`MyMetalView`, MatrixH.mm)
Orbit (1 finger) and pan (2 fingers) are the user's own **raw `touchesMoved`**: hand-tuned, keep them. Zoom is a `UIPinchGestureRecognizer` with the default `cancelsTouchesInView = YES`: once a pinch starts, the raw touches (and so the pan) stop. Keep it that way: with `NO`, the pan's `updateOLD()` ran under the pinch's cumulative `gesture.scale` and zoom compounded into NaN almost instantly. What makes them coexist with the sheet:
- `RendererTouchShield`: a recognizer on the renderer that **recognizes as soon as a touch lands and ends when the last finger lifts**, never cancels touches, and makes every recognizer on the renderer's ancestors (up to the `UIWindow`, where the sheet puts a pan) require it to fail (`requireGestureRecognizerToFail:`, set on every touch because the sheet adds recognizers later). Recognizing (instead of failing on lift) matters: failing released the sheet's pending pan with the swipe's velocity and made the sheet jump.
- Pinch and shield are added once (`didMoveToWindow` runs again on re-parenting).
- Debug builds log who cancelled the touches (`[MyMetalView] touches cancelled by ...`), skipping the shield itself.
- The handlers call `[renderer cameraGesture:…]` / `[renderer cameraUpdateOLD]` with the same arguments they used to pass to `cam.handleMouseEvents` / `cam.updateOLD`. The renderer hands them to the current page's viewer (`Viewer::handle_camera`), which calls its own camera; without pages they go straight to `pRender->cam`, as before.

### Floating toolbar capsule (`UI/AppletToolbar.h/.mm`, `UI/AppletToolbarView.swift`)
A small Liquid Glass capsule at the **bottom right of the renderer** (a thin material before macOS 26 / iOS 26) for quick actions. `AppletToolbar` is a plain Objective-C model (`Intelligence.appletToolbar`); SwiftUI draws it (`AppletToolbarView`), so the glass and pop-up menus are the system's own on both platforms. `.appletToolbar(toolbar, bottomInset:)` pins it to a view; in iOS `.floating` mode it sits above the sheet's 110 pt detent (the sheet covers it at `.medium`/`.large`).

```objc
[self.appletToolbar addButtonWithIcon:@"camera" title:@"Snapshot" onPress:^{ ... }];
[self.appletToolbar addToggleWithIcon:@"grid" title:@"Infinite Grid" value:YES onChange:^(BOOL on) { ... }];
[self.appletToolbar addMenuWithIcon:@"ellipsis" title:@"Camera"
                         itemTitles:@[@"Top", @"Front", @"Right"] itemIcons:nil
                           onSelect:^(NSInteger i) { ... }];
```
- Icons are SF Symbol names; `title` is the tooltip / accessibility label / menu header.
- Adding is safe from any thread; callbacks run on **main**, the thread the renderer draws on, so they can write `pRender` state (`renderInfiniteGrid`, `cam`) directly.
- Defaults added in `init` (`addDefaultToolbarItems`), present for every applet: an **Infinite Grid** toggle (`pRender->renderInfiniteGrid`) and a **"..." Camera** menu (Top / Front / Right via `Camera3D::setViewDirection`, which keeps the orbit distance: Front = -z, the side the camera starts on; Right = +x; Top = +y). Applets append after them; `removeAll` clears the defaults too.
- An applet that writes `renderInfiniteGrid` itself (e.g. `particle_sim`) leaves the toggle's highlight out of date.

## 5. Renderer hand-off helpers (`SOPs/Controllers.cpp`)
- `PointCloudController::update(points, colours)` / `clear()`: swapped in on the main thread. `clear()` swaps in a truly empty mesh (no buffer, no Metal wrapper). Don't hide by sending a 0-point cloud through `update()`: that creates a zero-length Metal buffer.
- `TriangleMeshController::update(vertices [V,3], colours [V,4], indices [3T])` / `clear()`: arbitrary coloured triangles (the 2.5D grid tiles). It supplies all-zero UVs because `DAGNodeVertexShader` declares a UV buffer at slot 5 and Metal's validation layer requires it bound (`MeshPrimitives` meshes always have UVs).
- Hand these **plain copies** (tapeless matrices), never a handle into a graph another thread is running (see [[Concurrency]]).
- Coordinates: the viewer is y up, looking down -z. KITTI / sensor coordinates (x forward, y left, z up) map to the viewer as `(-y, z, -x)`.

### Viewers, viewports and pages (`SOPs/Viewer.cpp`, `SOPs/ImageViewer.h`)
- Every `Viewer` has `visible` (hidden = not drawn, no events), `viewport` (x, y, w, h as fractions of the drawable, top-left origin; the renderer sets the Metal viewport + a clamped scissor rect) and an optional `camera` (null = `pRender->cam`; the renderer keeps its aspect in step with the viewport). Defaults = the old behaviour. macOS hit-testing still uses `pRender->cam` over the whole view.
- `ImageViewer` (`[pRender makeImageViewer]`): the base image as a Viewer; `update([H, W, 4] BGRA)` from any thread.
- **Pages (iOS only)**: `[self showPages:std::vector<std::shared_ptr<Viewer>>{a, b} titles:@[...]]` shows the viewers one at a time over the whole renderer. A horizontal swipe on the panel (not on a control, so sliders still work) or a tap on its page dots moves between them, with a spring settle; mid-swipe both pages are drawn side by side in one pass (shifted viewports). Orbit / pan / zoom (rerouted through `Viewer::handle_camera`) and the toolbar Camera menu move the current page's camera; `interactive = false` pages ignore them. The grid is drawn per page (`show_grid`). Skip work for hidden pages with `[pRender isPageOnScreen:i]` (any thread).

## 6. Applet catalogue

Every method of `Intelligence` (`MatrixH.mm`, `@implementation Intelligence`, roughly lines 12714-17045), grouped by era. Three generations live side by side:

- **Current**: the lazy `matrix` engine, `SliderGroup` controls, `attachToGCDQueue` loops, `Viewer` + controllers, `updateBaseImageV2:`. Copy these.
- **matrix demos / tests**: also the `matrix` engine, but no sliders; one-off demos, console test suites, experiments.
- **Legacy**: the templated `MatrixH<N, T>` API, `NSTimer` loops (often on the main thread after a `sleep(2)`), `UIComponent` sliders, `updateBaseImage:` (not V2), and the old render queues (`_objectQueue`, `_objectQueueInstanced`, `_ComposedObjectQueue`, `_NodesQueue`, `_NodesQueuePtr`, `_meshpointCloudObjectQueue`). Kept for reference; don't model new code on them (see [[legacy_vs_modern_op_pattern]] for the engine-side equivalent).

**Running now** (`3D_Editor.swift` `onAppear`): macOS `kitti_replay`, iOS `depth_sender`.

**Picking a template for a new applet**: static input + sliders → `perlin_playground`; heavy lazy work (RANSAC, normals) on its own queue → `point_cloud_environment`; a capture loop sharing a queue with sliders/buttons → `depth_sender`; a network receiver → `depth_reciver`.

### 6.1 Plumbing (not applets)

| Method | What it does |
|---|---|
| `init:` | Creates `components` (legacy), `activeTimers`, `appletPanel`, `appletToolbar`; one Metal device; two `MyMetalView`s (`view1` 120 FPS, `view2` 30 FPS; BGRA8 sRGB, Depth16, 4x MSAA, red clear colour) with a `Renderer` each (`pRender` view 0, `pRender2` view 1; `pRender2` also draws `view1`'s `debug_view`); `sidePanel` over `pRender->_NodesQueuePtr`; `Red/Green/Blue = 1`; then `addDefaultToolbarItems`. |
| `addDefaultToolbarItems` | Infinite Grid toggle + Camera menu (Top / Front / Right), see §4. |
| `attachToGCDQueue:withFPS:` | Timer loop on a new serial queue `com.engine.compute`. |
| `attachToGCDQueue:withFPS:onQueue:` | Same on the caller's serial queue (share it with a `SliderGroup`). Timer kept alive in `activeTimers`; nothing ever cancels it. |
| `attachContinuousLoop:` | Runs the block back to back, as fast as it can, on `com.engine.heavycompute` (re-dispatches itself inside an autorelease pool). Only `new_concurrancy_model` uses it. |
| `updateCam:::` | Empty stub. |
| `updateCamProjection:` | Sets `pRender->cam.updated = false`. |
| `updateCamOLD:` | `cam.updateOLD()` on view 1 or 2. Only called from commented-out code in `garbage.swift`; camera input now goes through `MyMetalView`. |
| `processAudioSample:` | RMS of an `int16` `MatrixH` buffer, logged every 30 calls (used by `MicTesting123`). |

### 6.2 Current applets (`matrix` + `SliderGroup`)

| Applet | Platform | What it does |
|---|---|---|
| `perlin_playground` | both | Demo of the slider system: Perlin noise with tracked `scale`/`octaves`/`persistence` (ScalarArg), colour sliders, an "Invert" button driving an unshown leaf. Graph built once, `on_change` publishes to `updateBaseImageV2`. |
| `point_cloud_environment` | macOS (hard-coded path) | Loads a PLY scan (`~/Downloads/Scan at 23.49.ply`), lazy `fovea::estimate_normals` → `segment_planar_surfaces_labels` → `colour_plane_labels`, with sliders for voxel size, max planes, distance, normal angle (90 = off), min inliers, iterations, and a "Colour planes" toggle. Own queue; publishes a plain copy and logs the inlier count per plane. The older fixed version is commented out below it. |
| `depth_sender` | iOS | LiDAR + camera, two **pages** (swipe on the panel): camera (top) over jet-coloured depth (bottom) at the depth map's size, and the depth unprojected into a camera-coloured point cloud with its own orbit camera; only the page(s) on screen are computed; "Near/Far" sliders + "Auto range" button (mean ± 2σ clamped to min/max, moves the sliders via `ui.set`); **"Send"** sends depth `[H,W]` Float16 + camera `[H,W,4]` BGRA UInt8 + `CameraIntrinsics` (aux), 3 times (UDP drops whole frames). Debug builds log depth min/mean/max once a second. |
| `depth_streamer` | iOS | Streams **only depth** + intrinsics continuously at 30 FPS (for `salsa_reciver`), shows jet depth (0.2-5 m) on the phone, logs every 90 frames. The older image-streaming version is commented out below it. |
| `depth_reciver` | both (used on the Mac) | Receives `depth_sender` frames. Persistent leaves, graph built once on the first frame: pixel-grid or **"Unproject with intrinsics"** (default on) point cloud, camera colours, lazy normals → RANSAC planes → `colour_plane_labels`; sliders for all RANSAC parameters. Re-runs only on a new frame or a slider change (RANSAC is random: re-running each tick would flicker). |
| `salsa_reciver` | macOS | Experiment: SalsaNext on the phone's streamed depth. Unproject → KITTI axes → SalsaNext → labelled points and/or 2.5D grid. Sliders to make phone data look like KITTI: phone height above the floor (moves the floor to -1.73 m), distance scale, fake remission. Logs the top predicted classes once a second. Expect mostly wrong labels (see [[LidarMapping]] §7). |
| `kitti_replay` | macOS | The PS 26053 pipeline on SemanticKITTI: plays a sequence (Play, Frame scrub slider, FPS), SalsaNext per scan, labelled points and/or the foveated 2.5D grid (tiles at max height, brighter = taller), colour by group or class, ground-truth toggle, "Reset accuracy". Logs per-scan timing and, every 50 scans, group IoU + accuracy by distance band. Paths: `kSequence`, `kModel` at the top. |

### 6.3 `matrix` demos, experiments and test suites (no sliders)

| Applet | Platform | What it does |
|---|---|---|
| `basics` | both | Teaching example: a few constructors (`ones`, `zeros`, `rand`) printed; `matrix::fromImage()` split into channels with `img[R(), R(), c]` and restacked RGBA → BGRA for `updateBaseImageV2`. |
| `random_test` | both | 1080x1920 BGRA image from three independent `perlin` channels. The `randint` image above it is built but unused; a commented-out accumulate-2000-frames benchmark shows the `eval_metal()` + `releaseTape()` every-N pattern. |
| `differential_eqn` | both | Euler-integrates dy/dt = -sin(t) per pixel on a 100x100 field whose start time is `perlin * 2π` (so y = cos t, animated), shown as a grey image at 30 FPS. **Collapses the tape every 30 frames** (`eval_metal` + `releaseTape`) because `y`/`t` recur; the reference for loops with state. |
| `particle_sim` | both | 20³ = 8000 particles on a grid with random colours, pulled to the origin by an inverse-square force (`v -= r/|r|³`, `x += 0.1 v`) at 30 FPS; a Perlin-textured (blue sea / green land) `SphereController` "Earth" at the centre, texture also shown as the base image. Turns the infinite grid off. Unlike `differential_eqn` it never releases the tape of the recurring `points`/`velocity`. |
| `particle_sim_intersteller` | both, **not declared** | 200³ = 8 M point grid with random colours, static. Unfinished start of a bigger `particle_sim`. Not in the `@interface`, so Swift can't call it. |
| `particle_render_graph` | both | 500x500 Gaussian bump as a point cloud, colour-mapped blue → green → red by height (`apply_colormap`), plus a "Glow" copy with 10x point size and depth bias -10; animates the amplitude (`10 sin(0.05 frame)`) at 60 FPS. |
| `resampler_tester` | both | Camera → `matrix::resample` to 20x20 (nearest) → back up to 1000x1000: a live pixelation. Holds two unused lambdas that were the prototypes of `resample`: `upsample` (corner-aligned `linespace` + `take`) and `resize_nearest` (uniform bins, `src = dst * src_size / dst_size`). |
| `plot_EDs` | both | `GraphViewer` plotting `x + perlin` over [0, 10] in red; the `Ed3` curves it was written for are commented out. |
| `render_graph` | both | Live part: a 3000-step ±1 random walk plotted in a `GraphViewer` (a `LineController` of `sin` is built but never added). The rest is a commented-out scrapbook: instanced cube hierarchy via `GeoNode` transforms, animated `PlotterViewer`, a microphone oscilloscope, a 3D helix `LineController`, solid + wireframe (pipeline state 6) + joints (state 5) on one mesh, the Stanford bunny. |
| `rendering_graph` | both | Empty. |
| `computational_graph` | both | `CPURasteriser` demo: "Hello World" drawn twice (`rast.drawText` and `MatrixH::drawText`) and a sine-wave `Plot` with titled axes into a 1000x1000 image shown via `pRender->offscreenTexture`; a `drag_event_subscribers` callback pans the plot's axes and redraws it on drag. Then times `(x + y) + x + y` over 1.2 M elements on the legacy `MatrixH` tape. Commented: line/circle/Bezier/polygon/`Path2D` rasteriser calls. |
| `computational_graphV2` | both | The big console test suite, CPU vs Metal with expected values printed: `cross` (+ VJP through `grad_graph_gpu`), element-wise `max`/`min` with broadcasting, `sum`/`mean`/`rms` per axis, `jit_graph_gpu` reused on two inputs, gradient descent on a slope with `compile_metal`/`execute_metal` + `visualise_graph_v2`, a `sin` interference image to `offscreenTexture`, `eye`/`concat`, batched matmul, single-neuron training, `multi_jit_graph_gpu` (several outputs, one execution), 2-parameter linear regression, `conv1d`/`conv2d`/N-D `conv`, `clamp`, `take`, `randint`/`rand`/`randn`, `argmax`/`argmin`, and `resample` 1D/2D nearest/linear incl. the identity-tail and collapsed-tail fast paths. Run it after touching an op. |
| `inverse_finder` | both | Learns the inverse of a 3x3 matrix by gradient descent on ‖I - w·x‖² (`grad_graph_gpu`), 1 M steps at lr 0.001, collapsing the tape every 1000; prints `w`. |
| `new_concurrancy_model` | both | Camera → horizontal-edge filter (3x3, rows -1/0/+1; the Gaussian above it is overwritten) as a depthwise `conv2d` (groups 3) → alpha re-added → shown, in `attachContinuousLoop` (unthrottled). Compiles `jit_graph_gpu(f, …)` but the loop calls plain `f`, so the JIT result is unused. |
| `MicTesting1234` | both | Microphone (`MicrophoneAudioCapture`, 1024 samples) → loudness `clamp(|Σ| * 0.05, 0, 1)` → multiplies the camera frame: the picture gets brighter as you talk. 60 FPS, prints the loudness each tick. |

### 6.4 iOS sensor experiments (superseded by `depth_sender` / `depth_streamer`)

| Applet | What it does |
|---|---|
| `depth_inplace` | Camera + LiDAR depth → points on a fixed ±5 pixel grid with z = -5·depth (no intrinsics) → `fovea::classify_ground_colors_green_tint` (ground tinted green over the camera colours) → `PointCloudController`, 30 FPS. A `GravitySensor` is pulled each tick but unused. Commented alternatives: `points_from_depth_intrinsics`, `ground_overlay_image`. |
| `arkit_inplace` | Same idea from `ARSessionCapture` (ARKit's depth map, zero-copy) with `classify_ground_colors`. Prints a 10x10 corner of the X/Y grids **every frame** (debug leftover). Point-cloud tensor, gravity and a colour-mapped variant are commented out. |
| `iOS_Depth` | Legacy `MatrixH`: one camera frame + depth → a `PointCloudNode` pushed once. Its update block is empty and never scheduled. |

### 6.5 Legacy `MatrixH<N, T>` applets

All macOS only except `EXR`, `vec_field`, `NewEfficientApproachTester` (both platforms): from `concatLogic` down, the implementations sit inside `#if !TARGET_OS_IPHONE`.

| Applet | What it does |
|---|---|
| `Logic` | The earliest applet. A triangle, and a cube repeated 4x linearly then 10x radially (`ArrayModifier`) on `pRender`, spinning. Camera → `HandDetector`: the distance between hands drives the array's rotation; hands are drawn and composited with an "AAADDII" text layer (invert → `blend` → `chromaKeyImg`) onto `pRender2`. |
| `concatLogic` | Still image + a "TUSHUUU" text layer, composited by invert / `blend` / `chromaKeyImg` / `blend`. |
| `PointCloudLogic` | Loads the bundled `11_7_2024.g` mesh (`ObjMesh`) into `_meshpointCloudObjectQueue`. |
| `MotionDetection` | Camera: the first frame is cached inverted; every frame, `cached + frame` then invert, so unchanged pixels cancel out and motion shows. 30 FPS `NSTimer`. |
| `MemoryTestLogic` | Camera → float → per-channel tint by the `Red`/`Green`/`Blue` properties, reusing one `cached` buffer in place (`Add` into it, `To` out) to test allocation-free updates; raw frame on `pRender`, tinted on `pRender2`. |
| `RGBCam` | Camera tinted by three `UIComponent` sliders (Red/Blue/Green) on a background run loop. The panel no longer shows `UIComponent`s, so the tint is stuck at 1 (§4). |
| `VideoMerger` | Decodes a whole video into memory (`fromVideo`, `~/Downloads/IMG_0605.mov`) and plays it on a 120 Hz timer, holding the last frame. |
| `VideoPlayer` | Same, plus `AVPlayer` for the audio and an RGB-tinted copy on `pRender2`. Its index can run one past the end (`i < shape[0]` then `i++`) and read out of bounds on the last frame. |
| `_3dVideoPlayer` | **Not declared.** Copies video frames into a textured quad's `MTLTexture` each tick (video on a 3D surface). |
| `_3dEditor` | Grid-textured quad on view 1; view 2 shows view 1's camera from outside (cube at the camera, circle at the target, a dotted line between them). |
| `MatTester` | `MatrixH` prints (`Range`, `Transpose`, shifted `eye`, `Dot`) and an MLX `grad` + `vmap` of x². |
| `MatTesterV2` | `MatrixH` slicing (`printNonCont`), `Transpose`, `conv` with `ConvMode::Same`; MLX arrays created and unused. |
| `DepthFromImage` | `fromImageWithDepth` → one instanced `TriangleNode` per pixel at (2x/W, 2y/H, depth), coloured by a custom pipeline (`CustomShader1` / `lightingFragmentShader`, image bound at vertex slot 4); the base image shows the depth in grey. |
| `convPerChannel` | Named after the per-channel blur experiments now commented out (plus OpenCV / MediaPipe dylib loading). Live code: camera on `pRender2` (timed as "AadiXLA") and 478 face + 42 hand landmark icospheres whose positions come from a MediaPipe call that is commented out, so they never move. |
| `NewEfficientApproachTester` | A quad, a 1000-point noise cloud, then `fromImageWithDepth` → a point cloud of (x/W, y/H, depth) in image colours; image as the base. Commented: text burned into a video and saved, `UIComponent` noise sliders. |
| `GeometryNodeTester` | Strided `MatrixH` views over a mesh's `Vertex3D` buffer (normals, positions); a 2D `LineNode`, a 3D helix `LineNode3D` and two arrows (`arr2` starts at `arr1`'s tip); a 1 Hz timer slides a cosine window along the line and turns the arrows. The per-vertex arrow instances are built but never queued, and their `new[]` transforms leak. |
| `GeometryNodeTesterV2` | Icosphere vertices displaced by noise as sphere instances; a noise-textured quad; four `UIComponent` sliders (Scale/Detail/Roughness/mult) whose callbacks read `_components[1…4]`: index 4 doesn't exist, so they'd throw if the panel still fired them. A background loop runs the custom Metal kernel `custom_kernel1` with a time uniform into a 1080x1920 image at 30 FPS (a shader toy). |
| `TextTesselator` | A tessellated 3D "Hello World" `Text3D` shape. |
| `TeselatorTester` | A draggable polygon with a draggable circle on each vertex; dragging a circle re-tessellates the polygon (`ConcavePolygon::updateShape`). Prints a `Derivative` of the points. |
| `MicTesting123` | `MicReader`, 200 `int16` samples at 60 Hz → `processAudioSample` logs the RMS twice a second. The image reshape after it is dead code. |
| `EXR` | Declared as `EXR_CVE_EXPLOIT`. Saves a white EXR, then hand-writes a malformed one over it (a 1638400 x 6553600 data window, every scanline offset pointing at one small chunk) and reads it back with `matrix::fromImage`, logging ImageIO's header dictionary: a probe of the system EXR decoder. Hard-coded path under `~/Documents/OUTPUT_IMAGES`. |
| `vec_field` | 100 point charges on the unit circle (instanced `CircleNode`s); the electric field on a 10³ grid computed vectorised (Σ 0.001 r/|r|³ over the charges) and drawn as instanced `ArrowNode`s. |

### 6.6 Header vs implementation mismatches

- `EXR_CVE_EXPLOIT` is declared, `EXR` is implemented: calling the declared name from Swift fails with "unrecognized selector".
- `particle_sim_intersteller` and `_3dVideoPlayer` are implemented but not declared.
- The `@interface` declares everything unconditionally, but `depth_inplace`, `arkit_inplace`, `depth_sender`, `depth_streamer`, `iOS_Depth` exist only on iOS, and `kitti_replay`, `salsa_reciver` and everything from `concatLogic` to `processAudioSample` (incl. `Logic`, `updateCam…`) only on macOS. Call only the ones your platform has.

### 6.7 Shared helpers

`unproject_depth(depth, cam)` (static, MatrixH.mm, above `kitti_replay`): pinhole unprojection that handles `CapReader`'s portrait depth vs landscape calibration (swaps fx/fy and cx/cy), rescales intrinsics from the reference resolution, returns viewer axes (x right, y up, -z forward), metres.

Phone ↔ Mac streaming uses `QuantumSenderUDP` / `QuantumReceiverUDP` (`QuantumStreamUDP.h/.mm`) to `192.168.2.1:8080`. `stream_frame(vector<matrix>, aux, size)` sends several matrices + one aux blob per frame. `pull_latest_data` returns true **once per new frame** (it resets `has_ready_frame`); the receiver's `dag_leaf_matrices` are refilled in place on each pull, so copy them (or copy into your own leaves) before building on them.

`attachToGCDQueue:withFPS:onQueue:` runs a loop on a serial queue the applet made, so a `SliderGroup` can share it. `attachToGCDQueue:withFPS:` makes its own queue (unchanged).
