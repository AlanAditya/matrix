# Concurrency: who may touch a graph, and how data crosses threads

The `matrix` graph is deliberately single-threaded: node state (`evaluated`, `version`, `last_visited_pass_id`) is plain fields, no atomics, no locks. Keep it that way. Correctness across threads comes from **ownership rules**, the way TensorFlow/JAX/MLX distribute across devices (each device owns its piece of the graph and devices meet only at explicit exchange points), not from locking. See [[versioning]] for dirty tracking, [[MemoryManagement]] for refcounts, [[Applets]] for how applets apply this.

## 1. The rules

1. **One owner thread per graph.** Only the owner writes the graph's leaves and runs its passes (`invalidate_pass` + `eval`). The renderer owns scene graphs on the main thread (`MTKView` calls `drawInMTKView:` on main). An applet owns its compute graph on its own serial queue (or main).
2. **Inputs arrive as messages.** Another thread doesn't write a leaf directly; it `dispatch_async`es a block to the owner queue, which writes and bumps the version. `SliderGroup` does exactly this.
3. **Outputs leave as snapshots.** Hand other threads data nobody will write again: a fresh matrix, or a plain tapeless copy (`matrix::withShape` + `memcpy`), never a handle into the live graph. A handle with a tape lets the renderer's `invalidate_pass` walk into a graph another thread is running.

Under these rules no atomics are needed beyond the refcounts `matrix` already has (atomic `refCount` keeps snapshots alive while they're passed around).

## 2. Why the older applets never raced (the four accidental safeguards)

Applets like `particle_sim` → `PointCloudController::update` were safe without any of this, because:
1. **They didn't really reuse the graph.** `points = points + velocity * 0.1` makes a new node and a new output buffer each tick: every published matrix is an immutable snapshot.
2. **Each thread has its own command buffer and encoder** (`thread_local _thread_gCommandBuffer` / `_thread_gCommandEncoder`, `Mods/GPUManager.h`).
3. **Evaluation finishes before publishing.** `commitCommandBuffer()` commits and `waitUntilCompleted`s, so the published matrix is `evaluated` and the renderer's `eval()` returns immediately: it only reads, never computes.
4. **The swap runs on the renderer's thread.** `update` `dispatch_async`es the assignment to main, between two draws.

The `CapReader` camera has its own staging area too: the capture thread stores the newest `CVPixelBuffer` under a lock; the reader retains it and copies out of it outside the lock.

**A version bump alone is not a publish.** It controls *recomputation*, not *reading*: the renderer draws a node's buffer every frame whether or not its version changed. A bump also *invites* the consumer to re-evaluate that path on its own thread. It's a publish only with both halves: finish writing (GPU done) then bump, **and** never write that buffer again while it may be read (one thread, a copy, double buffering, or a handshake).

## 3. The single float slider case

A slider writing a 4-byte leaf that a compute loop reads: an aligned float store can't tear on Apple Silicon, and a value changing mid-pass is a one-frame blip. The real bug is a **lost update**: ARM can reorder, so the reader can see the new version with the old value, recompute with the old value, record the new version, and stay stale until the next move. Fix: send the write to the owner queue (rule 2), which `SliderGroup(self, queue)` does.

## 4. Pass IDs

`invalidate_pass(pass_id)` only compares pass IDs for equality, to skip nodes already visited in this pass (diamond graphs). Two passes that reuse the same number on a shared node can skip a re-check and leave it stale. Take pass IDs from **`++global_epoch`** (atomic, unique across all threads); versions and pass IDs never get compared to each other, so sharing the counter is safe. `SliderGroup::run_graph_pass` does this. The `Viewer` still uses its own per-instance `current_pass_id` (a one-line change to `current_pass_id = ++global_epoch;` if it ever shares nodes).

## 5. Reusing a persistent graph on a background thread

Allowed. The only constraint is that the consumer must not read output buffer N while run N+1 writes it:

| Option | How | When |
|---|---|---|
| One thread | producer, pass and consumer on the same thread | sliders + static inputs |
| Copy on publish | run the graph, copy the output into a fresh tapeless buffer, publish that | the default for background applets (`point_cloud_environment`, `depth_reciver`, `kitti_replay`) |
| Double / triple buffering | 2-3 graph instances in rotation + `dispatch_semaphore` | heavy continuous compute (not built) |
| Handshake | producer waits until the consumer released N | simple, couples their rates |

## 6. Designed, not built: `ThreadBarrierPrimitive` / mailbox leaf

The user's idea, refined. **A barrier splits a graph** so each side runs on its own thread at its own rate (e.g. camera processing at 30 FPS, the renderer at 120 FPS). It's optional: without it a graph runs as today.

- **Producer side:** the producer runs its side, then `publish()`: evaluate up to the barrier, copy into a free slot (3 slots: writing / latest / in use), bump the barrier's version. It's the producer's job to publish when a change should cross.
- **Consumer side:** the barrier looks like a **leaf**. Its `invalidate_pass` doesn't recurse upstream; it just checks for a newer published slot; `eval` takes the latest slot. The consumer's passes never reach producer nodes.
- **The same primitive at the input end is a "mailbox leaf":** `post()` from any thread writes a pending slot; the owner's `invalidate_pass` swaps it in between passes. That keeps the "change one leaf from anywhere and the renderer figures it out" workflow race-free.
- Only the barrier holds atomics (the slot index). The rule for everything else: a thread writes only buffers on **its own** side.
- To refine: copy vs zero-copy publish (zero-copy fights `update_cache` rules), latest-wins vs queue mode, several consumers, shape changes (reallocate or throw), gradients stop at it, a debug check of which thread runs each side. It depends on §7.

## 7. Known gap: the CPU freeing a buffer the GPU is still reading

`drawInMTKView:` commits without waiting (`waitUntilCompleted` is commented out on purpose). `buildMetalBuffer()` wraps host memory with `newBufferWithBytesNoCopy` and an **empty deallocator**, so Metal doesn't own it. A `dispatch_async` swap on main can drop the last reference to frame N's vertex buffer and `delete[]` it while the GPU still reads it (a window of up to ~1 frame; usually invisible because freed pages aren't returned right away). Fix (not done): keep a copy of every matrix a frame binds and release them in `[cmdBuffer addCompletedHandler:]`. Also: `updateBaseImageV2:` called from background threads reassigns `offscreenTexture`/`width`/`height` unsynchronized on a resize.

## 8. Metal ordering facts relied on
- One compute encoder (default serial dispatch) runs dispatches in order, so a node reading a buffer a previous node wrote in the same pass is fine.
- Host code (`host_op`) can't see GPU results until the command buffer completes: `host_op`'s GPU path commits and waits first.
- A GPU closure should bind computed inputs with `setBufferOrBytes` (the GPU orders it), not read them on the host (see [[Generators]] §6).
