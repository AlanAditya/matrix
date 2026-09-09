# DAG Invalidation: From Dirty Flags to Versioning

## The Problem: Limitations of the Dirty Flag

Previously, the DAG evaluation engine relied on a boolean `dirty` flag to track changes. When a node was modified (e.g., its local transform was updated via dragging), its `dirty` flag was set to true. During the renderer's `invalidate_pass`, the graph was traversed to clear these flags and set `evaluated = false` along the affected paths.

However, this approach suffered from two critical architectural bugs:

### Bug 1: Premature Consumption in Diamond Dependencies (Shared Nodes)
In a diamond dependency graph, multiple paths lead to the same shared node. If a node was already evaluated but its parent was marked dirty:
1. The first path traversing down would hit the shared node. It would clear the `dirty` flag. 
2. Because of strict evaluation constraints (e.g., nodes shouldn't unconditionally flip `evaluated=false` without resolving their branches), there were edge cases where it failed to appropriately set itself as needing re-evaluation.
3. The `dirty` flag was now gone. 
4. When the second path traversed down to the same shared node, the `dirty` flag was missing. The second path had no way of knowing a change occurred, causing the invalidation signal to be silently dropped and preventing downstream nodes from updating.

### Bug 2: Synchronization Loss Across Multiple Consumer Views
When multiple consumers (e.g., different viewports or renderer instances) shared the same graph data (like different leaf-children of a common root node), the dirty flag acted as a consumable resource.
1. Viewer A would trigger its render loop, initiating an `invalidate_pass`.
2. This pass would *consume* the `dirty` flags on the modified root node and successfully propagate the updates to Viewer A's specific children.
3. Viewer B would then trigger its render loop. However, the root node's `dirty` flag had already been cleared by Viewer A.
4. Viewer B would therefore fail to update its children, causing desynchronization where changes to the root node were completely lost for whichever viewer updated second.

## The Solution: Global Epoch Versioning

To solve this, we migrated from a stateful, consumable `dirty` flag to a monotonic **Version Counting** system.

### Core Mechanics
1. **Global Epoch (`global_epoch`)**: A monotonically increasing global counter. Every time an external mutation occurs (e.g., a node is dragged, scaled, or its buffer is directly modified), the target node's `tape->version` is bumped to `++global_epoch`.
2. **Version Propagation**: Instead of a boolean flag, every primitive in the computational graph stores its own `version` integer (initialized to 1).
3. **Invalidation Pass**: During the `invalidate_pass`, a child node compares its own `version` against the `version` of *all* its inputs/parents:
   ```cpp
   bool inv = false;
   uint64_t new_version = this->version;

   if (parent.tape) {
       if (parent.tape->invalidate_pass(current_pass_id)) { inv = true; }
       if (parent.tape->version > this->version) inv = true;
       new_version = std::max(parent.tape->version, new_version);
   }

   if (inv) {
       this->evaluated = false;
       this->version = new_version;
       return true;
   }
   ```
   If *any* parent's version is strictly greater than the child's version, the child knows the parent has updated since the child last evaluated. The child then invalidates itself (`evaluated = false`), adopts the new maximum version, and returns `true` to propagate the invalidation further up the chain.

### Why Versioning Solves the Bugs
- **Non-Consumable**: Versions are read-only during the invalidation pass. Viewer A checking a parent's version does not reset it. When Viewer B checks the same parent later, the parent's version is still higher than Viewer B's child nodes, so Viewer B also correctly invalidates.
- **Diamond Dependencies**: Shared nodes safely update their internal versions to the maximum of their inputs. Subsequent visits in the same pass (tracked via `last_visited_pass_id`) just return `!evaluated` without modifying state, ensuring the invalidation propagates reliably across all paths.

## Trial, Error, and Edge Cases Encountered

During implementation, we encountered several significant hurdles:

1. **The "Max Version" Requirement for Multi-Input Primitives**
   Initially, we considered a simpler check. However, for primitives with multiple inputs (like `DotPrimitive` which takes `a` and `b_transposed`), it is essential to track the `max` version of *all* inputs.
   If we only had a single local version and didn't use `global_epoch`: if `a` updated first (bumping child to 2), and later `b` updated to 2, the child would check `b.version (2) > this->version (2)` and erroneously evaluate to `false`. 
   **Fix**: Because `global_epoch` strictly increments globally on *any* interaction, a new modification will always produce a version strictly greater than any existing version anywhere in the graph, eliminating tie-breaks.

2. **Missing Input Checks (The Propagation Black Hole)**
   While automating the rollout of the new versioning logic across all primitives (using a regex script), multi-input primitives (`DotPrimitive`, `SliceAssignPrimitive`, `TakePrimitive`) had their secondary inputs accidentally wiped from their `invalidate_pass` functions due to formatting variations.
   **Result**: When the root node was dragged, its new version propagated to `b_transposed`, but because `DotPrimitive` was missing the code to check `b_transposed.tape`, the invalidation stopped dead in its tracks. The children never moved.
   **Fix**: Manually audited and restored the version checks for all secondary inputs (`b_transposed`, `rhs`, `indices`).

3. **Silent Buffer Modifications (Dynamic Scaling)**
   In `GraphViewer::draw()`, the X and Y axes were being dynamically scaled based on camera zoom (`p_scale(...)`). This directly overwrote their local transform CPU buffers. Under the old system, this forced changes locally. Under versioning, because we weren't explicitly bumping the `tape->version` when scaling the axes, the `DotPrimitive` didn't know the scale had changed and skipped GPU execution (reusing the stale metal buffer).
   **Fix**: Explicitly bump `tape->version = ++global_epoch;` whenever directly mutating matrices that act as roots/leaves in the graph, even for localized viewport scaling.
