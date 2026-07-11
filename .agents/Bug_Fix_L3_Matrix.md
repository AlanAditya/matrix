Bug Fix L3: Matrix 

Fix: Metal command buffer race conditions (CTOU & Rug Pull crashes(Concurrent Encoding) )

Symptoms:
The application experienced two distinct, highly unpredictable Metal crashes during multithreaded execution (Main Render Thread + GCD Compute Queue):
1. The "Double Commit" Crash: `-[_MTLCommandBuffer addCompletedHandler:]:1011: failed assertion 'Completed handler provided after commit call'`
2. The "Rug Pull / Encoder (Concurrent Encoding)" Crash: `invalid usage because encoding has ended` or assertion failures when creating compute encoders. 

Root Cause:
`gCommandBuffer` and `gCommandEncoder` in `GlobalGPUManager` were shared globally across all threads without any synchronization. When the main render thread and GCD background compute threads attempted to encode GPU commands simultaneously, they violently overwrote each other's references. 

This led to fatal context-switch race conditions:
- If Thread A was actively encoding commands, a context switch could occur mid-execution. 
- Thread B would wake up, borrow the same global buffer/encoder, and eventually call `endEncoding` to finish its own work or to switch encoders for blitting during copyGPUInplace. 
- When the CPU context switched back to Thread A, Thread A had absolutely no idea that Thread B had just killed the encoder. Thread A would attempt to continue writing GPU commands to the dead encoder, instantly triggering the `invalid usage because encoding has ended` crash. 
- Similarly, a background thread could aggressively commit the main thread's active render buffer before the main thread had even finished adding its completion handlers.

Failed Fix Attempts & Analysis:
Attempt 1: The `oldBuffer` Check (CTOU Failure)
- Idea: Store `oldBuffer = gCommandBuffer` at the start of a function. If `gCommandBuffer` was different at the end, assume *this thread* created it and call `commit`.
- Why it failed: Check-Time-Of-Use (CTOU) race condition. Thread B would see `oldBuffer = null`. Meanwhile, Thread A would create the global buffer. Thread B finishes, sees the buffer is now non-null, mistakenly assumes it was the creator, and aggressively commits Thread A's buffer out from under it.

Attempt 2: The `oldBuffer` Check (CTOU Failure)
- Idea: Store `oldBuffer = GlobalGPUManager.getCommandBuffer()`(returns the current if available otherwise creates a new one so always non null) at the start of a function. If `gCommandBuffer` was same at the end, assume *this thread* created it and call `commit`.
- Why it failed: Thread A created the CommandBuffer and set it to its old buffer in the matrix::eval_metal now CONTEXT_SWITCH B starts its eval_metal on its own matrix and that too sees that same buffer sets it as its own old buffer and takes responsibility for committing it so thus double commit 
- Does not fix the end encoding error due to CONTEXT_SWITCH (concurrent encoding ). 

Attempt 3: `std::atomic<std::thread::id>` Ownership (The Rug Pull Failure)
- Idea: Extend Attempt 2 with Tag the global buffer with the Thread ID of its creator. Only the owning thread is allowed to call `commit`. 
- Why it failed: the double commit problem but not the end encoding error due to CONTEXT_SWITCH (concurrent encoding ). 
  - Scenario A (Encoder Collision): Thread A has an active Render Encoder on the buffer. Thread B grabs the same buffer and tries to create a Compute Encoder. Metal crashes instantly (only one active encoder allowed per buffer).
  - Scenario B (The Rug Pull): Thread B borrows Thread A's buffer and starts writing compute commands. Thread A finishes rendering, verifies it is the owner, and calls `commit`. Metal crashes because Thread B is still actively writing commands to a buffer that was just submitted to the GPU.

The Final Fix: Thread-Local Storage (TLS)
- Solution: The only way to safely submit GPU commands in a multithreaded C++ engine is to give every thread its own entirely isolated sandbox.
- Implementation: Replaced the global variables with `thread_local` storage.
- Overcoming C++ ARC Limitations: C++ `thread_local` strictly forbids Objective-C ARC objects (like `__strong id`). To bypass the compiler error, the command buffers are cast to `__bridge_retained void*` for TLS storage, and safely cast back to `id<MTLCommandBuffer>` during retrieval.
- Result: Threads now safely create, encode, and commit their own independent buffers with zero cross-thread interference.
