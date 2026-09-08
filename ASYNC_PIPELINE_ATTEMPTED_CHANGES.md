# Async Pipeline Attempted Changes — llama_wukong

**Status: ROLLED BACK** (2026-09-08)
**Reason: malloc corruption on model load — "smallbin double linked list corrupted"**
**Rollback target: 600d5c70c** (docs: mark NUMA replication Task 1 complete)

UnobligatedRascal | Created: 2026-09-08

Comprehensive record of all async pipeline changes attempted on llama_wukong between
NUMA replication completion (600d5c70c) and rollback. Preserved for future reference
when re-attempting async pipeline integration.

---

## Commits to Roll Back

These 5 commits were rolled back (HEAD → 600d5c70c):

| Commit | Subject | Type |
|--------|---------|------|
| 20470e5a7 | docs: update TODO tracking, NUMA replication notes, and project docs | docs + minor code (NUMA env var disable) |
| e94d7ad0e | fix: initialize RoPE LUT per-GPU at backend context creation | code |
| a3575507e | feat(sm_37): RoPE sin/cos lookup table for fast transcendental approximation | code |
| fe02f18e5 | feat: async GPU pipeline integration + fit.cpp tensor-split support | code |
| 073a8089b | docs: add standalone async_pipeline_plan.md | docs |

Docs-only commits between NUMA completion and async work (also rolled back):
| Commit | Subject |
|--------|---------|
| 7a7a3b794 | docs: add ARCHITECTURE_REFERENCE.md from recovered pitch |
| f0c0eed3c | docs: preserve original recovered pitch as historical reference |

---

## Failure Analysis

**Error:**
```
ggml_numa_replicate: node 0 has 36 CPUs
ggml_numa_replicate: node 1 has 36 CPUs
ggml_numa_replicate: ENABLED across 2 nodes, main process on node 0
loading model '/mnt/512gb_ssd/models/Qwen3.6-27B-Fusion-711-Uncensored-MTP-Q8_0-Opt.gguf'
malloc(): smallbin double linked list corrupted
Aborted
```

**Timing:** Failure occurs during model load (after NUMA init, during `load_model`).

**Likely culprits (in order of suspicion):**

1. **async-pipeline.cuh lazy init race condition** — The `ensure_init()` method uses compare_exchange_strong + mutex double-check, but:
   - Multiple threads (from ggml-cpu numactl) may call `ggml_cuda_async_mark_layer_complete()` concurrently during model load tensor distribution
   - Device context switching under lock with `ggml_cuda_set_device(dev_id)` can corrupt CUDA context state across threads
   - The destructor `~ggml_cuda_async_gpu_ctx()` destroys streams/events that may still be in use

2. **ggml-cuda.cu unreachable code after return** — The `ggml_cuda_init()` function has:
   ```cpp
   return info;   // <-- function returns here
   // Code below is DEAD/UNREACHABLE:
   #ifdef GGML_USE_NCCL
   ggml_cuda_nccl_init_env();
   #endif
   ggml_cuda_async_pipeline_global.enable();
   ```
   This was introduced in fe02f18e5. The async pipeline is never actually enabled at init time because the enable() call is after `return info;`. However, the global object is still constructed, and the NCCL env var init never runs.

3. **RoPE LUT initialization timing** — Commit a3575507e adds RoPE LUT init in ggml_cuda_init() BEFORE the return, but commit e94d7ad0e moves it to ggml_backend_cuda_context constructor. Both paths call `cudaSetDevice()` during init which could conflict with NUMA replication's own device context setup.

4. **nccl-stagger.cuh double-inclusion** — nccl-stagger.cuh is included both directly in ggml-cuda.cu and via common.cuh (through rope-lut.cuh include chain in e94d7ad0e), potentially causing symbol conflicts.

5. **fit.cpp tensor-split changes** — The removal of the early abort for SPLIT_MODE_TENSOR could cause memory calculation issues during model load with 27B model on 8x K80.

---

## Detailed Change Record

### Commit fe02f18e5 — async GPU pipeline integration (PRIMARY CODE CHANGE)

**New files:**
- `ggml/src/ggml-cuda/async-pipeline.cuh` (219 lines)
- `ggml/src/ggml-cuda/nccl-stagger.cuh` (103 lines)
- `ggml/src/ggml-cuda/numa-gpu-bind.cuh` (79 lines)

**async-pipeline.cuh:**
- `ggml_cuda_async_gpu_ctx`: Per-GPU context with prefetch stream, double-buffered completion events, lazy init with compare_exchange_strong + mutex
- `ggml_cuda_async_pipeline`: Global pipeline context, `enable()` sets flag only (lazy GPU init)
- `ggml_cuda_async_mark_layer_complete()`: Records cudaEvent on compute stream
- `ggml_cuda_async_schedule_prefetch()`: Queues async transfer on prefetch stream after waiting on completion event
- Double-checked locking in `ensure_init()` — potential race if threads proceed with null pointers before init completes

**Critical bug in design:**
```cpp
void ensure_init(int dev_id) {
    bool expected = false;
    if (!initialized.compare_exchange_strong(expected, true)) {
        return; // Another thread won — BUT we may continue with null pointers!
    }
    std::lock_guard<std::mutex> lock(init_mutex);
    // Double-check...
}
```
The CAS-winner proceeds, but the CAS-losers return early with `initialized=true` while streams/events are still nullptr. Subsequent calls to `record_layer_complete()` check `if (!initialized.load()) return;` — they'll NOT return because initialized is true, but the pointers are still null. The null checks in record_layer_complete() save it most of the time, but race windows exist.

**nccl-stagger.cuh (initial version):**
- `ggml_cuda_nccl_init_env()`: Sets NCCL_P2P_LEVEL=1, NCCL_IB_DISABLE=1, NCCL_SOCKET_NTHREADS=2, NCCL_ALGO=Ring
- `ggml_cuda_nccl_staggered_allreduce<T>()`: Two-phase allreduce — NUMA0 completes first, then NUMA1
- Hardcoded GPU groups: GPUs 0-3 = NUMA0, GPUs 4-7 = NUMA1

**numa-gpu-bind.cuh:**
- Thread pinning via `sched_setaffinity()` to NUMA-local cores
- Hardcoded core lists: NUMA0 = cores 0-17,36-53; NUMA1 = cores 18-35,54-71

**ggml-cuda.cu changes:**
- Added includes for async-pipeline.cuh, nccl-stagger.cuh, numa-gpu-bind.cuh
- Added global: `ggml_cuda_async_pipeline ggml_cuda_async_pipeline_global;`
- Added async init call AFTER `return info;` (BUG — unreachable code)
- Added `ggml_cuda_async_mark_layer_complete()` call in graph evaluation for MUL_MAT and MUL_MAT_ID ops only

**fit.cpp changes:**
- Removed early abort for SPLIT_MODE_TENSOR
- Added post-context-reduction memory verification for tensor-split mode

---

### Commit a3575507e — RoPE sin/cos lookup table for sm_37

**New file:**
- `ggml/src/ggml-cuda/rope-lut.cuh` (100 lines)

**rope-lut.cuh:**
- 4096-entry sin/cos LUT in `__constant__` memory (32KB of 64KB sm_37 limit)
- Bilinear interpolation for ~5 cycle lookup vs 20-30 cycles for sinf/cosf
- Host-side `init_rope_lut()` launches singleton kernel to populate tables
- Device-side `rope_sin_cos()` with fallback to sinf/cosf if not initialized
- `__device__ volatile bool lut_initialized` flag

**ggml-cuda.cu changes:**
- Added `#include "ggml-cuda/rope-lut.cuh"`
- Added RoPE LUT init loop in `ggml_cuda_init()` — calls `cudaSetDevice()` on each device then `rope_lut::init_rope_lut()`
- Added `ggml_cuda_nccl_numa_comms` field to `ggml_backend_cuda_comm_context` (for NUMA sub-communicators)
- Modified `ggml_backend_cuda_comm_allreduce_nccl()` to use NUMA-aware allreduce when `numa_comms.valid`
- Modified `ggml_backend_cuda_comm_init_nccl()` to call `ggml_cuda_nccl_init_env()`, init NUMA comms, pin NCCL threads

**nccl-stagger.cuh changes:**
- Major rewrite: removed `ggml_cuda_nccl_staggered_allreduce()` template
- Added `ggml_cuda_nccl_numa_comms` struct (stub with `valid=false`)
- Added `ggml_cuda_nccl_init_numa_comms()` — returns false because ncclCommSplit unavailable
- Added `ggml_cuda_nccl_numa_allreduce()` — standard NCCL allreduce wrapper (no actual staggering)
- Changed `ggml_cuda_nccl_init_env()` to also set NCCL_DEBUG=VERSION

**rope.cu changes:**
- Replaced `cosf(theta)`/`sinf(theta)` with `rope_lut::rope_sin_cos(theta, sin_raw, cos_raw)` in rope_yarn()

**BUG introduced:** The nccl-stagger.cuh was rewritten to remove the staggered allreduce because ncclCommSplit wasn't available, but the nccl init logic in ggml-cuda.cu was left calling the new NUMA-aware functions. The NUMA comms are always invalid, so it falls back to standard NCCL — functionally OK but misleading.

---

### Commit e94d7ad0e — initialize RoPE LUT per-GPU at backend context creation

**Purpose:** Move RoPE LUT initialization from ggml_cuda_init() (which runs before all backends are ready) to ggml_backend_cuda_context constructor (runs per-device when backend is created).

**common.cuh changes:**
- Added `#include "rope-lut.cuh"` inside CUDA vendor block
- Added RoPE LUT init in `ggml_backend_cuda_context` constructor:
  ```cpp
  explicit ggml_backend_cuda_context(int device) : device(device), ... {
      ggml_cuda_set_device(device);
      rope_lut::init_rope_lut();
  }
  ```

**ggml-cuda.cu changes:**
- Removed `#include "ggml-cuda/rope-lut.cuh"` (now included via common.cuh)
- Removed RoPE LUT init loop from `ggml_cuda_init()`
- Removed `numa_comms` field from `ggml_backend_cuda_comm_context`
- Reverted `ggml_backend_cuda_comm_allreduce_nccl()` to standard NCCL calls (removed NUMA-aware paths)
- Reverted `ggml_backend_cuda_comm_init_nccl()` to original simple NCCL init (removed NUMA init, thread pinning)

**nccl-stagger.cuh changes:**
- Reverted to original staggered allreduce design (restored `ggml_cuda_nccl_staggered_allreduce<T>()`)
- Removed NUMA comms stub structs
- This creates INCONSISTENCY: nccl-stagger.cuh has staggered allreduce code, but ggml-cuda.cu never calls it

**BUG:** This commit reverted most of the NUMA/NCCL changes from a3575507e but left nccl-stagger.cuh with the original staggered code that's never actually wired in. The include chain is now: ggml-cuda.cu includes nccl-stagger.cuh directly, AND common.cuh includes rope-lut.cuh which... doesn't include nccl-stagger.cuh. So nccl-stagger.cuh is included once. But the staggered functions are never called.

---

### Commit 20470e5a7 — docs update + NUMA runtime disable

**Code change in ggml-cpu-numa-replicate.c:**
- Added runtime disable via `GGML_NUMA_REPLICATE=0` env var check at start of `ggml_numa_replicate_init()`
- This is a USEFUL change that should be re-applied after rollback

**Doc changes:** NUMA_REPLICATION_FIX.md, PHASE1_TODO.md, TODO.md, TODO_PHASE4.md, llama_wukong.md updates

---

## Root Cause Hypothesis

The malloc corruption is most likely caused by one of these issues in the async pipeline code:

1. **Device context corruption from lazy init races** — `ggml_cuda_async_mark_layer_complete()` is called during graph evaluation which happens during model load (tensor distribution/initialization). Multiple threads call this concurrently. The lazy init path does `ggml_cuda_set_device(dev_id)` which can corrupt the CUDA context if another thread is using that device.

2. **Dead code misleading us** — The async pipeline enable() call is AFTER `return info;` in ggml_cuda_init(), so the pipeline is never enabled at startup. However, `ggml_cuda_async_mark_layer_complete()` checks `ggml_cuda_async_pipeline_global.enabled` which is false — so it should be a no-op. BUT: the global object construction itself (vector<ggml_cuda_async_gpu_ctx>) could have issues.

3. **Actually — wait.** If `enabled=false` always (because enable() is unreachable), then `ggml_cuda_async_mark_layer_complete()` should just return immediately:
   ```cpp
   inline void ggml_cuda_async_mark_layer_complete(int device, cudaStream_t compute_stream) {
       if (!ggml_cuda_async_pipeline_global.enabled) return;  // <-- always returns
       ...
   }
   ```
   So the async pipeline code is never actually executed. The bug must be elsewhere.

4. **Most likely: RoPE LUT initialization during backend context construction** — `ggml_backend_cuda_context` constructor calls `cudaSetDevice(device)` then launches a kernel. If this happens during NUMA replication initialization (which also uses cudaSetDevice and manages device contexts), there's a race or context corruption. The NUMA replication runs during model load, and backend contexts are created during model load — the ordering may be broken.

5. **fit.cpp tensor-split change** — The new tensor-split memory verification could be causing double-free or use-after-free in the memory allocation path for the 27B model with tensor-split across 8 GPUs.

---

## Future Re-Implementation Notes

When re-attempting async pipeline:

1. **DO NOT use lazy init during model load** — Initialize all GPU contexts at backend init time, before any model loading. Use `cudaSetDevice()` carefully in single-threaded context.

2. **Fix the unreachable enable()** — Move async pipeline initialization BEFORE the `return info;` in ggml_cuda_init().

3. **Test NUMA replication FIRST** — Confirm NUMA replication works alone before adding async pipeline on top.

4. **Isolate RoPE LUT** — Test RoPE LUT independently before combining with async pipeline.

5. **Do NOT call cudaSetDevice() in constructors** — Use explicit init functions called from a controlled single-threaded context.

6. **Consider simpler approach:** Just mark layer completion and don't actually prefetch yet. Get the event recording working without transfer overlap first.

7. **The NCCL stagger code** (from nccl-stagger.cuh original version) is sound and should be re-integrated — but it needs to be wired into ggml_backend_cuda_comm_allreduce_nccl(), which e94d7ad0e removed.

---

## Rollback Verification Steps

After rolling back to 600d5c70c:

1. Clean build directory
2. Build with known-working cmake flags
3. Run model load test with NUMA replication
4. Confirm no malloc corruption

---

## Files Preserved for Reference

- `async_pipeline_plan_PRE_ROLLBACK_20260908.md` — Original async pipeline plan document
- `ASYNC_PIPELINE_ATTEMPTED_CHANGES.md` — This file

---

END ASYNC_PIPELINE_ATTEMPTED_CHANGES.md — UnobligatedRascal
