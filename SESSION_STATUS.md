# Session Status — llama_wukong (9/4/2026 18:00)

## NUMA Replication — P0 COMPLETE

### What's done
- NUMA replicate layer built and tested on NOUGHT (dual-socket Xeon E5-2697 v4, 72c/144t, 2 nodes × 36 CPUs)
- `--numa mirror` flag works end-to-end with llama-server
- Verified: Qwen2.5-0.5B @ `--numa mirror -t 36` → 124 tok/s prompt, 28.6 tok/s gen

### Repos
- NOUGHT HEAD: c2ae57179 (`fix(ggml-cpu-numa-replicate): use correct numa_node_to_cpus bitmask API`)
  - Full buildable repo at `/home/whistler/llama_wukong`
- Local HEAD: ab1d45f (`docs(NUMA): mark P0 complete with implementation details`)
  - NUMA source backed up to `src/ggml-cpu-numa-replicate.{c,h}`
  - Integration notes in `src/NUMA_INTEGRATION_NOTES.md`

### Files (NOUGHT)
- `ggml/src/ggml-cpu/ggml-cpu-numa-replicate.c/.h` — libnuma per-node alloc, thread-node mapping
- `ggml/src/ggml-cpu/ggml-cpu.c` — MIRROR strategy in thread affinity, ggml_numa_replicate_init() call
- `ggml/include/ggml-cpu.h` — GGML_NUMA_STRATEGY_MIRROR = 4
- `common/arg.cpp` — `--numa mirror` option
- `ggml/src/ggml-cpu/CMakeLists.txt` — +numa lib link

### Caveats
- `numa_balancing=1` on NOUGHT degrades NUMA performance → should `sysctl kernel.numa_balancing=0`
- Existing Qwen3.6-27B server on port 4269 running → DO NOT KILL (unless explicitly asked)

### Remaining
- **Immediate:** Disable numa_balancing on NOUGHT
- **P1:** Per-node tensor allocation in ggml buffer type, async pipeline between nodes
- **Benchmark:** Run with production model (Qwen3.6-27B or similar) after disabling numa_balancing

### Mount info
- `D:\PROJECTS` on WHISTLER → `/mnt/whistler_projects` on NOUGHT (SMB)
- Can edit local files directly via NOUGHT using this path
