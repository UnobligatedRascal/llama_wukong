# NUMA Replication Integration Notes

## Files
- `ggml-cpu-numa-replicate.c/.h` — Per-node allocation, thread-node mapping (backed up here)
- `ggml/src/ggml-cpu/ggml-cpu.c` — MIRROR strategy, replicate_init() call
- `ggml/include/ggml-cpu.h` — GGML_NUMA_STRATEGY_MIRROR = 4
- `common/arg.cpp` — `--numa mirror` option

## ggml-cpu.h enum addition
```c
enum ggml_numa_strategy {
    GGML_NUMA_STRATEGY_DISABLED   = 0,
    GGML_NUMA_STRATEGY_DISTRIBUTE = 1,
    GGML_NUMA_STRATEGY_ISOLATE    = 2,
    GGML_NUMA_STRATEGY_NUMACTL    = 3,
    GGML_NUMA_STRATEGY_MIRROR     = 4,  // ← added
};
```

## ggml-cpu.c changes

### Include
```c
#include "ggml-cpu-numa-replicate.h"
```

### NUMA strategy switch (around line 2192)
```c
case GGML_NUMA_STRATEGY_MIRROR:
    // MIRROR: distribute threads and enable weight replication
    node_num = thread_n % g_state.numa.n_nodes;
    break;
```

### Init call in ggml_cpu_init() (around line 3939)
```c
/* Initialize NUMA replication (if supported) */
ggml_numa_replicate_init();
```

## arg.cpp --numa option (line ~2757)
```cpp
else if (value == "mirror") { params.numa = GGML_NUMA_STRATEGY_MIRROR; }
```

## CMakeLists.txt (ggml/src/ggml-cpu/)
- Add to GGML_CPU_SOURCES: `ggml-cpu/ggml-cpu-numa-replicate.c`
- Link libnuma on UNIX:
```cmake
if(UNIX)
    find_library(NUMA_LIB numa)
    if(NUMA_LIB)
        target_link_libraries(${GGML_CPU_NAME} PRIVATE ${NUMA_LIB})
    endif()
endif()
```

## Verified output on NOUGHT (dual-socket Xeon, 72c/144t)
```
ggml_numa_replicate: node 0 has 36 CPUs
ggml_numa_replicate: node 1 has 36 CPUs
ggml_numa_replicate: ENABLED across 2 nodes, main process on node 0
```

Test: Qwen2.5-0.5B @ `--numa mirror -t 36` → 124 tok/s prompt, 28.6 tok/s gen

## Caveat
`/proc/sys/kernel/numa_balancing=1` on NOUGHT degrades NUMA performance.
Fix: `sysctl kernel.numa_balancing=0`
