#!/bin/bash
# Preflight check for async pipeline tests
# Ensures environment is ready before running benchmarks

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="${SCRIPT_DIR}/build"

ERRORS=0

check() {
    local desc="$1"
    local cmd="$2"
    if eval "${cmd}" > /dev/null 2>&1; then
        echo "[OK] ${desc}"
    else
        echo "[FAIL] ${desc}"
        ((ERRORS++))
    fi
}

echo "=== PREFLIGHT CHECK ==="
echo "Timestamp: $(date)"
echo ""

# Build artifacts
check "llama-bench exists"   "test -x ${BUILD_DIR}/bin/llama-bench"
check "llama-server exists"  "test -x ${BUILD_DIR}/bin/llama-server"
check "libggml-cuda exists"  "test -f ${BUILD_DIR}/bin/libggml-cuda.so"

# Model
MODEL="/mnt/512gb_ssd/models/Qwen3.6-27B-Fable-Fus-711-UnHeretic-NM-DAU-NEO-MAX-NEO-MTP-Q4_K_M.gguf"
check "Test model exists"    "test -f ${MODEL}"

# CUDA/NVIDIA
check "nvidia-smi works"     "nvidia-smi --query-gpu=count --format=csv,noheader"

# Check all 8 GPUs visible
GPU_COUNT=$(nvidia-smi --query-gpu=count --format=csv,noheader 2>/dev/null | head -1 | tr -d '[:space:]')
if [[ "${GPU_COUNT}" == "8" ]]; then
    echo "[OK] All 8 GPUs visible"
else
    echo "[FAIL] Expected 8 GPUs, found ${GPU_COUNT:-unknown}"
    ((ERRORS++))
fi

# GPU memory check
echo ""
echo "GPU inventory:"
nvidia-smi --query-gpu=index,name,memory.total,memory.used --format=csv 2>/dev/null || echo "[FAIL] nvidia-smi query failed"

# Check for active processes on GPUs
echo ""
echo "Active GPU processes:"
nvidia-smi --query-compute-apps=pid,name,used_memory --format=csv 2>/dev/null || echo "No active compute apps"

# NCCL library
check "NCCL library found"   "test -f /usr/lib/x86_64-linux-gnu/libnccl.so"

# numactl
check "numactl available"    "which numactl"

# Async pipeline headers present
check "async-pipeline.cuh"   "test -f ${SCRIPT_DIR}/ggml/src/ggml-cuda/async-pipeline.cuh"
check "nccl-stagger.cuh"     "test -f ${SCRIPT_DIR}/ggml/src/ggml-cuda/nccl-stagger.cuh"
check "numa-gpu-bind.cuh"    "test -f ${SCRIPT_DIR}/ggml/src/ggml-cuda/numa-gpu-bind.cuh"

echo ""
echo "=== PREFLIGHT SUMMARY ==="
if [[ ${ERRORS} -eq 0 ]]; then
    echo "ALL CHECKS PASSED - Ready to test"
    exit 0
else
    echo "${ERRORS} CHECK(S) FAILED - Fix before running tests"
    exit 1
fi
