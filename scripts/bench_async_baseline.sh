#!/bin/bash
# Profile current baseline: compute/transfer patterns, NCCL sync, GPU utilization
# Run BEFORE any async pipeline changes
# Saves outputs to /home/whistler/llama_wukong/benches/nought-baseline/

set -e

OUTDIR="/home/whistler/llama_wukong/benches/nought-baseline"
mkdir -p "$OUTDIR"
echo "=== Baseline Profile ==="
echo "Output: $OUTDIR"

MODEL="/mnt/512gb_ssd/models/Qwen3.6-27B-Fable-Fus-711-UnHeretic-NM-DAU-NEO-MAX-NEO-MTP-Q4_K_M.gguf"
MMPROJ="/mnt/512gb_ssd/models/Qwen3.6-27B-mmproj-F16.gguf"
SERVER="/home/whistler/llama_wukong/build/bin/llama-server"
BENCH="/home/whistler/llama_wukong/build/bin/llama-bench"

echo ""
echo "=== Step 1: llama-bench baseline (8-GPU tensor split, 3 slots) ==="
sudo GGML_CUDA_P2P=1 -E nice -n -20 numactl --interleave=all "$BENCH" \
    -m "$MODEL" \
    -t 36 \
    -ngl 99 \
    --tensor-split 1,1,1,1,1,1,1,1 \
    --split-mode tensor \
    --cache-type-k q4_0 \
    --cache-type-v q4_0 \
    -c 262144 \
    -b 512 \
    -u 128 \
    -n 256 \
    -p 128 \
    --ctx-checkpoints 64 \
    --checkpoint-min-step 4096 \
    -np 3 \
    --mmproj "$MMPROJ" \
    --no-mmproj-offload \
    --spec-type draft-mtp \
    --spec-draft-p-min 0.75 \
    --spec-draft-n-max 3 \
    2>&1 | tee "$OUTDIR/llama-bench-baseline.txt"

echo ""
echo "=== Step 2: llama-bench with q2_K cache (memory pressure test) ==="
sudo GGML_CUDA_P2P=1 -E nice -n -20 numactl --interleave=all "$BENCH" \
    -m "$MODEL" \
    -t 36 \
    -ngl 99 \
    --tensor-split 1,1,1,1,1,1,1,1 \
    --split-mode tensor \
    --cache-type-k q2_K \
    --cache-type-v q2_K \
    -c 262144 \
    -b 512 \
    -u 128 \
    -n 256 \
    -p 128 \
    -np 3 \
    --mmproj "$MMPROJ" \
    --no-mmproj-offload \
    2>&1 | tee "$OUTDIR/llama-bench-q2K-cache.txt"

echo ""
echo "=== Step 3: NUMA-aware NUMA0-only test (8 GPUs, threads bound to NUMA0) ==="
sudo GGML_CUDA_P2P=1 -E nice -n -20 numactl --cpunodebind=0 --membind=0 "$BENCH" \
    -m "$MODEL" \
    -t 18 \
    -ngl 99 \
    --tensor-split 1,1,1,1,1,1,1,1 \
    --split-mode tensor \
    --cache-type-k q4_0 \
    --cache-type-v q4_0 \
    -c 262144 \
    -b 512 \
    -u 128 \
    -n 256 \
    -p 128 \
    -np 3 \
    --mmproj "$MMPROJ" \
    --no-mmproj-offload \
    2>&1 | tee "$OUTDIR/llama-bench-numa0-only.txt"

echo ""
echo "=== Done. Check $OUTDIR for results ==="
