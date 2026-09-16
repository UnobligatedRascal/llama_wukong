Plan: Fix CUBLAS_STATUS_ARCH_MISMATCH on Kepler (sm_37) with GGML_SCHED_MAX_COPIES>1 + multi-slot (-np >1)Root CauseError occurs in ggml_cuda_mul_mat_cublas_impl via cublasGemmBatchedEx(..., CUBLAS_GEMM_DEFAULT_TENSOR_OP).
Triggered under concurrent slots because batched path + tensor-op algo selection is used.
Kepler lacks tensor cores / BF16 MMA (requires Ampere sm_80+). CUBLAS_GEMM_DEFAULT_TENSOR_OP or BF16 data/compute path fails with ARCH_MISMATCH.
Current master has partial checks (fast_fp16_hardware_available, bf16_mma_hardware_available in common.cuh) but incomplete enforcement for all paths, especially batched + concurrent.
Old babal35 fork patch (supports_bf16 gate) is obsolete after cuBLAS refactor (PR #24216).
FP16 compute on GK210 is limited (storage/texture only; no fast FP16 arithmetic). Prefer FP32.

Primary Fix (source patches on current master)CMakeLists.txt (ggml/src/ggml-cuda/CMakeLists.txt):Inside if (CUDAToolkit_VERSION VERSION_LESS "13") block, before the existing 50-virtual list:

if (CUDAToolkit_VERSION VERSION_LESS "12")
    list(APPEND CMAKE_CUDA_ARCHITECTURES 35-virtual 37-virtual)
endif()

Build explicitly: -DCMAKE_CUDA_ARCHITECTURES=37 (or 37-real).

ggml-cuda.cu (ggml_cuda_mul_mat_cublas ~L1624+):After env override (GGML_CUDA_CUBLAS_COMPUTE_TYPE) and before switch:

const int cc = ggml_cuda_info().devices[ctx.device].cc;
if (compute_type == GGML_TYPE_BF16 && !bf16_mma_hardware_available(cc)) {
    compute_type = fast_fp16_hardware_available(cc) ? GGML_TYPE_F16 : GGML_TYPE_F32;
}
if (compute_type == GGML_TYPE_F16 && !fast_fp16_hardware_available(cc)) {
    compute_type = GGML_TYPE_F32;
}

In ggml_cuda_mul_mat_cublas_impl (batched/strided/GemmEx paths):Force CUBLAS_GEMM_DEFAULT (not CUBLAS_GEMM_DEFAULT_TENSOR_OP) when cc < GGML_CUDA_CC_VOLTA (or < GGML_CUDA_CC_AMPERE).
Prefer CUBLAS_COMPUTE_32F + CUDA_R_32F for cc < 600.

Ensure quantized → F16 only if fast_fp16_hardware_available; else F32.

common.cuh (already mostly correct):Confirm bf16_mma_hardware_available(cc) = NVIDIA && cc >= 800.
Confirm fast_fp16_hardware_available(cc) excludes Kepler (cc < 600).

Rebuild with user flags + -DGGML_CUDA_GRAPHS=OFF -DGGML_CUDA_FORCE_MMQ=ON (MMQ preferred on Kepler; avoids cuBLAS where possible). Keep -DGGML_SCHED_MAX_COPIES=4.

Runtime Workarounds (no rebuild)export GGML_CUDA_CUBLAS_COMPUTE_TYPE=f32
export CUDA_SCALE_LAUNCH_QUEUES=4x (helps pipeline)
Force -np 1 or limit concurrent slots until patched.
Prefer Q4/Q5/Q8 models; avoid pure BF16 weights.

ValidationBuild succeeds with sm_37.
Single-slot (-np 1) continues to work.
Multi-slot (-np 4, multiple clients) no longer crashes on GEMM.
Measure tokens/s vs. SCHED_MAX_COPIES=1 baseline (expect modest gain from better overlap, higher VRAM use).

Optional Kepler ImprovementsKeep FORCE_MMQ=ON (custom kernels better than cuBLAS on pre-Pascal).
Disable FA (already unsupported).
NCCL remains useful for multi-GPU layer/tensor split.
No viable GK210-specific acceleration beyond existing MMQ + FP32 cuBLAS; texture FP16 is irrelevant for compute.

MaintenanceTrack upstream PRs similar to #25680 (BF16 fallback) and apply if merged.
Re-apply CMake arch list + algo/compute gates after each major cuBLAS refactor.
babal35 fork is stale; maintain local patch set instead.

This restores concurrent multi-slot operation on GK210 while preserving newer-arch performance.

