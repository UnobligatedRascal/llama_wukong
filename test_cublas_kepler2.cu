#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <stdio.h>
#include <stdlib.h>

#define CHECK_CUDA(call) do { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { printf("CUDA ERROR: %s\n", cudaGetErrorString(err)); exit(1); } \
} while(0)

#define CHECK_CUBLAS(call) do { \
    cublasStatus_t status = (call); \
    if (status != CUBLAS_STATUS_SUCCESS) { printf("CUBLAS ERROR: %d\n", (int)status); exit(1); } \
} while(0)

int main() {
    cudaDeviceProp prop;
    cublasHandle_t handle;
    
    CHECK_CUDA(cudaSetDevice(0));
    CHECK_CUDA(cudaGetDeviceProperties(&prop, 0));
    printf("Device: %s, CC %d.%d\n", prop.name, prop.major, prop.minor);
    CHECK_CUBLAS(cublasCreate(&handle));
    
    int M = 2048, N = 2048, K = 4096, batches = 4;
    const float alpha = 1.0f, beta = 0.0f;
    
    float *d_A, *d_B, *d_C;
    CHECK_CUDA(cudaMalloc(&d_A, M * K * sizeof(float) * batches));
    CHECK_CUDA(cudaMalloc(&d_B, K * N * sizeof(float) * batches));
    CHECK_CUDA(cudaMalloc(&d_C, M * N * sizeof(float) * batches));
    CHECK_CUDA(cudaMemset(d_A, 0x3F, M * K * sizeof(float) * batches));
    CHECK_CUDA(cudaMemset(d_B, 0x3F, K * N * sizeof(float) * batches));
    
    printf("\n=== Testing cuBLAS on Kepler ===\n");
    printf("M=%d, N=%d, K=%d, batches=%d\n", M, N, K, batches);
    
    // TEST 1: F32 Sgemm (non-batched)
    printf("TEST 1: cublasSgemm...\n");
    CHECK_CUBLAS(cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N,
        M, N, K, &alpha, d_A, K, d_B, K, &beta, d_C, M));
    CHECK_CUDA(cudaDeviceSynchronize());
    printf("  -> PASSED\n");
    
    // TEST 2: F32 via GemmEx
    printf("TEST 2: cublasGemmEx...\n");
    CHECK_CUBLAS(cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N,
        M, N, K, &alpha, d_A, CUDA_R_32F, K, d_B, CUDA_R_32F, K,
        &beta, d_C, CUDA_R_32F, M, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
    CHECK_CUDA(cudaDeviceSynchronize());
    printf("  -> PASSED\n");
    
    // TEST 3: F32 via GemmStridedBatchedEx (same as batched but strided)
    printf("TEST 3: cublasGemmStridedBatchedEx...\n");
    CHECK_CUBLAS(cublasGemmStridedBatchedEx(handle, CUBLAS_OP_T, CUBLAS_OP_N,
        M, N, K, &alpha, d_A, CUDA_R_32F, K, M*K*sizeof(float),
        d_B, CUDA_R_32F, K, K*N*sizeof(float),
        &beta, d_C, CUDA_R_32F, M, M*N*sizeof(float),
        batches, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
    CHECK_CUDA(cudaDeviceSynchronize());
    printf("  -> PASSED\n");
    
    // TEST 4: F32 via GemmBatchedEx with device pointer arrays
    printf("TEST 4: cublasGemmBatchedEx...\n");
    float *h_ptrsA[batches], *h_ptrsB[batches], *h_ptrsC[batches];
    for (int i = 0; i < batches; i++) {
        h_ptrsA[i] = d_A + i * M * K;
        h_ptrsB[i] = d_B + i * K * N;
        h_ptrsC[i] = d_C + i * M * N;
    }
    void **d_ptrA, **d_ptrB, **d_ptrC;
    CHECK_CUDA(cudaMalloc(&d_ptrA, sizeof(void*) * batches));
    CHECK_CUDA(cudaMalloc(&d_ptrB, sizeof(void*) * batches));
    CHECK_CUDA(cudaMalloc(&d_ptrC, sizeof(void*) * batches));
    CHECK_CUDA(cudaMemcpy(d_ptrA, h_ptrsA, sizeof(void*) * batches, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_ptrB, h_ptrsB, sizeof(void*) * batches, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_ptrC, h_ptrsC, sizeof(void*) * batches, cudaMemcpyHostToDevice));
    
    CHECK_CUBLAS(cublasGemmBatchedEx(handle, CUBLAS_OP_T, CUBLAS_OP_N,
        M, N, K, &alpha, d_ptrA, CUDA_R_32F, K,
        d_ptrB, CUDA_R_32F, K,
        &beta, d_ptrC, CUDA_R_32F, M,
        batches, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
    CHECK_CUDA(cudaDeviceSynchronize());
    printf("  -> PASSED\n");
    
    cublasDestroy(handle);
    printf("\n=== ALL F32 TESTS PASSED ===\n");
    printf("Kepler sm_37: F32 data + F32 compute works in all batched GEMM variants.\n");
    return 0;
}
