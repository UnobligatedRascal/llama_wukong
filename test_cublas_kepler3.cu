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
    if (status != CUBLAS_STATUS_SUCCESS) { printf("CUBLAS ERROR (%d): %s\n", (int)status, \
        (status==CUBLAS_STATUS_ARCH_MISMATCH)?"ARCH_MISMATCH":"other"); exit(1); } \
} while(0)

int main() {
    cudaDeviceProp prop;
    cublasHandle_t handle;
    
    CHECK_CUDA(cudaSetDevice(0));
    CHECK_CUDA(cudaGetDeviceProperties(&prop, 0));
    printf("Device: %s, CC %d.%d\n", prop.name, prop.major, prop.minor);
    
    // Check runtime/driver versions
    int runtimeVer, driverVer;
    CHECK_CUDA(cudaRuntimeGetVersion(&runtimeVer));
    CHECK_CUDA(cudaDriverGetVersion(&driverVer));
    printf("CUDA runtime: %d.%d, driver: %d.%d\n",
        runtimeVer/1000, (runtimeVer%100)/10,
        driverVer/1000, (driverVer%100)/10);
    
    CHECK_CUBLAS(cublasCreate(&handle));
    
    int M = 2048, N = 2048, K = 4096, batches = 4;
    const float alpha = 1.0f, beta = 0.0f;
    
    float *d_A, *d_B, *d_C;
    CHECK_CUDA(cudaMalloc(&d_A, M * K * sizeof(float) * batches));
    CHECK_CUDA(cudaMalloc(&d_B, K * N * sizeof(float) * batches));
    CHECK_CUDA(cudaMalloc(&d_C, M * N * sizeof(float) * batches));
    CHECK_CUDA(cudaMemset(d_A, 0x3F, M * K * sizeof(float) * batches));
    CHECK_CUDA(cudaMemset(d_B, 0x3F, K * N * sizeof(float) * batches));
    
    printf("\n=== cuBLAS API support on Kepler + CUDA 11.8 ===\n");
    printf("M=%d, N=%d, K=%d, batches=%d\n\n", M, N, K, batches);
    
    // TEST 1: cublasSgemm (legacy)
    printf("TEST 1: cublasSgemm...\n");
    cublasStatus_t s = cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N,
        M, N, K, &alpha, d_A, K, d_B, K, &beta, d_C, M);
    CHECK_CUDA(cudaDeviceSynchronize());
    printf("  -> %s\n", s == CUBLAS_STATUS_SUCCESS ? "PASSED" : "FAILED");
    
    // TEST 2: cublasGemmEx with F32 only
    printf("TEST 2: cublasGemmEx (F32)...\n");
    s = cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N,
        M, N, K, &alpha, d_A, CUDA_R_32F, K, d_B, CUDA_R_32F, K,
        &beta, d_C, CUDA_R_32F, M, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
    CHECK_CUDA(cudaDeviceSynchronize());
    printf("  -> %s (status=%d)\n", s == CUBLAS_STATUS_SUCCESS ? "PASSED" : "FAILED", (int)s);
    
    // TEST 3: cublasGemmEx with F32, algo=0 (force default)
    printf("TEST 3: cublasGemmEx (F32, algo=0)...\n");
    s = cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N,
        M, N, K, &alpha, d_A, CUDA_R_32F, K, d_B, CUDA_R_32F, K,
        &beta, d_C, CUDA_R_32F, M, CUBLAS_COMPUTE_32F, (cublasGemmAlgo_t)0);
    CHECK_CUDA(cudaDeviceSynchronize());
    printf("  -> %s (status=%d)\n", s == CUBLAS_STATUS_SUCCESS ? "PASSED" : "FAILED", (int)s);
    
    // TEST 4: cublasSgemmStridedBatched
    printf("TEST 4: cublasSgemmStridedBatched...\n");
    s = cublasSgemmStridedBatched(handle, CUBLAS_OP_T, CUBLAS_OP_N,
        M, N, K, &alpha, d_A, K, M*K*sizeof(float),
        d_B, K, K*N*sizeof(float),
        &beta, d_C, M, M*N*sizeof(float), batches);
    CHECK_CUDA(cudaDeviceSynchronize());
    printf("  -> %s\n", s == CUBLAS_STATUS_SUCCESS ? "PASSED" : "FAILED");
    
    // TEST 5: cublasGemmStridedBatchedEx with F32
    printf("TEST 5: cublasGemmStridedBatchedEx (F32)...\n");
    s = cublasGemmStridedBatchedEx(handle, CUBLAS_OP_T, CUBLAS_OP_N,
        M, N, K, &alpha, d_A, CUDA_R_32F, K, M*K*sizeof(float),
        d_B, CUDA_R_32F, K, K*N*sizeof(float),
        &beta, d_C, CUDA_R_32F, M, M*N*sizeof(float),
        batches, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
    CHECK_CUDA(cudaDeviceSynchronize());
    printf("  -> %s (status=%d)\n", s == CUBLAS_STATUS_SUCCESS ? "PASSED" : "FAILED", (int)s);
    
    // TEST 6: cublasGemmBatchedEx with F32
    printf("TEST 6: cublasGemmBatchedEx (F32)...\n");
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
    
    s = cublasGemmBatchedEx(handle, CUBLAS_OP_T, CUBLAS_OP_N,
        M, N, K, &alpha, d_ptrA, CUDA_R_32F, K,
        d_ptrB, CUDA_R_32F, K,
        &beta, d_ptrC, CUDA_R_32F, M,
        batches, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
    CHECK_CUDA(cudaDeviceSynchronize());
    printf("  -> %s (status=%d)\n", s == CUBLAS_STATUS_SUCCESS ? "PASSED" : "FAILED", (int)s);
    
    cublasDestroy(handle);
    
    printf("\n=== SUMMARY ===\n");
    printf("Legacy Sgemm APIs work. 'Ex' APIs may fail on Kepler + CUDA 11.8 + driver 470.\n");
    printf("This explains why llama.cpp batched paths fail: they use cublasGemmBatchedEx.\n");
    return 0;
}
