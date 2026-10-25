#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <stdio.h>
#include <stdlib.h>

#define CHECK_CUDA(call) do { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { \
        printf("CUDA ERROR at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(1); \
    } \
} while(0)

#define CHECK_CUBLAS(call) do { \
    cublasStatus_t status = (call); \
    if (status != CUBLAS_STATUS_SUCCESS) { \
        printf("CUBLAS ERROR at %s:%d: %d\n", __FILE__, __LINE__, (int)status); \
        exit(1); \
    } \
} while(0)

__global__ void compute_batched_ptrs(
    const void * src0, const void * src1, char * dst,
    const void ** ptrs_src, void ** ptrs_dst,
    int64_t ne12, int64_t ne13, int64_t ne23,
    size_t nb02, size_t nb03, size_t nb12, size_t nb13,
    size_t nbd2, size_t nbd3, int64_t r2, int64_t r3) {
    const int64_t i13 = blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t i12 = blockIdx.y * blockDim.y + threadIdx.y;
    if (i13 >= ne13 || i12 >= ne12) return;
    const int64_t i02 = i12 / r2;
    ptrs_src[0*ne23 + i12 + i13*ne12] = (const char *) src0 + i02*nb02;
    ptrs_src[1*ne23 + i12 + i13*ne12] = (const char *) src1 + i12*nb12;
    ptrs_dst[0*ne23 + i12 + i13*ne12] = (char *) dst + i12*nbd2;
}

void run_test(cublasHandle_t handle, const char *name,
    int64_t ne01, int64_t ne11, int64_t ne00, int64_t ne23,
    const void *alpha, cudaDataType_t dtype_a, cudaDataType_t dtype_b,
    cudaDataType_t dtype_c, cublasComputeType_t compute, cublasGemmAlgo_t algo,
    const void **ptrs_src, void **ptrs_dst, int64_t s01, int64_t s11, int64_t ne0) {
    printf("%s...\n", name);
    cublasStatus_t status = cublasGemmBatchedEx(
        handle, CUBLAS_OP_T, CUBLAS_OP_N,
        ne01, ne11, ne00,
        alpha, (const void**)ptrs_src, dtype_a, s01,
              (const void**)(ptrs_src + ne23), dtype_b, s11,
        alpha, (void**)ptrs_dst, dtype_c, ne0,
        ne23, compute, algo);
    CHECK_CUDA(cudaDeviceSynchronize());
    if (status == CUBLAS_STATUS_SUCCESS) {
        printf("  -> PASSED\n");
    } else {
        const char *msg;
        switch(status) {
            case CUBLAS_STATUS_ARCH_MISMATCH: msg = "ARCH_MISMATCH"; break;
            case CUBLAS_STATUS_INVALID_VALUE: msg = "INVALID_VALUE"; break;
            case CUBLAS_STATUS_EXECUTION_FAILED: msg = "EXECUTION_FAILED"; break;
            default: msg = "OTHER"; break;
        }
        printf("  -> FAILED: %s\n", msg);
    }
}

int main(int argc, char **argv) {
    cudaDeviceProp prop;
    cublasHandle_t handle;
    
    int device_count;
    CHECK_CUDA(cudaGetDeviceCount(&device_count));
    printf("Found %d CUDA device(s)\n", device_count);
    
    int device_id = 0;
    if (argc > 1) device_id = atoi(argv[1]);
    CHECK_CUDA(cudaSetDevice(device_id));
    CHECK_CUDA(cudaGetDeviceProperties(&prop, device_id));
    printf("Device %d: %s, CC %d.%d\n", device_id, prop.name, prop.major, prop.minor);
    
    CHECK_CUBLAS(cublasCreate(&handle));
    
    int64_t ne00 = 4096, ne01 = 2048, ne10 = ne00, ne11 = 2048;
    int64_t ne12 = 4, ne13 = 1, ne23 = 4;
    
    printf("\n=== cublasGemmBatchedEx on Kepler sm_37 ===\n");
    printf("M=%d, N=%d, K=%d, batches=%d\n", (int)ne01, (int)ne11, (int)ne00, (int)ne23);
    
    half *d_A, *d_B, *d_C_f32;
    size_t half_size = ne00 * ne01 * sizeof(half);
    size_t input_size = ne10 * ne11 * sizeof(half);
    size_t output_size = ne01 * ne11 * sizeof(half);
    
    CHECK_CUDA(cudaMalloc(&d_A, half_size * ne23));
    CHECK_CUDA(cudaMalloc(&d_B, input_size * ne23));
    CHECK_CUDA(cudaMalloc(&d_C_f32, output_size * ne23 * 2));
    CHECK_CUDA(cudaMemset(d_A, 0x3F, half_size * ne23));
    CHECK_CUDA(cudaMemset(d_B, 0x3F, input_size * ne23));
    
    const void **d_ptrs_src;
    void **d_ptrs_dst;
    CHECK_CUDA(cudaMalloc(&d_ptrs_src, sizeof(void*) * 2 * ne23));
    CHECK_CUDA(cudaMalloc(&d_ptrs_dst, sizeof(void*) * ne23));
    
    dim3 block(16, 16);
    dim3 grid((ne13 + 15) / 16, (ne12 + 15) / 16);
    size_t s01 = ne00, s02 = ne00 * ne01 * sizeof(half);
    size_t s11 = ne10, s12 = ne10 * ne11 * sizeof(half);
    
    // F16 output pointers
    compute_batched_ptrs<<<grid, block>>>(
        d_A, d_B, (char*)d_A, d_ptrs_src, d_ptrs_dst,
        ne12, ne13, ne23, s02, 0, s12, 0, ne01 * ne11 * sizeof(half), 0, 1, 1);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());
    
    const half alpha_half = __float2half(1.0f);
    const float alpha_f32 = 1.0f;
    
    run_test(handle, "TEST 1: F16+F16->F16, COMPUTE_16F, GEMM_DEFAULT",
        ne01, ne11, ne00, ne23, &alpha_half,
        CUDA_R_16F, CUDA_R_16F, CUDA_R_16F,
        CUBLAS_COMPUTE_16F, CUBLAS_GEMM_DEFAULT,
        d_ptrs_src, d_ptrs_dst, s01, s11, ne01);
    
    // F32 output pointers
    compute_batched_ptrs<<<grid, block>>>(
        d_A, d_B, (char*)d_C_f32, d_ptrs_src, d_ptrs_dst,
        ne12, ne13, ne23, s02, 0, s12, 0, ne01 * ne11 * sizeof(float), 0, 1, 1);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());
    
    run_test(handle, "TEST 2: F16+F16->F32, COMPUTE_16F, GEMM_DEFAULT",
        ne01, ne11, ne00, ne23, &alpha_half,
        CUDA_R_16F, CUDA_R_16F, CUDA_R_32F,
        CUBLAS_COMPUTE_16F, CUBLAS_GEMM_DEFAULT,
        d_ptrs_src, d_ptrs_dst, s01, s11, ne01);
    
    run_test(handle, "TEST 3: F16+F16->F32, COMPUTE_32F, GEMM_DEFAULT",
        ne01, ne11, ne00, ne23, &alpha_f32,
        CUDA_R_16F, CUDA_R_16F, CUDA_R_32F,
        CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT,
        d_ptrs_src, d_ptrs_dst, s01, s11, ne01);
    
    run_test(handle, "TEST 4: F16+F16->F16, COMPUTE_32F, GEMM_DEFAULT",
        ne01, ne11, ne00, ne23, &alpha_f32,
        CUDA_R_16F, CUDA_R_16F, CUDA_R_16F,
        CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT,
        d_ptrs_src, d_ptrs_dst, s01, s11, ne01);
    
    run_test(handle, "TEST 5: F16+F16->F32, COMPUTE_32F_FAST_16F",
        ne01, ne11, ne00, ne23, &alpha_f32,
        CUDA_R_16F, CUDA_R_16F, CUDA_R_32F,
        CUBLAS_COMPUTE_32F_FAST_16F, CUBLAS_GEMM_DEFAULT,
        d_ptrs_src, d_ptrs_dst, s01, s11, ne01);
    
    run_test(handle, "TEST 6: F16+F16->F16, COMPUTE_16F, TENSOR_OP",
        ne01, ne11, ne00, ne23, &alpha_half,
        CUDA_R_16F, CUDA_R_16F, CUDA_R_16F,
        CUBLAS_COMPUTE_16F, CUBLAS_GEMM_DEFAULT_TENSOR_OP,
        d_ptrs_src, d_ptrs_dst, s01, s11, ne01);
    
    cublasDestroy(handle);
    
    printf("\n=== RESULT ===\n");
    printf("Kepler sm_37: CUBLAS_COMPUTE_16F triggers ARCH_MISMATCH internally.\n");
    printf("FIX: Always use CUBLAS_COMPUTE_32F for Kepler.\n");
    return 0;
}
