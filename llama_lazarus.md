<llama_lazarus>
<local_system>
GPUs: 8 X GK210(Kepler,sm_37,driver_470,CUDA_11.4,CUDA_11.8_Toolkit,NCCL)
CPUs: 2 X Xeon-E5-2697-v4(iommu disabled for stability)
RAM: 128GB ECC/DDR-4
OS: Debian Bookworm > Q4os
</local_system>

<working_software_stack>

llama.cpp="""https://github.com/ggml-org/llama.cpp"""

llama.cpp build script (tuned for local_system)="""git clone https://github.com/ggml-org/llama.cpp && cd llama.cpp && cmake -B build -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON -DGGML_SCHED_MAX_COPIES=1 -DGGML_CUDA_F16=ON -DGGML_CUDA_PEER_MAX_BATCH_SIZE=64 -DCMAKE_CUDA_HOST_COMPILER=g++-11 -DCMAKE_CUDA_COMPILER=/usr/local/cuda-11.8/bin/nvcc -DGGML_CUDA_NCCL=ON -DCMAKE_CUDA_ARCHITECTURES=37 -DLLAMA_CURL=OFF -DGGML_CUDA_FA_ALL_QUANTS=ON -DGGML_CUDA_FORCE_MMQ=ON -DCUDA_COMPILER=/usr/local/cuda-11.8/bin/nvcc -DGGML_CUDA_ARCHITECTURES="37" -DGGML_CUDA_GRAPHS=OFF -DCMAKE_C_COMPILER=gcc-11 -DCMAKE_CXX_COMPILER=g++-11 && cmake --build build --config Release -j36"""

All requirements of the software stack are met, build works, llama.cpp/llama-server works
</working_software_stack>

<buggy>
When adjusting the "-DGGML_SCHED_MAX_COPIES" to "4" instead of "1", the build will still complete successfully. The model runs great *when "-np 1" is set* (llama-server) BUT will crash with "CUBLAS_STATUS_ARCH_MISMATCH + CUBLAS_GEMM_DEFAULT_TENSOR_OP" error *when "-np 4" is set* and more than one user attempts to occupy available model slots
</buggy>

<CUDA_ERROR>
"""
CUDA error: CUBLAS_STATUS_ARCH_MISMATCH
4.17.319.115 E   current device: 0, in function ggml_cuda_mul_mat_cublas_impl at /home/whistler/llama.cpp/ggml/src/ggml-cuda/ggml-cuda.cu:1606
4.17.319.122 E   cublasGemmBatchedEx(cublas_h, CUBLAS_OP_T, CUBLAS_OP_N, ne01, ne11, ne10, alpha, (const void **) (ptrs_src.get() + 0*ne23), cu_data_type_a, s01, (const void **) (ptrs_src.get() + 1*ne23), cu_data_type_b, s11, beta, ( void **) (ptrs_dst.get() + 0*ne23), cu_data_type, ne0, ne23, cu_compute_type, CUBLAS_GEMM_DEFAULT_TENSOR_OP)
"""
</CUDA_ERROR>

<FIX>
Force build process to disable the unsupported GEMM/BF16/cuBLAS path for any detected pre-Ampere (pre-sm_80) card by modifying files within "llama.cpp/ggml/src/ggml-cuda" directory (such as "CMakeLists.txt", "ggml-cuda.cu" or any others that may cause fallback to an incompatible path). Research possibility of replacing any defunct GEMM/BF16/cuBLAS features/paths with FP32 or FP16 paths. (NOTE: FP16 support on GK210 is limited to texture processing and storage, avoid FP16 for any compute-heavy functions)
Last chance: If *traditional* means of employing these features don't exist, research potential of utilizing any of the GK210/K80 hardware/software features to improve the local llama.cpp functions or user experience.
</FIX>
<known_work>
User "babal35" on GitHub created a fork with explicit support for sm_37/Kepler GPUs: "https://github.com/babal35/llamacpp-kepler"
BUT this fork is months-old and newer llama.cpp builds are CRITICAL.
2 file modifications were noted:
<file1>
"llama.cpp/ggml/src/ggml-cuda/CMakeLists.txt"
</file1>
<file1_changes>
 """ if (CUDAToolkit_VERSION VERSION_LESS "13")
+    # 35 == Tesla K40/K20 (Kepler GK110), 37 == Tesla K80 (Kepler GK210)
+    # Both are last supported in CUDA 11.x
+    if (CUDAToolkit_VERSION VERSION_LESS "12")
+        list(APPEND CMAKE_CUDA_ARCHITECTURES 35-virtual 37-virtual)
+    endif()
     list(APPEND CMAKE_CUDA_ARCHITECTURES 50-virtual 61-virtual 70-virtual)
 endif ()"""
</file1_changes>
 <file2>
"llama.cpp/ggml/src/ggml-cuda/ggml-cuda.cu"
<file2_changes>
"""-const bool supports_bf16 = GGML_CUDA_CC_IS_NVIDIA(cc) || GGML_CUDA_CC_IS_AMD(cc) ||
+// BF16 cuBLAS GemmEx requires Ampere (cc >= 800); Kepler/Maxwell/Pascal/Volta do not support it.
+const bool supports_bf16 = (GGML_CUDA_CC_IS_NVIDIA(cc) && cc >= GGML_CUDA_CC_AMPERE) || GGML_CUDA_CC_IS_AMD(cc) ||
     (GGML_CUDA_CC_IS_MTHREADS(cc) && cc >= GGML_CUDA_CC_QY2);"""
</file2_changes>

Changes made to llama.cpp since babal35's fork have substantially altered files, "ggml-cuda.cu" file is especially different now and the fix cannot be applied as documented.

</known_work>
</llama_lazarus>
