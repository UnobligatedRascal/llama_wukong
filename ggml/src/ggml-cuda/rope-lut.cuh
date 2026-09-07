#pragma once

// RoPE sin/cos lookup table - "Quake-fast" approach for sm_37 (K80)
// Transcendentals on K80 are 20-30 cycles each; LUT + interpolation is ~5 cycles
// Configurable precision; default 4096 entries = 32KB constant mem (sm_37 limit: 64KB)

#ifndef ROPE_USE_LUT
#define ROPE_USE_LUT 1
#endif

#ifndef ROPE_LUT_SIZE
#define ROPE_LUT_SIZE 4096  // entries; must be power of 2; max ~8192 for sm_37
#endif

#ifndef ROPE_LUT_MASK
#define ROPE_LUT_MASK (ROPE_LUT_SIZE - 1)
#endif

namespace rope_lut {

constexpr int LUT_SIZE = ROPE_LUT_SIZE;
constexpr int LUT_MASK = ROPE_LUT_MASK;

// Constant memory tables (one per GPU, shared by all blocks)
__constant__ float lut_sin[LUT_SIZE];
__constant__ float lut_cos[LUT_SIZE];

// Initialization flag
__device__ volatile bool lut_initialized = false;

// Initialize the lookup table (call once per GPU at startup)
static __global__ void init_rope_lut_kernel() {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        constexpr float TWO_PI = 6.28318530717958647692f;
        constexpr float SCALE = static_cast<float>(LUT_SIZE) / TWO_PI;
        
        for (int i = 0; i < LUT_SIZE; i++) {
            float angle = static_cast<float>(i) / SCALE;
            lut_sin[i] = sinf(angle);
            lut_cos[i] = cosf(angle);
        }
        lut_initialized = true;
    }
    __syncthreads();
}

// Host-side initialization wrapper
inline void init_rope_lut(cudaStream_t stream = 0) {
    init_rope_lut_kernel<<<1, 1, 0, stream>>>();
}

// Normalize angle to [0, 2*pi) using integer arithmetic
__device__ __forceinline__ float normalize_angle(float angle) {
    constexpr float INV_TWO_PI = 0.15915494309189533577f;
    float cycles = angle * INV_TWO_PI;
    float frac = cycles - truncf(cycles);
    return frac * 6.28318530717958647692f;
}

// Bilinear interpolation lookup - ~5 cycles vs 20-30 for sinf/cosf
__device__ __forceinline__ void lookup_sin_cos(float angle, float& out_sin, float& out_cos) {
    float norm = normalize_angle(angle);
    
    // Convert to index range [0, LUT_SIZE)
    float idx_f = norm * static_cast<float>(LUT_SIZE) * 0.15915494309189533577f;
    
    // Clamp to valid range
    if (idx_f < 0.0f) idx_f = 0.0f;
    if (idx_f >= static_cast<float>(LUT_SIZE - 1)) idx_f = static_cast<float>(LUT_SIZE - 2);
    
    int idx = static_cast<int>(idx_f);
    float t = idx_f - idx;
    int idx_next = idx + 1;
    
    // Bilinear interpolation
    float s0 = lut_sin[idx];
    float s1 = lut_sin[idx_next];
    float c0 = lut_cos[idx];
    float c1 = lut_cos[idx_next];
    
    out_sin = s0 + t * (s1 - s0);
    out_cos = c0 + t * (c1 - c0);
}

// Device-side lookup with fallback - use in kernels
__device__ __forceinline__ void rope_sin_cos(float angle, float& out_sin, float& out_cos) {
#if ROPE_USE_LUT
    if (lut_initialized) {
        lookup_sin_cos(angle, out_sin, out_cos);
    } else {
        out_cos = cosf(angle);
        out_sin = sinf(angle);
    }
#else
    out_cos = cosf(angle);
    out_sin = sinf(angle);
#endif
}

} // namespace rope_lut
