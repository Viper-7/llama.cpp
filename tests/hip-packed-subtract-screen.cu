// Compile-only EXP-015 probe.  This file intentionally does not include or
// alter ggml HIP helpers, and is never linked or launched.
#include <hip/hip_runtime.h>
#include <stdint.h>

typedef int8_t int8x4_t __attribute__((ext_vector_type(4)));

static __device__ __forceinline__ int current_vsub4(const int a, const int b) {
    const int8x4_t va = reinterpret_cast<const int8x4_t &>(a);
    const int8x4_t vb = reinterpret_cast<const int8x4_t &>(b);
    const int8x4_t c = __builtin_elementwise_sub_sat(va, vb);
    return reinterpret_cast<const int &>(c);
}

// Verbatim PR27962 head a2731fb9743510dd574bf479bc58c53b04cb9548 body.
static __device__ __forceinline__ int candidate_vsub4(const int a, const int b) {
    const unsigned int a_large = a | 0x80808080;
    const unsigned int b_small = b & 0x7f7f7f7f;
    const unsigned int result_low_7bits = a_large - b_small;
    const unsigned int should_flip_high_1bit = (a ^ ~b) & 0x80808080;
    return result_low_7bits ^ should_flip_high_1bit;
}

#if defined(EXP015_CANDIDATE)
#define EXP015_VSUB4 candidate_vsub4
#else
#define EXP015_VSUB4 current_vsub4
#endif

static __device__ __forceinline__ uint32_t expand_sign_nibble(const uint8_t signs) {
    return (((signs & 0x0fu) * 0x00204081u) & 0x01010101u) * 0xffu;
}

// Exact vec_dot_iq3_s_q8_1 (MMVQ) sign/subtract shape at vecdotq.cuh:1222-26.
extern "C" __global__ __attribute__((noinline)) void exp015_iq3s_mmvq(
        const uint32_t * __restrict__ grid, const uint8_t * __restrict__ signs,
        volatile int * __restrict__ out) {
    const int lane = threadIdx.x + blockIdx.x * blockDim.x;
    const uint32_t sign_l = expand_sign_nibble(signs[lane & 15]);
    const uint32_t sign_h = expand_sign_nibble(signs[(lane + 1) & 15] >> 4);
    const int grid_l = EXP015_VSUB4(grid[lane & 511] ^ sign_l, sign_l);
    const int grid_h = EXP015_VSUB4(grid[(lane + 1) & 511] ^ sign_h, sign_h);
    out[2 * lane + 0] = grid_l;
    out[2 * lane + 1] = grid_h;
}

// Exact ggml_cuda_mmq_load_tiles_iq3_s sign/subtract shape at
// mmq-load-tiles.cuh:1400-07.  The grid indexing is intentionally external:
// this probe isolates helper codegen, not table placement or a runtime launch.
extern "C" __global__ __attribute__((noinline)) void exp015_iq3s_mmq(
        const uint32_t * __restrict__ grid, const uint8_t * __restrict__ signs,
        volatile int * __restrict__ out) {
    const int lane = threadIdx.x + blockIdx.x * blockDim.x;
    const uint32_t sign_l = expand_sign_nibble(signs[lane & 15]);
    const uint32_t sign_h = expand_sign_nibble(signs[(lane + 1) & 15] >> 4);
    const int grid_l = EXP015_VSUB4(grid[lane & 511] ^ sign_l, sign_l);
    const int grid_h = EXP015_VSUB4(grid[(lane + 1) & 511] ^ sign_h, sign_h);
    out[2 * lane + 0] = grid_l;
    out[2 * lane + 1] = grid_h;
}
