// Minimal Turbo3 KV-cache device helpers.  This deliberately excludes Turbo2/4,
// InnerQ, and all weight-quantization machinery.
#pragma once

#include "common.cuh"

#define QR_TURBO3 1

static __constant__ float TURBO3_CENTROIDS[8] = {
    -0.190207f, -0.118786f, -0.066822f, -0.021663f,
     0.021663f,  0.066822f,  0.118786f,  0.190207f,
};

static __constant__ float TURBO3_MIDPOINTS[7] = {
    -0.154496f, -0.092804f, -0.044243f, 0.0f,
     0.044243f,  0.092804f,  0.154496f,
};

// Signed, normalized 128-wide Walsh-Hadamard rotation.  These match the CPU
// Turbo3 reference in ggml-turbo-quant.c.
static __constant__ float TURBO3_WHT_SIGNS1[128] = {
    -1,1,1,-1,-1,1,-1,1,-1,-1,1,1,1,1,1,1, 1,-1,1,-1,1,-1,-1,1,1,1,-1,1,1,-1,-1,-1,
    -1,1,1,-1,1,1,-1,1,-1,1,1,-1,-1,1,-1,1, 1,1,1,-1,-1,-1,-1,-1,1,-1,1,1,1,1,-1,1,
    -1,-1,1,-1,-1,-1,1,-1,-1,-1,1,-1,-1,-1,1,1, 1,-1,-1,1,1,1,-1,-1,1,1,-1,1,1,-1,1,-1,
    -1,1,1,-1,1,-1,1,-1,1,1,1,1,-1,1,-1,1, 1,-1,1,1,-1,-1,-1,-1,-1,1,1,-1,1,1,-1,1,
};

static __constant__ float TURBO3_WHT_SIGNS2[128] = {
    1,1,1,1,-1,1,1,-1,1,-1,-1,-1,1,-1,-1,-1, 1,1,-1,-1,1,-1,1,-1,1,-1,-1,1,-1,1,1,1,
    1,1,-1,-1,-1,1,-1,-1,-1,-1,-1,-1,1,1,1,-1, 1,-1,1,1,1,-1,-1,1,-1,-1,-1,-1,-1,-1,1,1,
    1,-1,1,-1,-1,-1,-1,1,-1,1,-1,1,-1,-1,1,1, -1,1,-1,1,1,-1,1,-1,-1,-1,-1,1,-1,-1,1,-1,
    1,-1,1,1,1,-1,-1,1,-1,1,-1,1,1,-1,-1,1, -1,1,-1,1,1,-1,1,-1,1,-1,-1,-1,-1,-1,1,-1,
};

static __device__ __forceinline__ uint8_t turbo3_nearest_centroid(float value) {
    if      (value < TURBO3_MIDPOINTS[0]) return 0;
    else if (value < TURBO3_MIDPOINTS[1]) return 1;
    else if (value < TURBO3_MIDPOINTS[2]) return 2;
    else if (value < TURBO3_MIDPOINTS[3]) return 3;
    else if (value < TURBO3_MIDPOINTS[4]) return 4;
    else if (value < TURBO3_MIDPOINTS[5]) return 5;
    else if (value < TURBO3_MIDPOINTS[6]) return 6;
    return 7;
}

static __device__ __forceinline__ float turbo3_dequant_element(
        const block_turbo3_0 * __restrict__ block, int index, float norm) {
    const uint8_t low2 = (block->qs[index/4] >> ((index % 4)*2)) & 0x3;
    const uint8_t high1 = (block->signs[index/8] >> (index % 8)) & 0x1;
    return TURBO3_CENTROIDS[low2 | (high1 << 2)] * norm;
}
