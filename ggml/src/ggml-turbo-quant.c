/*
 * Turbo3 runtime KV-cache codec.
 *
 * This is the deliberately narrow, correctness-first subset of TurboQuant's
 * PolarQuant implementation. GLM5Next's shared DSA latent is grouped in
 * 128-value WHT blocks; weight formats and the Turbo2/Turbo4 variants are not
 * part of this integration.
 */

#include "ggml-quants.h"
#include "ggml-common.h"
#include "ggml-impl.h"

#include <assert.h>
#include <math.h>
#include <string.h>

static const float turbo3_centroids[8] = {
    -0.190207f, -0.118786f, -0.066822f, -0.021663f,
     0.021663f,  0.066822f,  0.118786f,  0.190207f,
};

static const float turbo3_s1[QK_TURBO3] = {
    -1,1,1,-1,-1,1,-1,1,-1,-1,1,1,1,1,1,1,1,-1,1,-1,1,-1,-1,1,1,1,-1,1,1,-1,-1,-1,
    -1,1,1,-1,1,1,-1,1,-1,1,1,-1,-1,1,-1,1,1,1,1,-1,-1,-1,-1,-1,1,-1,1,1,1,1,-1,1,
    -1,-1,1,-1,-1,-1,1,-1,-1,-1,1,-1,-1,-1,1,1,1,-1,-1,1,1,1,-1,-1,1,1,-1,1,1,-1,1,-1,
    -1,1,1,-1,1,-1,1,-1,1,1,1,1,-1,1,-1,1,1,-1,1,1,-1,-1,-1,-1,-1,1,1,-1,1,1,-1,1,
};

static const float turbo3_s2[QK_TURBO3] = {
    1,1,1,1,-1,1,1,-1,1,-1,-1,-1,1,-1,-1,-1,1,1,-1,-1,1,-1,1,-1,1,-1,-1,1,-1,1,1,1,
    1,1,-1,-1,-1,1,-1,-1,-1,-1,-1,-1,1,1,1,-1,1,-1,1,1,1,-1,-1,1,-1,-1,-1,-1,-1,-1,1,1,
    1,-1,1,-1,-1,-1,-1,1,-1,1,-1,1,-1,-1,1,1,-1,1,-1,1,1,-1,1,-1,-1,-1,-1,1,-1,-1,1,-1,
    1,-1,1,1,1,-1,-1,1,-1,1,-1,1,1,-1,-1,1,-1,1,-1,1,1,-1,1,-1,1,-1,-1,-1,-1,-1,1,-1,
};

static void turbo3_fwht(float * x, const float * signs_in, const float * signs_out) {
    for (int i = 0; i < QK_TURBO3; ++i) {
        x[i] *= signs_in[i];
    }

    for (int h = 1; h < QK_TURBO3; h *= 2) {
        for (int i = 0; i < QK_TURBO3; i += 2*h) {
            for (int j = i; j < i + h; ++j) {
                const float a = x[j];
                const float b = x[j + h];
                x[j]     = a + b;
                x[j + h] = a - b;
            }
        }
    }

    for (int i = 0; i < QK_TURBO3; ++i) {
        x[i] *= 0.08838834764831845f * signs_out[i];
    }
}

void turbo3_cpu_fwht_forward(float * x) {
    turbo3_fwht(x, turbo3_s1, turbo3_s2);
}

void turbo3_cpu_fwht_inverse(float * x) {
    turbo3_fwht(x, turbo3_s2, turbo3_s1);
}

static int turbo3_nearest_centroid(float value) {
    if (value < -0.154496f) return 0;
    if (value < -0.092804f) return 1;
    if (value < -0.044243f) return 2;
    if (value <  0.000000f) return 3;
    if (value <  0.044243f) return 4;
    if (value <  0.092804f) return 5;
    if (value <  0.154496f) return 6;
    return 7;
}

void quantize_row_turbo3_0_ref(
        const float * GGML_RESTRICT x,
        block_turbo3_0 * GGML_RESTRICT y,
        int64_t k) {
    assert(k % QK_TURBO3 == 0);

    const int64_t n_blocks = k / QK_TURBO3;
    for (int64_t block = 0; block < n_blocks; ++block) {
        const float * src = x + block*QK_TURBO3;
        block_turbo3_0 * dst = y + block;
        float rotated[QK_TURBO3];
        float norm_sq = 0.0f;

        for (int i = 0; i < QK_TURBO3; ++i) {
            rotated[i] = src[i];
            norm_sq += src[i]*src[i];
        }

        const float norm = sqrtf(norm_sq);
        const float inv_norm = norm > 1e-10f ? 1.0f/norm : 0.0f;
        for (int i = 0; i < QK_TURBO3; ++i) {
            rotated[i] *= inv_norm;
        }
        turbo3_cpu_fwht_forward(rotated);

        memset(dst->qs, 0, sizeof(dst->qs));
        memset(dst->signs, 0, sizeof(dst->signs));
        float reconstructed_norm_sq = 0.0f;
        for (int i = 0; i < QK_TURBO3; ++i) {
            const int index = turbo3_nearest_centroid(rotated[i]);
            dst->qs[i/4] |= (uint8_t) ((index & 0x3) << ((i % 4)*2));
            if (index & 0x4) {
                dst->signs[i/8] |= (uint8_t) (1u << (i % 8));
            }
            reconstructed_norm_sq += turbo3_centroids[index]*turbo3_centroids[index];
        }

        const float reconstructed_norm = sqrtf(reconstructed_norm_sq);
        const float corrected_norm = reconstructed_norm > 1e-10f ? norm/reconstructed_norm : norm;
        dst->norm = GGML_FP32_TO_FP16(corrected_norm);
    }
}

void dequantize_row_turbo3_0(
        const block_turbo3_0 * GGML_RESTRICT x,
        float * GGML_RESTRICT y,
        int64_t k) {
    assert(k % QK_TURBO3 == 0);

    const int64_t n_blocks = k / QK_TURBO3;
    for (int64_t block = 0; block < n_blocks; ++block) {
        const float norm = GGML_FP16_TO_FP32(x[block].norm);
        for (int i = 0; i < QK_TURBO3; ++i) {
            const uint8_t low = (x[block].qs[i/4] >> ((i % 4)*2)) & 0x3;
            const uint8_t high = (x[block].signs[i/8] >> (i % 8)) & 0x1;
            y[block*QK_TURBO3 + i] = turbo3_centroids[low | (high << 2)]*norm;
        }
    }
}

size_t quantize_turbo3_0(
        const float * GGML_RESTRICT src,
        void * GGML_RESTRICT dst,
        int64_t nrows,
        int64_t n_per_row,
        const float * imatrix) {
    GGML_UNUSED(imatrix);
    assert(n_per_row % QK_TURBO3 == 0);

    const size_t row_size = (n_per_row/QK_TURBO3)*sizeof(block_turbo3_0);
    for (int64_t row = 0; row < nrows; ++row) {
        quantize_row_turbo3_0_ref(
                src + row*n_per_row,
                (block_turbo3_0 *) ((char *) dst + row*row_size),
                n_per_row);
    }
    return nrows*row_size;
}
