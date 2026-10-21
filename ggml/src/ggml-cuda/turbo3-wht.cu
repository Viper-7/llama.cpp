#include "turbo3.cuh"
#include "turbo3-wht.cuh"

template <int direction>
static __global__ void k_turbo3_wht_f32(const float * __restrict__ src, float * __restrict__ dst, int64_t n_groups) {
    const int64_t group = blockIdx.x;
    if (group >= n_groups) {
        return;
    }

    const int lane = threadIdx.x;
    __shared__ float values[QK_TURBO3];
    const int64_t offset = group*QK_TURBO3 + lane;

    values[lane] = src[offset] * (direction == 0 ? TURBO3_WHT_SIGNS1[lane] : TURBO3_WHT_SIGNS2[lane]);
    __syncthreads();

#define TURBO3_WHT_STAGE(width) \
    if (lane % (2*(width)) < (width)) { \
        const float a = values[lane]; const float b = values[lane + (width)]; \
        values[lane] = a + b; values[lane + (width)] = a - b; \
    } \
    __syncthreads();
    TURBO3_WHT_STAGE(1)
    TURBO3_WHT_STAGE(2)
    TURBO3_WHT_STAGE(4)
    TURBO3_WHT_STAGE(8)
    TURBO3_WHT_STAGE(16)
    TURBO3_WHT_STAGE(32)
    TURBO3_WHT_STAGE(64)
#undef TURBO3_WHT_STAGE

    dst[offset] = values[lane]*0.08838834764831845f*(direction == 0 ? TURBO3_WHT_SIGNS2[lane] : TURBO3_WHT_SIGNS1[lane]);
}

void ggml_cuda_turbo3_wht(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src = dst->src[0];
    GGML_ASSERT(src->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32);
    GGML_ASSERT(ggml_is_contiguous(src) && ggml_is_contiguous(dst));
    GGML_ASSERT(dst->src[1] == nullptr && "Turbo3 cache WHT does not implement InnerQ scaling");

    int direction;
    int group_size;
    memcpy(&direction, dst->op_params, sizeof(direction));
    memcpy(&group_size, dst->op_params + sizeof(direction), sizeof(group_size));
    GGML_ASSERT((direction == 0 || direction == 1) && group_size == QK_TURBO3);
    GGML_ASSERT(ggml_nelements(src) % QK_TURBO3 == 0);

    const int64_t n_groups = ggml_nelements(src)/QK_TURBO3;
    if (direction == 0) {
        k_turbo3_wht_f32<0><<<n_groups, QK_TURBO3, 0, ctx.stream()>>>((const float *) src->data, (float *) dst->data, n_groups);
    } else {
        k_turbo3_wht_f32<1><<<n_groups, QK_TURBO3, 0, ctx.stream()>>>((const float *) src->data, (float *) dst->data, n_groups);
    }
}
