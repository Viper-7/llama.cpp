#include "conv1d-grouped.cuh"

__global__ void conv1d_grouped_f32_kernel(
        const float * __restrict__ kernel,
        const float * __restrict__ input,
        float * __restrict__ output,
        const int kw,
        const int ic_g,
        const int oc,
        const int iw,
        const int ic,
        const int n_batches,
        const int ow,
        const int stride,
        const int padding,
        const int dilation,
        const int groups) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = n_batches * oc * ow;

    if (idx >= total) {
        return;
    }

    const int out_x = idx % ow;
    const int out_c = (idx / ow) % oc;
    const int batch = idx / (ow * oc);
    const int oc_g = oc / groups;
    const int group = out_c / oc_g;

    float sum = 0.0f;

    for (int kx = 0; kx < kw; ++kx) {
        const int in_x = out_x * stride + kx * dilation - padding;
        if (in_x < 0 || in_x >= iw) {
            continue;
        }

        const float * x = input + batch * (ic * iw) + group * ic_g * iw + in_x;
        const float * w = kernel + out_c * (ic_g * kw) + kx;

        for (int in_c = 0; in_c < ic_g; ++in_c) {
            sum += w[in_c * kw] * x[in_c * iw];
        }
    }

    output[batch * (oc * ow) + out_c * ow + out_x] = sum;
}

void ggml_cuda_op_conv1d_grouped(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * kernel = dst->src[0];
    const ggml_tensor * input  = dst->src[1];

    GGML_ASSERT(kernel->type == GGML_TYPE_F32 && input->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32);
    GGML_ASSERT(ggml_is_contiguous(kernel));
    GGML_ASSERT(ggml_is_contiguous(input));
    GGML_ASSERT(ggml_is_contiguous(dst));

    const int32_t * p = (const int32_t *) dst->op_params;
    const int stride   = p[0];
    const int padding  = p[1];
    const int dilation = p[2];
    const int groups   = p[3];

    const int kw        = kernel->ne[0];
    const int ic_g      = kernel->ne[1];
    const int oc        = kernel->ne[2];
    const int iw        = input->ne[0];
    const int ic        = input->ne[1];
    const int n_batches = input->ne[2];
    const int ow        = dst->ne[0];

    GGML_ASSERT(groups > 0);
    GGML_ASSERT(ic_g == ic / groups);
    GGML_ASSERT(oc % groups == 0);

    const int total = n_batches * oc * ow;
    const int blocks = (total + CUDA_CONV1D_GROUPED_BLOCK_SIZE - 1) / CUDA_CONV1D_GROUPED_BLOCK_SIZE;

    conv1d_grouped_f32_kernel<<<blocks, CUDA_CONV1D_GROUPED_BLOCK_SIZE, 0, ctx.stream()>>>(
            (const float *) kernel->data,
            (const float *) input->data,
            (float *) dst->data,
            kw, ic_g, oc, iw, ic, n_batches, ow, stride, padding, dilation, groups);
}
