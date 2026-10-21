#include "zaya-qk-norm.cuh"

template<int block_size>
static __global__ void zaya_qk_norm_f32(
        const float * qk,
        const float * q_raw,
        const float * k_raw,
        const float * k_scale,
        float * dst,
        int n_embd_head,
        int n_head,
        int n_head_kv,
        int n_gqa,
        int n_tokens) {
    const int head_idx = blockIdx.x;
    const int tok      = blockIdx.y;
    const int tid      = threadIdx.x;

    const int n_embd_q = n_embd_head * n_head;
    const int n_embd_k = n_embd_head * n_head_kv;
    const int n_qk     = n_embd_q + n_embd_k;

    float sum = 0.0f;
    if (head_idx < n_head) {
        const int h      = head_idx;
        const int kv     = h / n_gqa;
        const int q_off  = tok * n_embd_q + h  * n_embd_head;
        const int k_off  = tok * n_embd_k + kv * n_embd_head;
        const int out_off = tok * n_qk + h * n_embd_head;

        for (int d = tid; d < n_embd_head; d += block_size) {
            const float v = qk[out_off + d] + 0.5f * (q_raw[q_off + d] + k_raw[k_off + d]);
            dst[out_off + d] = v;
            sum += v * v;
        }

        extern __shared__ float s_sum[];
        sum = block_reduce<block_reduce_method::SUM, block_size>(sum, s_sum);
        const float scale = sqrtf((float) n_embd_head) * rsqrtf(fmaxf(sum, 1e-24f));

        for (int d = tid; d < n_embd_head; d += block_size) {
            dst[out_off + d] *= scale;
        }
    } else {
        const int kv      = head_idx - n_head;
        const int k_off   = tok * n_embd_k + kv * n_embd_head;
        const int out_off = tok * n_qk + n_embd_q + kv * n_embd_head;

        for (int d = tid; d < n_embd_head; d += block_size) {
            float q_sum = 0.0f;
            for (int g = 0; g < n_gqa; ++g) {
                const int h = kv * n_gqa + g;
                q_sum += q_raw[tok * n_embd_q + h * n_embd_head + d];
            }
            const float q_mean = q_sum / (float) n_gqa;
            const float v = qk[out_off + d] + 0.5f * (q_mean + k_raw[k_off + d]);
            dst[out_off + d] = v;
            sum += v * v;
        }

        extern __shared__ float s_sum[];
        sum = block_reduce<block_reduce_method::SUM, block_size>(sum, s_sum);
        const float scale = sqrtf((float) n_embd_head) * k_scale[kv] * rsqrtf(fmaxf(sum, 1e-24f));

        for (int d = tid; d < n_embd_head; d += block_size) {
            dst[out_off + d] *= scale;
        }
    }
}

void ggml_cuda_op_zaya_qk_norm(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * qk      = dst->src[0];
    const ggml_tensor * q_raw   = dst->src[1];
    const ggml_tensor * k_raw   = dst->src[2];
    const ggml_tensor * k_scale = dst->src[3];

    GGML_ASSERT(qk->type      == GGML_TYPE_F32);
    GGML_ASSERT(q_raw->type   == GGML_TYPE_F32);
    GGML_ASSERT(k_raw->type   == GGML_TYPE_F32);
    GGML_ASSERT(k_scale->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type     == GGML_TYPE_F32);
    GGML_ASSERT(ggml_is_contiguous(qk));
    GGML_ASSERT(ggml_is_contiguous(q_raw));
    GGML_ASSERT(ggml_is_contiguous(k_raw));
    GGML_ASSERT(ggml_is_contiguous(k_scale));
    GGML_ASSERT(ggml_is_contiguous(dst));

    const int n_embd_q  = q_raw->ne[0];
    const int n_embd_k  = k_raw->ne[0];
    const int n_tokens  = q_raw->ne[1];
    const int n_head_kv = k_scale->ne[0];
    const int n_embd_head = n_embd_k / n_head_kv;
    const int n_head = n_embd_q / n_embd_head;
    const int n_gqa  = n_head / n_head_kv;

    GGML_ASSERT(qk->ne[0] == n_embd_q + n_embd_k);
    GGML_ASSERT(qk->ne[1] == n_tokens);
    GGML_ASSERT(k_raw->ne[1] == n_tokens);
    GGML_ASSERT(dst->ne[0] == qk->ne[0]);
    GGML_ASSERT(dst->ne[1] == qk->ne[1]);
    GGML_ASSERT(n_embd_k % n_head_kv == 0);
    GGML_ASSERT(n_head % n_head_kv == 0);

    const float * qk_d      = (const float *) qk->data;
    const float * q_raw_d   = (const float *) q_raw->data;
    const float * k_raw_d   = (const float *) k_raw->data;
    const float * k_scale_d = (const float *) k_scale->data;
    float *       dst_d     = (float *) dst->data;

    const dim3 blocks(n_head + n_head_kv, n_tokens, 1);
    const dim3 block_dims(256, 1, 1);
    cudaStream_t stream = ctx.stream();

    zaya_qk_norm_f32<256><<<blocks, block_dims, 32 * sizeof(float), stream>>>(
            qk_d, q_raw_d, k_raw_d, k_scale_d, dst_d,
            n_embd_head, n_head, n_head_kv, n_gqa, n_tokens);
}
