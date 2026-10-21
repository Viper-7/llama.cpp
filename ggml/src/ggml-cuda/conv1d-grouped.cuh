#pragma once

#include "common.cuh"

#define CUDA_CONV1D_GROUPED_BLOCK_SIZE 256

void ggml_cuda_op_conv1d_grouped(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
