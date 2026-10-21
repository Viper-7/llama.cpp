// gfx906 specialization for GLM-5.3's D=512 F32 absorbed MLA latent.

#include "../fattn-vec.cuh"

DECL_FATTN_VEC_CASE(512, GGML_TYPE_F32, GGML_TYPE_F32);
