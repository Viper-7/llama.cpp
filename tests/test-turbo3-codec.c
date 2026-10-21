#include "ggml.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

extern void quantize_row_turbo3_0_ref(const float * x, void * y, int64_t k);
extern void dequantize_row_turbo3_0(const void * x, float * y, int64_t k);
extern void turbo3_cpu_fwht_forward(float * x);
extern void turbo3_cpu_fwht_inverse(float * x);

static int check_d512_round_trip(void) {
    enum { D = 512 };
    float input[D];
    float output[D];

    for (int i = 0; i < D; ++i) {
        input[i] = 4.0f*sinf(0.031f*i) + 0.75f*cosf(0.173f*i) + 0.001f*i;
    }

    const size_t row_size = ggml_row_size(GGML_TYPE_TURBO3_0, D);
    void * encoded_a = malloc(row_size);
    void * encoded_b = malloc(row_size);
    if (encoded_a == NULL || encoded_b == NULL) {
        free(encoded_a);
        free(encoded_b);
        return 1;
    }

    quantize_row_turbo3_0_ref(input, encoded_a, D);
    const size_t quantized = ggml_quantize_chunk(GGML_TYPE_TURBO3_0, input, encoded_b, 0, 1, D, NULL);
    if (quantized != row_size || memcmp(encoded_a, encoded_b, row_size) != 0) {
        fprintf(stderr, "Turbo3 encoding is not deterministic across the row and chunk APIs\n");
        free(encoded_a);
        free(encoded_b);
        return 1;
    }

    dequantize_row_turbo3_0(encoded_a, output, D);
    for (int offset = 0; offset < D; offset += 128) {
        turbo3_cpu_fwht_inverse(output + offset);
    }

    double dot = 0.0;
    double input_norm = 0.0;
    double output_norm = 0.0;
    double squared_error = 0.0;
    for (int i = 0; i < D; ++i) {
        const double input_value = input[i];
        const double output_value = output[i];
        const double error = input_value - output_value;
        dot += input_value*output_value;
        input_norm += input_value*input_value;
        output_norm += output_value*output_value;
        squared_error += error*error;
    }
    const double cosine = dot/sqrt(input_norm*output_norm);
    const double nmse = squared_error/input_norm;

    printf("Turbo3 D512: bytes=%zu cosine=%.9f nmse=%.9f\n", row_size, cosine, nmse);
    free(encoded_a);
    free(encoded_b);

    return row_size != 200 || cosine < 0.98 || nmse > 0.05;
}

static int check_zero_block(void) {
    float input[128] = { 0 };
    float output[128];
    uint8_t encoded[50];

    quantize_row_turbo3_0_ref(input, encoded, 128);
    dequantize_row_turbo3_0(encoded, output, 128);
    turbo3_cpu_fwht_inverse(output);

    for (int i = 0; i < 128; ++i) {
        if (!isfinite(output[i]) || output[i] != 0.0f) {
            fprintf(stderr, "Turbo3 zero block produced %.9g at %d\n", (double) output[i], i);
            return 1;
        }
    }
    return 0;
}

static int check_attention_orientation(void) {
    enum { D = 512 };
    float key[D];
    float query[D];
    float key_rotated[D];
    float key_model[D];
    float query_rotated[D];
    uint8_t encoded[200];

    for (int i = 0; i < D; ++i) {
        key[i] = 1.7f*sinf(0.041f*i) - 0.4f*cosf(0.127f*i) + 0.0007f*i;
        query[i] = 0.9f*cosf(0.023f*i) + 0.3f*sinf(0.191f*i) - 0.0003f*i;
    }

    quantize_row_turbo3_0_ref(key, encoded, D);
    dequantize_row_turbo3_0(encoded, key_rotated, D);
    memcpy(key_model, key_rotated, sizeof(key_model));
    memcpy(query_rotated, query, sizeof(query_rotated));
    for (int offset = 0; offset < D; offset += 128) {
        turbo3_cpu_fwht_inverse(key_model + offset);
        turbo3_cpu_fwht_forward(query_rotated + offset);
    }

    double dot_model = 0.0;
    double dot_rotated = 0.0;
    double dot_wrong = 0.0;
    for (int i = 0; i < D; ++i) {
        dot_model += (double) query[i]*(double) key_model[i];
        dot_rotated += (double) query_rotated[i]*(double) key_rotated[i];
        dot_wrong += (double) query[i]*(double) key_rotated[i];
    }

    const double scale = fmax(1.0, fmax(fabs(dot_model), fabs(dot_rotated)));
    const double relative_error = fabs(dot_model - dot_rotated)/scale;
    const double wrong_relative_error = fabs(dot_model - dot_wrong)/scale;
    printf("Turbo3 attention orientation: relative=%.9g unrotated=%.9g\n",
        relative_error, wrong_relative_error);
    return relative_error > 1e-5 || wrong_relative_error < 1e-2;
}

int main(void) {
    if (strcmp(ggml_type_name(GGML_TYPE_TURBO3_0), "turbo3") != 0 ||
            ggml_blck_size(GGML_TYPE_TURBO3_0) != 128) {
        fprintf(stderr, "Turbo3 type traits are not registered\n");
        return 1;
    }

    return check_d512_round_trip() || check_zero_block() || check_attention_orientation();
}
