#pragma once

#ifdef __cplusplus
extern "C" {
#endif

int kiln_lm_head_argmax_bf16_batch(
    const void *x,
    const void *weight_t,
    float *scores,
    unsigned int *tokens,
    int batch,
    int hidden,
    int vocab,
    void *stream
);

#ifdef __cplusplus
}
#endif
