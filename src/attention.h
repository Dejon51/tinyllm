#ifndef ATTENTION_H
#define ATTENTION_H

#include "model.h"


float dot_product(
    float a[EMBED_SIZE],
    float b[EMBED_SIZE]
);

void softmax(
    float scores[CONTEXT_SIZE],
    int length
);

void softmax_masked(float scores[CONTEXT_SIZE],int length);

void compute_attention_scores(
    float queries[CONTEXT_SIZE][EMBED_SIZE],
    float keys[CONTEXT_SIZE][EMBED_SIZE],
    int length,
    float scores[CONTEXT_SIZE][CONTEXT_SIZE]
);

void compute_attention_output(
    float scores[CONTEXT_SIZE][CONTEXT_SIZE],
    float values[CONTEXT_SIZE][EMBED_SIZE],
    int length,
    float output[CONTEXT_SIZE][EMBED_SIZE]
);

void compute_attention_scores_causal(
    float queries[CONTEXT_SIZE][EMBED_SIZE],
    float keys[CONTEXT_SIZE][EMBED_SIZE],
    int length,
    float scores[CONTEXT_SIZE][CONTEXT_SIZE]
);

void compute_query_head(
    Model *model,
    float input[EMBED_SIZE],
    int head,
    float output[HEAD_SIZE]
);
#endif