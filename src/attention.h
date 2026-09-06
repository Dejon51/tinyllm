#ifndef ATTENTION_H
#define ATTENTION_H

#include "model.h"
#include "cache.h"
#include <math.h>

// Basic utilities
float dot_product(float a[EMBED_SIZE], float b[EMBED_SIZE]);
void softmax(float scores[CONTEXT_SIZE], int length);
void softmax_masked(float scores[CONTEXT_SIZE], int length);

// Single-head attention legacy functions (unused but kept)
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

// Multi-head attention
void compute_query_head(Model *model, float input[EMBED_SIZE], int head, float output[HEAD_SIZE]);
void compute_key_head(Model *model, float input[EMBED_SIZE], int head, float output[HEAD_SIZE]);
void compute_value_head(Model *model, float input[EMBED_SIZE], int head, float output[HEAD_SIZE]);
void compute_attention_scores_head(
    float queries[CONTEXT_SIZE][HEAD_SIZE],
    float keys[CONTEXT_SIZE][HEAD_SIZE],
    int length,
    float scores[CONTEXT_SIZE][CONTEXT_SIZE]
);
void compute_attention_output_head(
    float scores[CONTEXT_SIZE][CONTEXT_SIZE],
    float values[CONTEXT_SIZE][HEAD_SIZE],
    int length,
    float output[CONTEXT_SIZE][HEAD_SIZE]
);
void multi_head_attention(
    Model *model,
    float input[CONTEXT_SIZE][EMBED_SIZE],
    int length,
    float output[CONTEXT_SIZE][EMBED_SIZE],
    ForwardCache *cache
);

// Layer normalization
void layer_norm(
    float input[EMBED_SIZE],
    float gamma[EMBED_SIZE],
    float beta[EMBED_SIZE],
    float output[EMBED_SIZE]
);

// Feed-forward (inference)
void feed_forward(
    Model *model,
    float input[EMBED_SIZE],
    float output[EMBED_SIZE]
);

// Feed-forward with hidden capture (for training)
void feed_forward_forward(
    Model *model,
    float input[EMBED_SIZE],
    float output[EMBED_SIZE],
    float hidden_pre_relu[FFN_SIZE]
);

// Transformer block (forward, cache optional)
void transformer_block(
    Model *model,
    float input[CONTEXT_SIZE][EMBED_SIZE],
    int length,
    float output[CONTEXT_SIZE][EMBED_SIZE],
    TransformerBlockCache *cache
);

// Logits and sampling
void compute_logits(Model *model, float hidden[EMBED_SIZE], float logits[VOCAB_SIZE]);
int sample_token(float logits[VOCAB_SIZE], float temperature);

#endif