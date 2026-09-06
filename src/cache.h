#ifndef CACHE_H
#define CACHE_H

#include "model.h"

// Cache for multi-head attention forward pass
typedef struct {
    float queries[NUM_HEADS][CONTEXT_SIZE][HEAD_SIZE];
    float keys[NUM_HEADS][CONTEXT_SIZE][HEAD_SIZE];
    float values[NUM_HEADS][CONTEXT_SIZE][HEAD_SIZE];

    // Scores before softmax (with causal mask)
    float scores[NUM_HEADS][CONTEXT_SIZE][CONTEXT_SIZE];
    // Attention weights after softmax
    float weights[NUM_HEADS][CONTEXT_SIZE][CONTEXT_SIZE];

    // Concatenated head outputs (before Wo)
    float concatenated[CONTEXT_SIZE][EMBED_SIZE];
    // Final attention output (after Wo)
    float attention_output[CONTEXT_SIZE][EMBED_SIZE];

    int length;
} ForwardCache;

// Cache for the whole transformer block
typedef struct {
    // Input to block (needed for residual)
    float input[CONTEXT_SIZE][EMBED_SIZE];

    // Sub-cache for attention
    ForwardCache attn_cache;

    // After first residual (input + attention output)
    float residual1[CONTEXT_SIZE][EMBED_SIZE];

    // After first layer norm
    float norm1[CONTEXT_SIZE][EMBED_SIZE];

    // Pre-ReLU hidden values for FFN (per token)
    float ffn_hidden[CONTEXT_SIZE][FFN_SIZE];

    // After second residual (norm1 + FFN output)
    float residual2[CONTEXT_SIZE][EMBED_SIZE];

    // Final output after second layer norm
    float output[CONTEXT_SIZE][EMBED_SIZE];

    int length;
} TransformerBlockCache;

// Initialize caches to zero
void cache_init(ForwardCache *cache);
void transformer_cache_init(TransformerBlockCache *cache);

#endif