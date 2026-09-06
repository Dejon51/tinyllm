#ifndef TRAINING_H
#define TRAINING_H

#include "model.h"
#include "attention.h"

// Structure to store gradients of model parameters
typedef struct {
    float embeddings[VOCAB_SIZE][EMBED_SIZE];
    float position_embeddings[CONTEXT_SIZE][EMBED_SIZE];

    float Wq[NUM_HEADS][EMBED_SIZE][HEAD_SIZE];
    float Wk[NUM_HEADS][EMBED_SIZE][HEAD_SIZE];
    float Wv[NUM_HEADS][EMBED_SIZE][HEAD_SIZE];
    float Wo[EMBED_SIZE][EMBED_SIZE];

    float ln1_gamma[EMBED_SIZE];
    float ln1_beta[EMBED_SIZE];
    float ln2_gamma[EMBED_SIZE];
    float ln2_beta[EMBED_SIZE];

    float W1[EMBED_SIZE][FFN_SIZE];
    float b1[FFN_SIZE];
    float W2[FFN_SIZE][EMBED_SIZE];
    float b2[EMBED_SIZE];

    float output_projection[EMBED_SIZE][VOCAB_SIZE];
} Gradients;

typedef struct {
    int *tokens;
    int length;
} Dataset;

Dataset *load_dataset(const char *filename);
void free_dataset(Dataset *dataset);
void create_batch(Dataset *dataset, int batch_size, int context_length, int *inputs, int *targets);

float train_step(Model *model, Dataset *dataset, float learning_rate);

void init_gradients(Gradients *grads);
void zero_gradients(Gradients *grads);
void apply_gradients(Model *model, Gradients *grads, float learning_rate);

// Backward functions
float cross_entropy_loss(float logits[VOCAB_SIZE], int target, float grad_logits[VOCAB_SIZE]);
void layer_norm_backward(
    float input[EMBED_SIZE],
    float gamma[EMBED_SIZE],
    float beta[EMBED_SIZE],
    float grad_output[EMBED_SIZE],
    float grad_input[EMBED_SIZE],
    float grad_gamma[EMBED_SIZE],
    float grad_beta[EMBED_SIZE]
);
void feed_forward_backward(
    Model *model,
    Gradients *grads,
    float input[EMBED_SIZE],
    float hidden_pre_relu[FFN_SIZE],
    float grad_output[EMBED_SIZE],
    float grad_input[EMBED_SIZE]
);
void attention_backward(
    Model *model,
    Gradients *grads,
    float input[CONTEXT_SIZE][EMBED_SIZE],
    ForwardCache *cache,
    float grad_output[CONTEXT_SIZE][EMBED_SIZE],
    float grad_input[CONTEXT_SIZE][EMBED_SIZE]
);
void transformer_block_backward(
    Model *model,
    Gradients *grads,
    TransformerBlockCache *cache,
    float grad_output[CONTEXT_SIZE][EMBED_SIZE],
    float grad_input[CONTEXT_SIZE][EMBED_SIZE]
);

#endif