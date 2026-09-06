#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "training.h"
#include "tokenizer.h"

/* -------------------- Dataset -------------------- */

Dataset *load_dataset(const char *filename)
{
    FILE *file = fopen(filename, "rb");
    if (!file) return NULL;
    fseek(file, 0, SEEK_END);
    long size = ftell(file);
    fseek(file, 0, SEEK_SET);
    char *buf = malloc(size+1);
    fread(buf, 1, size, file);
    buf[size] = '\0';
    fclose(file);

    Dataset *ds = malloc(sizeof(Dataset));
    ds->length = size;
    ds->tokens = malloc(size * sizeof(int));
    for (int i=0; i<size; i++) ds->tokens[i] = encode_char(buf[i]);
    free(buf);
    return ds;
}

void free_dataset(Dataset *ds)
{
    if (ds) {
        free(ds->tokens);
        free(ds);
    }
}

void create_batch(Dataset *ds, int batch_size, int context_length, int *inputs, int *targets)
{
    for (int b=0; b<batch_size; b++) {
        int max_start = ds->length - context_length - 1;
        int start = rand() % (max_start+1);
        for (int i=0; i<context_length; i++)
            inputs[b*context_length + i] = ds->tokens[start+i];
        targets[b] = ds->tokens[start+context_length];
    }
}

/* -------------------- Gradients -------------------- */

void init_gradients(Gradients *g) { memset(g, 0, sizeof(Gradients)); }
void zero_gradients(Gradients *g) { memset(g, 0, sizeof(Gradients)); }

void apply_gradients(Model *model, Gradients *g, float lr)
{
    for (int i=0; i<VOCAB_SIZE; i++)
        for (int j=0; j<EMBED_SIZE; j++)
            model->embeddings[i][j] -= lr * g->embeddings[i][j];
    for (int i=0; i<CONTEXT_SIZE; i++)
        for (int j=0; j<EMBED_SIZE; j++)
            model->position_embeddings[i][j] -= lr * g->position_embeddings[i][j];

    for (int h=0; h<NUM_HEADS; h++)
        for (int i=0; i<EMBED_SIZE; i++)
            for (int j=0; j<HEAD_SIZE; j++) {
                model->Wq[h][i][j] -= lr * g->Wq[h][i][j];
                model->Wk[h][i][j] -= lr * g->Wk[h][i][j];
                model->Wv[h][i][j] -= lr * g->Wv[h][i][j];
            }
    for (int i=0; i<EMBED_SIZE; i++)
        for (int j=0; j<EMBED_SIZE; j++)
            model->Wo[i][j] -= lr * g->Wo[i][j];

    for (int i=0; i<EMBED_SIZE; i++) {
        model->ln1_gamma[i] -= lr * g->ln1_gamma[i];
        model->ln1_beta[i] -= lr * g->ln1_beta[i];
        model->ln2_gamma[i] -= lr * g->ln2_gamma[i];
        model->ln2_beta[i] -= lr * g->ln2_beta[i];
    }

    for (int i=0; i<EMBED_SIZE; i++)
        for (int j=0; j<FFN_SIZE; j++)
            model->W1[i][j] -= lr * g->W1[i][j];
    for (int i=0; i<FFN_SIZE; i++)
        model->b1[i] -= lr * g->b1[i];
    for (int i=0; i<FFN_SIZE; i++)
        for (int j=0; j<EMBED_SIZE; j++)
            model->W2[i][j] -= lr * g->W2[i][j];
    for (int i=0; i<EMBED_SIZE; i++)
        model->b2[i] -= lr * g->b2[i];

    for (int i=0; i<EMBED_SIZE; i++)
        for (int j=0; j<VOCAB_SIZE; j++)
            model->output_projection[i][j] -= lr * g->output_projection[i][j];
}

/* -------------------- Loss -------------------- */

float cross_entropy_loss(float logits[VOCAB_SIZE], int target, float grad_logits[VOCAB_SIZE])
{
    float max_logit = -INFINITY;
    for (int i=0; i<VOCAB_SIZE; i++) if (logits[i] > max_logit) max_logit = logits[i];
    float probs[VOCAB_SIZE];
    float sum = 0.0f;
    for (int i=0; i<VOCAB_SIZE; i++) {
        probs[i] = expf(logits[i] - max_logit);
        sum += probs[i];
    }
    for (int i=0; i<VOCAB_SIZE; i++) probs[i] /= sum;
    float loss = -logf(probs[target] + 1e-9f);
    for (int i=0; i<VOCAB_SIZE; i++) grad_logits[i] = probs[i];
    grad_logits[target] -= 1.0f;
    return loss;
}

/* -------------------- Backward functions -------------------- */

void layer_norm_backward(
    float input[EMBED_SIZE],
    float gamma[EMBED_SIZE],
    float beta[EMBED_SIZE],
    float grad_output[EMBED_SIZE],
    float grad_input[EMBED_SIZE],
    float grad_gamma[EMBED_SIZE],
    float grad_beta[EMBED_SIZE])
{
    float mean = 0.0f;
    float var = 0.0f;
    for (int i=0; i<EMBED_SIZE; i++) mean += input[i];
    mean /= EMBED_SIZE;
    for (int i=0; i<EMBED_SIZE; i++) { float d = input[i]-mean; var += d*d; }
    var /= EMBED_SIZE;
    float std = sqrtf(var + 1e-5f);
    float inv_std = 1.0f / std;

    float sum_grad = 0.0f;
    float sum_grad_norm = 0.0f;
    for (int i=0; i<EMBED_SIZE; i++) {
        float norm = (input[i] - mean) * inv_std;
        grad_gamma[i] = grad_output[i] * norm;
        grad_beta[i] = grad_output[i];
        sum_grad += grad_output[i];
        sum_grad_norm += grad_output[i] * norm;
    }

    for (int i=0; i<EMBED_SIZE; i++) {
        float norm = (input[i] - mean) * inv_std;
        grad_input[i] = (grad_output[i] - sum_grad/EMBED_SIZE - norm * sum_grad_norm/EMBED_SIZE) * inv_std * gamma[i];
    }
}

void feed_forward_backward(
    Model *model,
    Gradients *grads,
    float input[EMBED_SIZE],
    float hidden_pre_relu[FFN_SIZE],
    float grad_output[EMBED_SIZE],
    float grad_input[EMBED_SIZE])
{
    float grad_hidden[FFN_SIZE] = {0};

    // Grad through W2 and ReLU
    for (int j=0; j<FFN_SIZE; j++) {
        float relu_deriv = hidden_pre_relu[j] > 0 ? 1.0f : 0.0f;
        float grad = 0.0f;
        for (int i=0; i<EMBED_SIZE; i++)
            grad += model->W2[j][i] * grad_output[i];
        grad_hidden[j] = grad * relu_deriv;
    }

    // W2, b2 grads
    for (int j=0; j<FFN_SIZE; j++) {
        float activated = hidden_pre_relu[j] > 0 ? hidden_pre_relu[j] : 0.0f;
        for (int i=0; i<EMBED_SIZE; i++)
            grads->W2[j][i] += activated * grad_output[i];
    }
    for (int i=0; i<EMBED_SIZE; i++) grads->b2[i] += grad_output[i];

    // W1, b1 grads
    for (int i=0; i<EMBED_SIZE; i++)
        for (int j=0; j<FFN_SIZE; j++)
            grads->W1[i][j] += input[i] * grad_hidden[j];
    for (int j=0; j<FFN_SIZE; j++) grads->b1[j] += grad_hidden[j];

    // grad_input
    for (int i=0; i<EMBED_SIZE; i++) {
        grad_input[i] = 0.0f;
        for (int j=0; j<FFN_SIZE; j++)
            grad_input[i] += model->W1[i][j] * grad_hidden[j];
    }
}

void attention_backward(
    Model *model,
    Gradients *grads,
    float input[CONTEXT_SIZE][EMBED_SIZE],
    ForwardCache *cache,
    float grad_output[CONTEXT_SIZE][EMBED_SIZE],
    float grad_input[CONTEXT_SIZE][EMBED_SIZE])
{
    int length = cache->length;

    // 1. Gradient w.r.t. concatenated (before Wo)
    float grad_concat[CONTEXT_SIZE][EMBED_SIZE] = {0};
    for (int pos=0; pos<length; pos++)
        for (int i=0; i<EMBED_SIZE; i++)
            for (int j=0; j<EMBED_SIZE; j++)
                grad_concat[pos][j] += grad_output[pos][i] * model->Wo[j][i];

    // Wo gradient
    for (int i=0; i<EMBED_SIZE; i++)
        for (int j=0; j<EMBED_SIZE; j++)
            for (int pos=0; pos<length; pos++)
                grads->Wo[i][j] += cache->concatenated[pos][i] * grad_output[pos][j];

    // 2. Per head backward
    for (int h=0; h<NUM_HEADS; h++) {
        // Split grad_concat into per-head grad_output_head
        float grad_head_output[CONTEXT_SIZE][HEAD_SIZE] = {0};
        for (int pos=0; pos<length; pos++)
            for (int d=0; d<HEAD_SIZE; d++)
                grad_head_output[pos][d] = grad_concat[pos][h*HEAD_SIZE + d];

        // Gradients for values, keys, queries (accumulate)
        float grad_values[CONTEXT_SIZE][HEAD_SIZE] = {0};
        float grad_queries[CONTEXT_SIZE][HEAD_SIZE] = {0};
        float grad_keys[CONTEXT_SIZE][HEAD_SIZE] = {0};

        // gradient through attention output: d_output = sum_j weights * values
        // grad_values[j] += sum_i grad_output_head[i] * weights[i][j]
        for (int j=0; j<length; j++)
            for (int d=0; d<HEAD_SIZE; d++)
                for (int i=0; i<length; i++)
                    grad_values[j][d] += cache->weights[h][i][j] * grad_head_output[i][d];

        // gradient through softmax: grad_scores from grad_head_output and values
        float grad_scores[CONTEXT_SIZE][CONTEXT_SIZE] = {0};
        for (int i=0; i<length; i++)
            for (int j=0; j<length; j++) {
                float dot = 0.0f;
                for (int d=0; d<HEAD_SIZE; d++)
                    dot += grad_head_output[i][d] * cache->values[h][j][d];
                grad_scores[i][j] = dot;
            }

        // softmax backward (masked)
        for (int i=0; i<length; i++) {
            float sum_grad = 0.0f;
            for (int j=0; j<length; j++)
                sum_grad += grad_scores[i][j] * cache->weights[h][i][j];
            for (int j=0; j<length; j++)
                grad_scores[i][j] = cache->weights[h][i][j] * (grad_scores[i][j] - sum_grad);
            // Zero out masked positions
            for (int j=i+1; j<length; j++) grad_scores[i][j] = 0.0f;
        }

        // gradient through scaling: scores = dot/sqrt(HEAD_SIZE)
        float scale = 1.0f / sqrtf((float)HEAD_SIZE);
        // Actually scores = dot * scale, so grad_dot = grad_scores * scale.
        // We'll incorporate scale into queries/keys directly.
        for (int i=0; i<length; i++)
            for (int j=0; j<length; j++)
                grad_scores[i][j] *= scale;

        // Now grad_scores is gradient w.r.t. dot product between queries[i] and keys[j]
        // grad_queries[i] += sum_j grad_scores[i][j] * keys[j]
        // grad_keys[j] += sum_i grad_scores[i][j] * queries[i]
        for (int i=0; i<length; i++)
            for (int d=0; d<HEAD_SIZE; d++)
                for (int j=0; j<length; j++) {
                    grad_queries[i][d] += grad_scores[i][j] * cache->keys[h][j][d];
                    grad_keys[j][d] += grad_scores[i][j] * cache->queries[h][i][d];
                }

        // Project gradients back to input and compute Wq/Wk/Wv gradients
        // For each head, input is shared. Accumulate into grad_input.
        for (int pos=0; pos<length; pos++) {
            for (int i=0; i<EMBED_SIZE; i++) {
                float grad_q = 0.0f, grad_k = 0.0f, grad_v = 0.0f;
                for (int d=0; d<HEAD_SIZE; d++) {
                    grad_q += grad_queries[pos][d] * model->Wq[h][i][d];
                    grad_k += grad_keys[pos][d] * model->Wk[h][i][d];
                    grad_v += grad_values[pos][d] * model->Wv[h][i][d];
                }
                grad_input[pos][i] += grad_q + grad_k + grad_v;
            }
        }

        // Weight gradients
        for (int pos=0; pos<length; pos++) {
            for (int i=0; i<EMBED_SIZE; i++) {
                for (int d=0; d<HEAD_SIZE; d++) {
                    grads->Wq[h][i][d] += input[pos][i] * grad_queries[pos][d];
                    grads->Wk[h][i][d] += input[pos][i] * grad_keys[pos][d];
                    grads->Wv[h][i][d] += input[pos][i] * grad_values[pos][d];
                }
            }
        }
    }
}

void transformer_block_backward(
    Model *model,
    Gradients *grads,
    TransformerBlockCache *cache,
    float grad_output[CONTEXT_SIZE][EMBED_SIZE],
    float grad_input[CONTEXT_SIZE][EMBED_SIZE])
{
    int length = cache->length;

    // Back through layer_norm2
    float grad_residual2[CONTEXT_SIZE][EMBED_SIZE] = {0};
    float grad_ln2_gamma[EMBED_SIZE] = {0};
    float grad_ln2_beta[EMBED_SIZE] = {0};
    for (int pos=0; pos<length; pos++) {
        float grad_in[EMBED_SIZE];
        layer_norm_backward(cache->residual2[pos], model->ln2_gamma, model->ln2_beta,
                            grad_output[pos], grad_in, grad_ln2_gamma, grad_ln2_beta);
        for (int i=0; i<EMBED_SIZE; i++) {
            grad_residual2[pos][i] += grad_in[i];
            grads->ln2_gamma[i] += grad_ln2_gamma[i];
            grads->ln2_beta[i] += grad_ln2_beta[i];
        }
    }

    // Back through residual2: grad_residual2 splits to norm1 and ff_output equally
    float grad_norm1[CONTEXT_SIZE][EMBED_SIZE] = {0};
    float grad_ff_output[CONTEXT_SIZE][EMBED_SIZE] = {0};
    for (int pos=0; pos<length; pos++)
        for (int i=0; i<EMBED_SIZE; i++) {
            grad_norm1[pos][i] = grad_residual2[pos][i];
            grad_ff_output[pos][i] = grad_residual2[pos][i];
        }

    // Back through FFN
    float grad_norm1_from_ffn[CONTEXT_SIZE][EMBED_SIZE] = {0};
    for (int pos=0; pos<length; pos++) {
        float grad_input_ffn[EMBED_SIZE];
        feed_forward_backward(model, grads, cache->norm1[pos], cache->ffn_hidden[pos],
                              grad_ff_output[pos], grad_input_ffn);
        for (int i=0; i<EMBED_SIZE; i++)
            grad_norm1_from_ffn[pos][i] = grad_input_ffn[i];
    }

    // Add to grad_norm1
    for (int pos=0; pos<length; pos++)
        for (int i=0; i<EMBED_SIZE; i++)
            grad_norm1[pos][i] += grad_norm1_from_ffn[pos][i];

    // Back through layer_norm1
    float grad_residual1[CONTEXT_SIZE][EMBED_SIZE] = {0};
    float grad_ln1_gamma[EMBED_SIZE] = {0};
    float grad_ln1_beta[EMBED_SIZE] = {0};
    for (int pos=0; pos<length; pos++) {
        float grad_in[EMBED_SIZE];
        layer_norm_backward(cache->residual1[pos], model->ln1_gamma, model->ln1_beta,
                            grad_norm1[pos], grad_in, grad_ln1_gamma, grad_ln1_beta);
        for (int i=0; i<EMBED_SIZE; i++) {
            grad_residual1[pos][i] += grad_in[i];
            grads->ln1_gamma[i] += grad_ln1_gamma[i];
            grads->ln1_beta[i] += grad_ln1_beta[i];
        }
    }

    // Back through residual1: splits to input and attention output
    float grad_attention_output[CONTEXT_SIZE][EMBED_SIZE] = {0};
    for (int pos=0; pos<length; pos++)
        for (int i=0; i<EMBED_SIZE; i++) {
            grad_input[pos][i] = grad_residual1[pos][i];  // part to input
            grad_attention_output[pos][i] = grad_residual1[pos][i]; // part to attention
        }

    // Back through attention
    attention_backward(model, grads, cache->input, &cache->attn_cache, grad_attention_output, grad_input);
}

/* -------------------- Training step -------------------- */

float train_step(Model *model, Dataset *dataset, float learning_rate)
{
    const int context_length = CONTEXT_SIZE - 1;
    int inputs[context_length];
    int target;

    int max_start = dataset->length - context_length - 1;
    if (max_start < 0) return 0.0f;
    int start = rand() % (max_start + 1);
    for (int i = 0; i < context_length; i++)
        inputs[i] = dataset->tokens[start + i];
    target = dataset->tokens[start + context_length];

    // Forward pass with cache
    TransformerBlockCache block_cache;
    transformer_cache_init(&block_cache);

    float embeddings[CONTEXT_SIZE][EMBED_SIZE];
    embed_sequence(model, inputs, context_length, embeddings);

    float block_output[CONTEXT_SIZE][EMBED_SIZE];
    transformer_block(model, embeddings, context_length, block_output, &block_cache);

    float logits[VOCAB_SIZE];
    compute_logits(model, block_output[context_length - 1], logits);

    float grad_logits[VOCAB_SIZE];
    float loss = cross_entropy_loss(logits, target, grad_logits);

    // Gradients for output projection and last hidden
    Gradients grads;
    init_gradients(&grads);

    float grad_last_hidden[EMBED_SIZE];
    // backward output projection
    for (int i = 0; i < EMBED_SIZE; i++)
        for (int v = 0; v < VOCAB_SIZE; v++)
            grads.output_projection[i][v] += block_output[context_length - 1][i] * grad_logits[v];

    for (int i = 0; i < EMBED_SIZE; i++) {
        grad_last_hidden[i] = 0.0f;
        for (int v = 0; v < VOCAB_SIZE; v++)
            grad_last_hidden[i] += model->output_projection[i][v] * grad_logits[v];
    }

    // We need gradient w.r.t. all block outputs; only last token has gradient, others zero.
    float grad_block_output[CONTEXT_SIZE][EMBED_SIZE] = {0};
    for (int i = 0; i < EMBED_SIZE; i++)
        grad_block_output[context_length - 1][i] = grad_last_hidden[i];

    // Backward through transformer block
    float grad_embeddings[CONTEXT_SIZE][EMBED_SIZE] = {0};
    transformer_block_backward(model, &grads, &block_cache, grad_block_output, grad_embeddings);

    // Gradient for embeddings (token + position)
    for (int pos = 0; pos < context_length; pos++) {
        int token = inputs[pos];
        for (int i = 0; i < EMBED_SIZE; i++) {
            grads.embeddings[token][i] += grad_embeddings[pos][i];
            grads.position_embeddings[pos][i] += grad_embeddings[pos][i];
        }
    }

    // Apply gradients
    apply_gradients(model, &grads, learning_rate);

    return loss;
}