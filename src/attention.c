#include "attention.h"
#include <math.h>
#include <stdlib.h>

/* -------------------- Basic utilities -------------------- */

float dot_product(float a[EMBED_SIZE], float b[EMBED_SIZE])
{
    float result = 0.0f;
    for (int i = 0; i < EMBED_SIZE; i++)
        result += a[i] * b[i];
    return result;
}

void softmax(float scores[CONTEXT_SIZE], int length)
{
    float sum = 0.0f;
    for (int i = 0; i < length; i++)
    {
        scores[i] = expf(scores[i]);
        sum += scores[i];
    }
    for (int i = 0; i < length; i++)
        scores[i] /= sum;
}

void softmax_masked(float scores[CONTEXT_SIZE], int length)
{
    float max_score = -INFINITY;
    for (int i = 0; i < length; i++)
        if (scores[i] > max_score) max_score = scores[i];

    float sum = 0.0f;
    for (int i = 0; i < length; i++)
    {
        scores[i] = expf(scores[i] - max_score);
        sum += scores[i];
    }
    for (int i = 0; i < length; i++)
        scores[i] /= sum;
}

/* -------------------- Single-head legacy -------------------- */

void compute_attention_scores(
    float queries[CONTEXT_SIZE][EMBED_SIZE],
    float keys[CONTEXT_SIZE][EMBED_SIZE],
    int length,
    float scores[CONTEXT_SIZE][CONTEXT_SIZE])
{
    for (int i = 0; i < length; i++)
        for (int j = 0; j < length; j++)
            scores[i][j] = dot_product(queries[i], keys[j]) / sqrtf((float)EMBED_SIZE);
}

void compute_attention_output(
    float scores[CONTEXT_SIZE][CONTEXT_SIZE],
    float values[CONTEXT_SIZE][EMBED_SIZE],
    int length,
    float output[CONTEXT_SIZE][EMBED_SIZE])
{
    for (int i = 0; i < length; i++)
        for (int k = 0; k < EMBED_SIZE; k++)
        {
            output[i][k] = 0.0f;
            for (int j = 0; j < length; j++)
                output[i][k] += scores[i][j] * values[j][k];
        }
}

void compute_attention_scores_causal(
    float queries[CONTEXT_SIZE][EMBED_SIZE],
    float keys[CONTEXT_SIZE][EMBED_SIZE],
    int length,
    float scores[CONTEXT_SIZE][CONTEXT_SIZE])
{
    for (int i = 0; i < length; i++)
        for (int j = 0; j < length; j++)
        {
            if (j <= i)
                scores[i][j] = dot_product(queries[i], keys[j]) / sqrtf((float)EMBED_SIZE);
            else
                scores[i][j] = -INFINITY;
        }
}

/* -------------------- Multi-head projections -------------------- */

void compute_query_head(Model *model, float input[EMBED_SIZE], int head, float output[HEAD_SIZE])
{
    for (int i = 0; i < HEAD_SIZE; i++)
    {
        output[i] = 0.0f;
        for (int j = 0; j < EMBED_SIZE; j++)
            output[i] += input[j] * model->Wq[head][j][i];
    }
}

void compute_key_head(Model *model, float input[EMBED_SIZE], int head, float output[HEAD_SIZE])
{
    for (int i = 0; i < HEAD_SIZE; i++)
    {
        output[i] = 0.0f;
        for (int j = 0; j < EMBED_SIZE; j++)
            output[i] += input[j] * model->Wk[head][j][i];
    }
}

void compute_value_head(Model *model, float input[EMBED_SIZE], int head, float output[HEAD_SIZE])
{
    for (int i = 0; i < HEAD_SIZE; i++)
    {
        output[i] = 0.0f;
        for (int j = 0; j < EMBED_SIZE; j++)
            output[i] += input[j] * model->Wv[head][j][i];
    }
}

void compute_attention_scores_head(
    float queries[CONTEXT_SIZE][HEAD_SIZE],
    float keys[CONTEXT_SIZE][HEAD_SIZE],
    int length,
    float scores[CONTEXT_SIZE][CONTEXT_SIZE])
{
    for (int i = 0; i < length; i++)
        for (int j = 0; j < length; j++)
        {
            if (j <= i)
            {
                float dot = 0.0f;
                for (int k = 0; k < HEAD_SIZE; k++)
                    dot += queries[i][k] * keys[j][k];
                scores[i][j] = dot / sqrtf((float)HEAD_SIZE);
            }
            else
                scores[i][j] = -INFINITY;
        }
}

void compute_attention_output_head(
    float scores[CONTEXT_SIZE][CONTEXT_SIZE],
    float values[CONTEXT_SIZE][HEAD_SIZE],
    int length,
    float output[CONTEXT_SIZE][HEAD_SIZE])
{
    for (int i = 0; i < length; i++)
        for (int k = 0; k < HEAD_SIZE; k++)
        {
            output[i][k] = 0.0f;
            for (int j = 0; j < length; j++)
                output[i][k] += scores[i][j] * values[j][k];
        }
}

/* -------------------- Multi-head attention (with cache) -------------------- */

void multi_head_attention(
    Model *model,
    float input[CONTEXT_SIZE][EMBED_SIZE],
    int length,
    float output[CONTEXT_SIZE][EMBED_SIZE],
    ForwardCache *cache)
{
    float head_outputs[NUM_HEADS][CONTEXT_SIZE][HEAD_SIZE];
    float concatenated[CONTEXT_SIZE][EMBED_SIZE];

    if (cache != NULL) cache->length = length;

    for (int h = 0; h < NUM_HEADS; h++)
    {
        float queries[CONTEXT_SIZE][HEAD_SIZE];
        float keys[CONTEXT_SIZE][HEAD_SIZE];
        float values[CONTEXT_SIZE][HEAD_SIZE];
        float scores[CONTEXT_SIZE][CONTEXT_SIZE];

        for (int pos = 0; pos < length; pos++)
        {
            compute_query_head(model, input[pos], h, queries[pos]);
            compute_key_head(model, input[pos], h, keys[pos]);
            compute_value_head(model, input[pos], h, values[pos]);
        }

        compute_attention_scores_head(queries, keys, length, scores);

        if (cache != NULL)
            for (int i = 0; i < length; i++)
                for (int j = 0; j < length; j++)
                    cache->scores[h][i][j] = scores[i][j];

        for (int i = 0; i < length; i++)
            softmax_masked(scores[i], length);

        if (cache != NULL)
            for (int i = 0; i < length; i++)
                for (int j = 0; j < length; j++)
                    cache->weights[h][i][j] = scores[i][j];

        compute_attention_output_head(scores, values, length, head_outputs[h]);

        if (cache != NULL)
            for (int pos = 0; pos < length; pos++)
                for (int d = 0; d < HEAD_SIZE; d++)
                {
                    cache->queries[h][pos][d] = queries[pos][d];
                    cache->keys[h][pos][d] = keys[pos][d];
                    cache->values[h][pos][d] = values[pos][d];
                }
    }

    for (int pos = 0; pos < length; pos++)
        for (int h = 0; h < NUM_HEADS; h++)
            for (int d = 0; d < HEAD_SIZE; d++)
                concatenated[pos][h * HEAD_SIZE + d] = head_outputs[h][pos][d];

    if (cache != NULL)
        for (int pos = 0; pos < length; pos++)
            for (int i = 0; i < EMBED_SIZE; i++)
                cache->concatenated[pos][i] = concatenated[pos][i];

    for (int pos = 0; pos < length; pos++)
        for (int i = 0; i < EMBED_SIZE; i++)
        {
            output[pos][i] = 0.0f;
            for (int j = 0; j < EMBED_SIZE; j++)
                output[pos][i] += concatenated[pos][j] * model->Wo[j][i];
        }

    if (cache != NULL)
        for (int pos = 0; pos < length; pos++)
            for (int i = 0; i < EMBED_SIZE; i++)
                cache->attention_output[pos][i] = output[pos][i];
}

/* -------------------- Layer norm -------------------- */

void layer_norm(
    float input[EMBED_SIZE],
    float gamma[EMBED_SIZE],
    float beta[EMBED_SIZE],
    float output[EMBED_SIZE])
{
    float mean = 0.0f;
    float variance = 0.0f;

    for (int i = 0; i < EMBED_SIZE; i++) mean += input[i];
    mean /= EMBED_SIZE;

    for (int i = 0; i < EMBED_SIZE; i++)
    {
        float diff = input[i] - mean;
        variance += diff * diff;
    }
    variance /= EMBED_SIZE;

    float std = sqrtf(variance + 1e-5f);
    for (int i = 0; i < EMBED_SIZE; i++)
        output[i] = gamma[i] * (input[i] - mean) / std + beta[i];
}

/* -------------------- Feed-forward -------------------- */

void feed_forward(Model *model, float input[EMBED_SIZE], float output[EMBED_SIZE])
{
    float hidden[FFN_SIZE];
    feed_forward_forward(model, input, output, hidden);
}

void feed_forward_forward(Model *model, float input[EMBED_SIZE], float output[EMBED_SIZE], float hidden_pre_relu[FFN_SIZE])
{
    for (int i = 0; i < FFN_SIZE; i++)
    {
        float sum = model->b1[i];
        for (int j = 0; j < EMBED_SIZE; j++)
            sum += input[j] * model->W1[j][i];
        hidden_pre_relu[i] = sum;
    }

    for (int i = 0; i < EMBED_SIZE; i++)
    {
        float sum = model->b2[i];
        for (int j = 0; j < FFN_SIZE; j++)
        {
            float activated = hidden_pre_relu[j] > 0.0f ? hidden_pre_relu[j] : 0.0f;
            sum += activated * model->W2[j][i];
        }
        output[i] = sum;
    }
}

/* -------------------- Transformer block -------------------- */

void transformer_block(
    Model *model,
    float input[CONTEXT_SIZE][EMBED_SIZE],
    int length,
    float output[CONTEXT_SIZE][EMBED_SIZE],
    TransformerBlockCache *cache)
{
    float attention_output[CONTEXT_SIZE][EMBED_SIZE];
    float residual1[CONTEXT_SIZE][EMBED_SIZE];
    float norm1[CONTEXT_SIZE][EMBED_SIZE];
    float ff_output[CONTEXT_SIZE][EMBED_SIZE];
    float residual2[CONTEXT_SIZE][EMBED_SIZE];

    ForwardCache *attn_cache = (cache != NULL) ? &cache->attn_cache : NULL;
    multi_head_attention(model, input, length, attention_output, attn_cache);

    for (int pos = 0; pos < length; pos++)
        for (int i = 0; i < EMBED_SIZE; i++)
            residual1[pos][i] = input[pos][i] + attention_output[pos][i];

    for (int pos = 0; pos < length; pos++)
        layer_norm(residual1[pos], model->ln1_gamma, model->ln1_beta, norm1[pos]);

    for (int pos = 0; pos < length; pos++)
    {
        if (cache != NULL)
            feed_forward_forward(model, norm1[pos], ff_output[pos], cache->ffn_hidden[pos]);
        else
            feed_forward(model, norm1[pos], ff_output[pos]);
    }

    for (int pos = 0; pos < length; pos++)
        for (int i = 0; i < EMBED_SIZE; i++)
            residual2[pos][i] = norm1[pos][i] + ff_output[pos][i];

    for (int pos = 0; pos < length; pos++)
        layer_norm(residual2[pos], model->ln2_gamma, model->ln2_beta, output[pos]);

    if (cache != NULL)
    {
        cache->length = length;
        for (int pos = 0; pos < length; pos++)
            for (int i = 0; i < EMBED_SIZE; i++)
            {
                cache->input[pos][i] = input[pos][i];
                cache->residual1[pos][i] = residual1[pos][i];
                cache->norm1[pos][i] = norm1[pos][i];
                cache->residual2[pos][i] = residual2[pos][i];
                cache->output[pos][i] = output[pos][i];
            }
    }
}

/* -------------------- Logits & sampling -------------------- */

void compute_logits(Model *model, float hidden[EMBED_SIZE], float logits[VOCAB_SIZE])
{
    for (int v = 0; v < VOCAB_SIZE; v++)
    {
        logits[v] = 0.0f;
        for (int i = 0; i < EMBED_SIZE; i++)
            logits[v] += hidden[i] * model->output_projection[i][v];
    }
}

int sample_token(float logits[VOCAB_SIZE], float temperature)
{
    float probs[VOCAB_SIZE];
    float sum = 0.0f;
    for (int i = 0; i < VOCAB_SIZE; i++)
    {
        probs[i] = expf(logits[i] / temperature);
        sum += probs[i];
    }
    for (int i = 0; i < VOCAB_SIZE; i++)
        probs[i] /= sum;

    float r = (float)rand() / (float)RAND_MAX;
    float cumsum = 0.0f;
    for (int i = 0; i < VOCAB_SIZE; i++)
    {
        cumsum += probs[i];
        if (r < cumsum)
            return i;
    }
    return VOCAB_SIZE - 1;
}