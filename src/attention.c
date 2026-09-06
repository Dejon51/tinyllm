#include "attention.h"
#include <math.h>

float dot_product(
    float a[EMBED_SIZE],
    float b[EMBED_SIZE])
{
    float result = 0.0f;

    for (int i = 0; i < EMBED_SIZE; i++)
    {
        result += a[i] * b[i];
    }

    return result;
}

void softmax(
    float scores[CONTEXT_SIZE],
    int length)
{
    float sum = 0.0f;

    for (int i = 0; i < length; i++)
    {
        scores[i] = expf(scores[i]);
        sum += scores[i];
    }

    for (int i = 0; i < length; i++)
    {
        scores[i] /= sum;
    }
}

void softmax_masked(
    float scores[CONTEXT_SIZE],
    int length)
{
    float max_score = -INFINITY;

    // Find the maximum score (only over non-masked positions)
    for (int i = 0; i < length; i++)
    {
        if (scores[i] > max_score)
            max_score = scores[i];
    }

    float sum = 0.0f;

    // Exponentiate with max subtraction for numerical stability
    for (int i = 0; i < length; i++)
    {
        scores[i] = expf(scores[i] - max_score);
        sum += scores[i];
    }

    // Normalize to get probabilities
    for (int i = 0; i < length; i++)
    {
        scores[i] /= sum;
    }
}

void compute_attention_scores(
    float queries[CONTEXT_SIZE][EMBED_SIZE],
    float keys[CONTEXT_SIZE][EMBED_SIZE],
    int length,
    float scores[CONTEXT_SIZE][CONTEXT_SIZE])
{
    for (int i = 0; i < length; i++)
    {
        for (int j = 0; j < length; j++)
        {
            scores[i][j] =
                dot_product(queries[i], keys[j]) / sqrtf((float)EMBED_SIZE);
        }
    }
}

void compute_attention_output(
    float scores[CONTEXT_SIZE][CONTEXT_SIZE],
    float values[CONTEXT_SIZE][EMBED_SIZE],
    int length,
    float output[CONTEXT_SIZE][EMBED_SIZE])
{
    for (int i = 0; i < length; i++)
    {
        for (int k = 0; k < EMBED_SIZE; k++)
        {
            output[i][k] = 0.0f;

            for (int j = 0; j < length; j++)
            {
                output[i][k] +=
                    scores[i][j] * values[j][k];
            }
        }
    }
}

void compute_attention_scores_causal(
    float queries[CONTEXT_SIZE][EMBED_SIZE],
    float keys[CONTEXT_SIZE][EMBED_SIZE],
    int length,
    float scores[CONTEXT_SIZE][CONTEXT_SIZE])
{
    for (int i = 0; i < length; i++)
    {
        for (int j = 0; j < length; j++)
        {
            // Only allow attention to positions j <= i (past and current)
            if (j <= i)
            {
                scores[i][j] =
                    dot_product(queries[i], keys[j]) / sqrtf((float)EMBED_SIZE);
            }
            else
            {
                // Mask future positions with -infinity
                scores[i][j] = -INFINITY;
            }
        }
    }
}

void compute_query_head(
    Model *model,
    float input[EMBED_SIZE],
    int head,
    float output[HEAD_SIZE])
{
    for (int i = 0; i < HEAD_SIZE; i++)
    {
        output[i] = 0.0f;
        for (int j = 0; j < EMBED_SIZE; j++)
        {
            output[i] += input[j] * model->Wq[head][j][i];
        }
    }
}

void compute_key_head(
    Model *model,
    float input[EMBED_SIZE],
    int head,
    float output[HEAD_SIZE])
{
    for (int i = 0; i < HEAD_SIZE; i++)
    {
        output[i] = 0.0f;
        for (int j = 0; j < EMBED_SIZE; j++)
        {
            output[i] += input[j] * model->Wk[head][j][i];
        }
    }
}

void compute_key_head(
    Model *model,
    float input[EMBED_SIZE],
    int head,
    float output[HEAD_SIZE]
);