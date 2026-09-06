#include "attention.h"
#include <math.h>

float dot_product(
    float a[EMBED_SIZE],
    float b[EMBED_SIZE]
)
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
    int length
)
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

void compute_query(
    Model *model,
    float input[EMBED_SIZE],
    float output[EMBED_SIZE]
)
{
    for (int i = 0; i < EMBED_SIZE; i++) {

        output[i] = 0.0f;

        for (int j = 0; j < EMBED_SIZE; j++) {
            output[i] += input[j] * model->Wq[j][i];
        }
    }
}

void compute_key(
    Model *model,
    float input[EMBED_SIZE],
    float output[EMBED_SIZE]
)
{
    for (int i = 0; i < EMBED_SIZE; i++) {

        output[i] = 0.0f;

        for (int j = 0; j < EMBED_SIZE; j++) {
            output[i] +=
                input[j] * model->Wk[j][i];
        }
    }
}

void compute_value(
    Model *model,
    float input[EMBED_SIZE],
    float output[EMBED_SIZE]
)
{
    for (int i = 0; i < EMBED_SIZE; i++) {

        output[i] = 0.0f;

        for (int j = 0; j < EMBED_SIZE; j++) {
            output[i] +=
                input[j] * model->Wv[j][i];
        }
    }
}

void compute_attention_scores(
    float queries[CONTEXT_SIZE][EMBED_SIZE],
    float keys[CONTEXT_SIZE][EMBED_SIZE],
    int length,
    float scores[CONTEXT_SIZE][CONTEXT_SIZE]
)
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

