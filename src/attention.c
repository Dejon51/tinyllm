#include "attention.h"

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