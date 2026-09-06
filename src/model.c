#include <stdlib.h>
#include "model.h"

void model_init(Model *model)
{
    for (int token = 0; token < VOCAB_SIZE; token++)
    {
        for (int i = 0; i < EMBED_SIZE; i++)
        {

            float random = (float)rand() / (float)RAND_MAX;

            model->embeddings[token][i] = random * 0.02f - 0.01f;
        }
    }

    for (int pos = 0; pos < CONTEXT_SIZE; pos++)
    {
        for (int i = 0; i < EMBED_SIZE; i++)
        {

            float random = (float)rand() / (float)RAND_MAX;

            model->position_embeddings[pos][i] = random * 0.02f - 0.01f;
        }
    }
    // Attention weights for each head
    for (int h = 0; h < NUM_HEADS; h++)
    {
        for (int i = 0; i < EMBED_SIZE; i++)
        {
            for (int j = 0; j < HEAD_SIZE; j++)
            {
                float random = (float)rand() / (float)RAND_MAX;
                model->Wq[h][i][j] = random * 0.02f - 0.01f;

                random = (float)rand() / (float)RAND_MAX;
                model->Wk[h][i][j] = random * 0.02f - 0.01f;

                random = (float)rand() / (float)RAND_MAX;
                model->Wv[h][i][j] = random * 0.02f - 0.01f;
            }
        }
    }

    // Output projection Wo remains 2D
    for (int i = 0; i < EMBED_SIZE; i++)
    {
        for (int j = 0; j < EMBED_SIZE; j++)
        {
            float random = (float)rand() / (float)RAND_MAX;
            model->Wo[i][j] = random * 0.02f - 0.01f;
        }
    }
    // Layer normalization parameters (gamma = 1, beta = 0)
    for (int i = 0; i < EMBED_SIZE; i++)
    {
        model->ln1_gamma[i] = 1.0f;
        model->ln1_beta[i] = 0.0f;
        model->ln2_gamma[i] = 1.0f;
        model->ln2_beta[i] = 0.0f;
    }

    // Feed-forward network weights
    for (int i = 0; i < EMBED_SIZE; i++)
    {
        for (int j = 0; j < FFN_SIZE; j++)
        {
            float random = (float)rand() / (float)RAND_MAX;
            model->W1[i][j] = random * 0.02f - 0.01f;
        }
    }
    for (int i = 0; i < FFN_SIZE; i++)
    {
        model->b1[i] = 0.0f;
    }
    for (int i = 0; i < FFN_SIZE; i++)
    {
        for (int j = 0; j < EMBED_SIZE; j++)
        {
            float random = (float)rand() / (float)RAND_MAX;
            model->W2[i][j] = random * 0.02f - 0.01f;
        }
    }
    for (int i = 0; i < EMBED_SIZE; i++)
    {
        model->b2[i] = 0.0f;
    }
    // Output projection weights
    for (int i = 0; i < EMBED_SIZE; i++)
    {
        for (int j = 0; j < VOCAB_SIZE; j++)
        {
            float random = (float)rand() / (float)RAND_MAX;
            model->output_projection[i][j] = random * 0.02f - 0.01f;
        }
    }
}
void get_embedding(
    Model *model,
    int token,
    float *output)
{
    if (token < 0 || token >= VOCAB_SIZE)
        return;

    for (int i = 0; i < EMBED_SIZE; i++)
    {
        output[i] = model->embeddings[token][i];
    }
}

void embed_sequence(
    Model *model,
    int *tokens,
    int length,
    float output[CONTEXT_SIZE][EMBED_SIZE])
{
    for (int pos = 0; pos < length; pos++)
    {

        int token = tokens[pos];
        if (token < 0 || token >= VOCAB_SIZE)
            continue;

        for (int i = 0; i < EMBED_SIZE; i++)
        {
            output[pos][i] = model->embeddings[token][i] + model->position_embeddings[pos][i];
        }
    }
}

// void feed_forward(
//     Model *model,
//     float input[EMBED_SIZE],
//     float output[EMBED_SIZE])
// {
//     float hidden[FFN_SIZE];

//     // First linear layer + ReLU
//     for (int i = 0; i < FFN_SIZE; i++)
//     {
//         hidden[i] = model->b1[i];
//         for (int j = 0; j < EMBED_SIZE; j++)
//         {
//             hidden[i] += input[j] * model->W1[j][i];
//         }
//         hidden[i] = hidden[i] > 0.0f ? hidden[i] : 0.0f;
//     }

//     // Second linear layer
//     for (int i = 0; i < EMBED_SIZE; i++)
//     {
//         output[i] = model->b2[i];
//         for (int j = 0; j < FFN_SIZE; j++)
//         {
//             output[i] += hidden[j] * model->W2[j][i];
//         }
//     }
// }

// void compute_logits(
//     Model *model,
//     float hidden[EMBED_SIZE],
//     float logits[VOCAB_SIZE])
// {
//     for (int v = 0; v < VOCAB_SIZE; v++)
//     {
//         logits[v] = 0.0f;
//         for (int i = 0; i < EMBED_SIZE; i++)
//         {
//             logits[v] += hidden[i] * model->output_projection[i][v];
//         }
//     }
// }