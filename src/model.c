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