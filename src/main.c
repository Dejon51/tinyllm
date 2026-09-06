#include <stdio.h>
#include <stdlib.h>

#include "tokenizer.h"
#include "model.h"
#include "attention.h"

int main(void)
{
    srand(42);
    Model model;

    model_init(&model);

    char text[] = "hello";

    int tokens[5];

    // Convert characters into token IDs
    for (int i = 0; i < 5; i++)
    {
        tokens[i] = encode_char(text[i]);
    }

    float embeddings[CONTEXT_SIZE][EMBED_SIZE];

    embed_sequence(
        &model,
        tokens,
        5,
        embeddings);

    // Print the results
    for (int pos = 0; pos < 5; pos++)
    {

        printf("'%c' (%d): ",
               text[pos],
               tokens[pos]);

        for (int i = 0; i < EMBED_SIZE; i++)
        {
            printf("%.3f ", embeddings[pos][i]);
        }

        printf("\n");
    }
    float query[EMBED_SIZE];

    compute_query(
        &model,
        embeddings[0],
        query);

    printf("\nQuery for first token:\n");

    for (int i = 0; i < EMBED_SIZE; i++)
    {
        printf("%.6f ", query[i]);
    }

    printf("\n");
    float key[EMBED_SIZE];

    compute_key(
        &model,
        embeddings[0],
        key);

    printf("\nKey for first token:\n");

    for (int i = 0; i < EMBED_SIZE; i++)
    {
        printf("%.6f ", key[i]);
    }

    printf("\n");

    // Compute Q, K, and V for every token
    float queries[CONTEXT_SIZE][EMBED_SIZE];
    float keys[CONTEXT_SIZE][EMBED_SIZE];
    float values[CONTEXT_SIZE][EMBED_SIZE];

    for (int pos = 0; pos < 5; pos++)
    {
        compute_query(
            &model,
            embeddings[pos],
            queries[pos]);

        compute_key(
            &model,
            embeddings[pos],
            keys[pos]);

        compute_value(
            &model,
            embeddings[pos],
            values[pos]);
    }

    // Compute attention scores
    float scores[CONTEXT_SIZE][CONTEXT_SIZE];

    compute_attention_scores(
        queries,
        keys,
        5,
        scores);

    // Softmax each row

    printf("\nRaw attention scores:\n");

    for (int i = 0; i < 5; i++)
    {
        printf("'%c': ", text[i]);

        for (int j = 0; j < 5; j++)
        {
            printf("%.10f ", scores[i][j]);
        }

        printf("\n");
    }
    for (int i = 0; i < 5; i++)
    {
        softmax(scores[i], 5);
    }

    // Print attention weights
    printf("\nAttention weights:\n");

    for (int i = 0; i < 5; i++)
    {
        printf("'%c': ", text[i]);

        for (int j = 0; j < 5; j++)
        {
            printf("%.3f ", scores[i][j]);
        }

        printf("\n");
    }

    return 0;
}