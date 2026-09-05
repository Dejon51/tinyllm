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
    return 0;
}