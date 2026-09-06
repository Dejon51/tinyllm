#include <stdio.h>
#include <stdlib.h>
#include "tokenizer.h"
#include "model.h"
#include "attention.h"
#include "training.h"

void generate_text(
    Model *model,
    char *prompt,
    int num_tokens_to_generate)
{
    int tokens[CONTEXT_SIZE];
    int length = 0;

    // Encode the prompt (up to CONTEXT_SIZE-1 to leave room for generation)
    for (int i = 0; prompt[i] != '\0' && length < CONTEXT_SIZE - 1; i++)
    {
        tokens[length] = encode_char(prompt[i]);
        length++;
    }

    // Print the prompt
    printf("%s", prompt);

    // Generate new tokens
    for (int step = 0; step < num_tokens_to_generate; step++)
    {
        float embeddings[CONTEXT_SIZE][EMBED_SIZE];
        float block_output[CONTEXT_SIZE][EMBED_SIZE];
        float logits[VOCAB_SIZE];

        // Embed the current sequence
        embed_sequence(model, tokens, length, embeddings);

        // Pass through transformer block
        transformer_block(model, embeddings, length, block_output, NULL);

        // Use the last token's output (the prediction for next token)
        float *last_hidden = block_output[length - 1];

        // Compute logits over vocabulary
        compute_logits(model, last_hidden, logits);

        // Sample the next token (temperature = 1.0 for now)
        int next_token = sample_token(logits, 1.0f);

        // Append to sequence
        tokens[length] = next_token;
        length++;

        // Print the new character
        printf("%c", decode_token(next_token));
        fflush(stdout);

        // Stop if we hit the max context size
        if (length >= CONTEXT_SIZE - 1)
            break;
    }
    printf("\n");
}


int main(void)
{
    srand(42);
    Model model;
    model_init(&model);

    Dataset *data = load_dataset("data.txt");
    if (!data) {
        printf("Failed to load dataset\n");
        return 1;
    }

    float learning_rate = 0.001f;
    for (int step = 0; step < 1000000; step++) {
        float loss = train_step(&model, data, learning_rate);
        if (step % 100 == 0) {
            printf("Step %d, loss = %f\n", step, loss);
            // Optionally generate a sample
            char prompt[] = "The ";
            generate_text(&model, prompt, 100);
        }
    }

    free_dataset(data);
    return 0;
}