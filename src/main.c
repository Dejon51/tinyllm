#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <time.h>
#include "tokenizer.h"
#include "model.h"
#include "attention.h"
#include "training.h"


void generate_text(Model *model, char *prompt, int num_tokens_to_generate)
{
    int tokens[CONTEXT_SIZE];
    int length = 0;

    // Make a mutable copy of the prompt because strtok modifies it
    char prompt_copy[CONTEXT_SIZE * MAX_WORD_LEN];
    strncpy(prompt_copy, prompt, sizeof(prompt_copy) - 1);
    prompt_copy[sizeof(prompt_copy) - 1] = '\0';

    // Tokenize the prompt into word IDs
    char *token = strtok(prompt_copy, " \t\n\r\f\v.,;:!?\"()[]{}<>-");
    while (token != NULL && length < CONTEXT_SIZE - 1)
    {
        // Convert to lowercase (assuming your vocab is lowercase)
        for (int i = 0; token[i]; i++) token[i] = tolower(token[i]);
        tokens[length++] = encode_word(token);
        token = strtok(NULL, " \t\n\r\f\v.,;:!?\"()[]{}<>-");
    }

    // Print the original prompt (without modification)
    printf("%s", prompt);

    for (int step = 0; step < num_tokens_to_generate; step++)
    {
        float embeddings[CONTEXT_SIZE][EMBED_SIZE];
        float block_output[CONTEXT_SIZE][EMBED_SIZE];
        float logits[VOCAB_SIZE];

        embed_sequence(model, tokens, length, embeddings);
        transformer_block(model, embeddings, length, block_output, NULL);

        float *last_hidden = block_output[length - 1];
        compute_logits(model, last_hidden, logits);

        int next_token = sample_token_topk(logits, 0.8f, 100);  // pick from top 100 words;
        tokens[length++] = next_token;

        // Print the decoded word followed by a space
        printf("%s ", decode_token(next_token));
        fflush(stdout);

        if (length >= CONTEXT_SIZE - 1)
            break;
    }
    printf("\n");
}

void chat_mode(Model *model)
{
    char prompt[CONTEXT_SIZE * 2]; // generous buffer
    printf("Chat mode. Type 'exit' to quit.\n");

    while (1)
    {
        printf("> ");
        fflush(stdout);
        if (fgets(prompt, sizeof(prompt), stdin) == NULL)
            break;

        // Remove trailing newline
        prompt[strcspn(prompt, "\n")] = '\0';

        if (strcmp(prompt, "exit") == 0 || strcmp(prompt, "quit") == 0)
            break;

        if (strlen(prompt) == 0)
            continue;

        generate_text(model, prompt, 200);
    }
}

int main(int argc, char *argv[])
{
    if (load_vocab("vocab.txt") < 0) {
        printf("Failed to load vocabulary\n");
        return 1;
    }

    Model *model = malloc(sizeof(Model));
    if (!model) { printf("Out of memory\n"); return 1; }

    if (argc > 1 && strcmp(argv[1], "chat") == 0)
    {
        srand((unsigned)time(NULL));
        FILE *checkpoint = fopen("checkpoint.bin", "rb");
        if (!checkpoint) {
            printf("No checkpoint found. Train first or provide checkpoint.bin.\n");
            return 1;
        }
        fclose(checkpoint);
        load_model(model, "checkpoint.bin");
        printf("Loaded checkpoint for chat.\n");
        chat_mode(model);
        return 0;
    }

    // Training mode (default)
    srand(42);
    model_init(model);

    Dataset *data = load_dataset("data.txt");
    if (!data) {
        printf("Failed to load dataset\n");
        return 1;
    }

    int batch_size = 16;
    float learning_rate = 0.001f;

    for (int step = 0; step < 1000000; step++)
    {
        if (step == 200000) learning_rate = 0.0005f;
        if (step == 500000) learning_rate = 0.0001f;

        float loss = train_batch(model, data, batch_size, learning_rate);

        if (step % 100 == 0) {
            printf("Step %d, loss = %f, lr = %f\n", step, loss, learning_rate);
            char prompt[] = "The ";
            generate_text(model, prompt, 200);
        }

        if (step % 10000 == 0 && step > 0) {
            save_model(model, "checkpoint.bin");
            printf("Checkpoint saved.\n");
        }
    }

    free_dataset(data);
    return 0;
}