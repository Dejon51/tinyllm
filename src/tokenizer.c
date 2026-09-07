#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <ctype.h>
#include "tokenizer.h"

char vocab[MAX_VOCAB_SIZE][MAX_WORD_LEN];
int vocab_size = 0;

int load_vocab(const char *filename)
{
    FILE *file = fopen(filename, "r");
    if (!file) return -1;
    vocab_size = 0;
    char line[MAX_WORD_LEN];
    while (fgets(line, sizeof(line), file) && vocab_size < MAX_VOCAB_SIZE) {
        // Remove newline
        line[strcspn(line, "\n")] = '\0';
        strncpy(vocab[vocab_size], line, MAX_WORD_LEN-1);
        vocab[vocab_size][MAX_WORD_LEN-1] = '\0';
        vocab_size++;
    }
    fclose(file);
    return vocab_size;
}

int encode_word(const char *word)
{
    // Simple linear search (could be improved with hash table)
    for (int i = 0; i < vocab_size; i++) {
        if (strcmp(vocab[i], word) == 0) {
            return i;
        }
    }
    return 0; // <UNK>
}

const char* decode_token(int token)
{
    if (token < 0 || token >= vocab_size) return "<UNK>";
    return vocab[token];
}