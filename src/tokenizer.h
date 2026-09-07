#ifndef TOKENIZER_H
#define TOKENIZER_H

#define MAX_VOCAB_SIZE 10060   // 10,000 words + <UNK> + possible padding
#define MAX_WORD_LEN 32

extern char vocab[MAX_VOCAB_SIZE][MAX_WORD_LEN];
extern int vocab_size;

// Load vocabulary from file
int load_vocab(const char *filename);

// Convert word to token ID (returns 0 for <UNK>)
int encode_word(const char *word);

// Convert token ID to word string (returns "<UNK>" for out-of-range)
const char* decode_token(int token);

#endif