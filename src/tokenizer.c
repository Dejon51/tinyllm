#include "tokenizer.h"

int encode_char(char c)
{
    unsigned char value = (unsigned char)c;

    if (value >= VOCAB_SIZE)
        return 0;

    return value;
}

char decode_token(int token)
{
    if (token < 0 || token >= VOCAB_SIZE)
        return '?';

    return (char)token;
}