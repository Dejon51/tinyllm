#include <stdio.h>
#include "tokenizer.h"

int main(void)
{
    char text[] = "hello world";

    for (int i = 0; text[i] != '\0'; i++) {
        int token = encode_char(text[i]);

        printf("'%c' -> %d -> '%c'\n",
               text[i],
               token,
               decode_token(token));
    }

    return 0;
}