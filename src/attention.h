#ifndef ATTENTION_H
#define ATTENTION_H

#include "model.h"

float dot_product(
    float a[EMBED_SIZE],
    float b[EMBED_SIZE]
);

void compute_query(
    Model *model,
    float input[EMBED_SIZE],
    float output[EMBED_SIZE]
);

void compute_key(
    Model *model,
    float input[EMBED_SIZE],
    float output[EMBED_SIZE]
);

void compute_value(
    Model *model,
    float input[EMBED_SIZE],
    float output[EMBED_SIZE]
);

#endif