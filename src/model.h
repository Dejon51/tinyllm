#ifndef MODEL_H
#define MODEL_H

#define FFN_SIZE 128

#define VOCAB_SIZE 128
#define EMBED_SIZE 32
#define CONTEXT_SIZE 64
#define NUM_HEADS 4
#define HEAD_SIZE (EMBED_SIZE / NUM_HEADS)

typedef struct
{
    float embeddings[VOCAB_SIZE][EMBED_SIZE];
    float position_embeddings[CONTEXT_SIZE][EMBED_SIZE];

    float Wq[NUM_HEADS][EMBED_SIZE][HEAD_SIZE];
    float Wk[NUM_HEADS][EMBED_SIZE][HEAD_SIZE];
    float Wv[NUM_HEADS][EMBED_SIZE][HEAD_SIZE];
    float Wo[EMBED_SIZE][EMBED_SIZE];

    float ln1_gamma[EMBED_SIZE];
    float ln1_beta[EMBED_SIZE];
    float ln2_gamma[EMBED_SIZE];
    float ln2_beta[EMBED_SIZE];

    float W1[EMBED_SIZE][FFN_SIZE];
    float b1[FFN_SIZE];
    float W2[FFN_SIZE][EMBED_SIZE];
    float b2[EMBED_SIZE];

    float output_projection[EMBED_SIZE][VOCAB_SIZE];
} Model;

void model_init(Model *model);
void get_embedding(Model *model, int token, float *output);
void embed_sequence(Model *model, int *tokens, int length, float output[CONTEXT_SIZE][EMBED_SIZE]);

#endif