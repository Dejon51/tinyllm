#include "attention.h"
#include <math.h>
#include <stdlib.h>
#include <immintrin.h>

/* -------------------- AVX2/FMA helpers -------------------- */

/* Horizontal sum of 8 packed floats down to a scalar. */
static inline float hsum256_ps(__m256 v)
{
    __m128 lo = _mm256_castps256_ps128(v);
    __m128 hi = _mm256_extractf128_ps(v, 1);
    lo = _mm_add_ps(lo, hi);
    __m128 shuf = _mm_movehdup_ps(lo);
    __m128 sums = _mm_add_ps(lo, shuf);
    shuf = _mm_movehl_ps(shuf, sums);
    sums = _mm_add_ss(sums, shuf);
    return _mm_cvtss_f32(sums);
}

/* dot(a, b) over n elements, 8-wide FMA with a scalar tail for n % 8 != 0. */
static inline float dot_avx(const float *a, const float *b, int n)
{
    __m256 acc = _mm256_setzero_ps();
    int i = 0;
    for (; i + 8 <= n; i += 8)
    {
        __m256 va = _mm256_loadu_ps(a + i);
        __m256 vb = _mm256_loadu_ps(b + i);
        acc = _mm256_fmadd_ps(va, vb, acc);
    }
    float result = hsum256_ps(acc);
    for (; i < n; i++)
        result += a[i] * b[i];
    return result;
}

/* out[i] += scalar * w[i] for i in [0, n), 8-wide FMA with scalar tail.
   This is the core operation behind every mat-vec loop below: each column
   of a weight matrix gets accumulated into the output, scaled by one
   input element at a time. */
static inline void fma_scale_accumulate(float *out, const float *w, float scalar, int n)
{
    __m256 vs = _mm256_set1_ps(scalar);
    int i = 0;
    for (; i + 8 <= n; i += 8)
    {
        __m256 vw = _mm256_loadu_ps(w + i);
        __m256 vout = _mm256_loadu_ps(out + i);
        vout = _mm256_fmadd_ps(vs, vw, vout);
        _mm256_storeu_ps(out + i, vout);
    }
    for (; i < n; i++)
        out[i] += scalar * w[i];
}

/* out[i] = 0 for i in [0, n). */
static inline void vec_zero(float *out, int n)
{
    __m256 vz = _mm256_setzero_ps();
    int i = 0;
    for (; i + 8 <= n; i += 8)
        _mm256_storeu_ps(out + i, vz);
    for (; i < n; i++)
        out[i] = 0.0f;
}

/* out[i] *= scalar for i in [0, n). */
static inline void vec_scale(float *out, float scalar, int n)
{
    __m256 vs = _mm256_set1_ps(scalar);
    int i = 0;
    for (; i + 8 <= n; i += 8)
    {
        __m256 v = _mm256_loadu_ps(out + i);
        v = _mm256_mul_ps(v, vs);
        _mm256_storeu_ps(out + i, v);
    }
    for (; i < n; i++)
        out[i] *= scalar;
}

/* -------------------- Basic utilities -------------------- */

float dot_product(float a[EMBED_SIZE], float b[EMBED_SIZE])
{
    return dot_avx(a, b, EMBED_SIZE);
}

void softmax(float scores[CONTEXT_SIZE], int length)
{
    /* expf has no portable AVX2 equivalent in libc, so this stays scalar.
       The sum and the final normalization are vectorized instead. */
    float sum = 0.0f;
    for (int i = 0; i < length; i++)
    {
        scores[i] = expf(scores[i]);
        sum += scores[i];
    }
    vec_scale(scores, 1.0f / sum, length);
}

void softmax_masked(float scores[CONTEXT_SIZE], int length)
{
    float max_score = -INFINITY;
    for (int i = 0; i < length; i++)
        if (scores[i] > max_score) max_score = scores[i];

    float sum = 0.0f;
    for (int i = 0; i < length; i++)
    {
        scores[i] = expf(scores[i] - max_score);
        sum += scores[i];
    }
    vec_scale(scores, 1.0f / sum, length);
}

/* -------------------- Single-head legacy -------------------- */

void compute_attention_scores(
    float queries[CONTEXT_SIZE][EMBED_SIZE],
    float keys[CONTEXT_SIZE][EMBED_SIZE],
    int length,
    float scores[CONTEXT_SIZE][CONTEXT_SIZE])
{
    float inv_sqrt_embed = 1.0f / sqrtf((float)EMBED_SIZE);

    for (int i = 0; i < length; i++)
        for (int j = 0; j < length; j++)
            scores[i][j] = dot_avx(queries[i], keys[j], EMBED_SIZE) * inv_sqrt_embed;
}

void compute_attention_output(
    float scores[CONTEXT_SIZE][CONTEXT_SIZE],
    float values[CONTEXT_SIZE][EMBED_SIZE],
    int length,
    float output[CONTEXT_SIZE][EMBED_SIZE])
{
    for (int i = 0; i < length; i++)
    {
        vec_zero(output[i], EMBED_SIZE);
        for (int j = 0; j < length; j++)
            fma_scale_accumulate(output[i], values[j], scores[i][j], EMBED_SIZE);
    }
}

void compute_attention_scores_causal(
    float queries[CONTEXT_SIZE][EMBED_SIZE],
    float keys[CONTEXT_SIZE][EMBED_SIZE],
    int length,
    float scores[CONTEXT_SIZE][CONTEXT_SIZE])
{
    float inv_sqrt_embed = 1.0f / sqrtf((float)EMBED_SIZE);

    for (int i = 0; i < length; i++)
        for (int j = 0; j < length; j++)
        {
            if (j <= i)
                scores[i][j] = dot_avx(queries[i], keys[j], EMBED_SIZE) * inv_sqrt_embed;
            else
                scores[i][j] = -INFINITY;
        }
}

/* -------------------- Multi-head projections -------------------- */

void compute_query_head(Model *model, float input[EMBED_SIZE], int head, float output[HEAD_SIZE])
{
    vec_zero(output, HEAD_SIZE);
    for (int j = 0; j < EMBED_SIZE; j++)
        fma_scale_accumulate(output, model->Wq[head][j], input[j], HEAD_SIZE);
}

void compute_key_head(Model *model, float input[EMBED_SIZE], int head, float output[HEAD_SIZE])
{
    vec_zero(output, HEAD_SIZE);
    for (int j = 0; j < EMBED_SIZE; j++)
        fma_scale_accumulate(output, model->Wk[head][j], input[j], HEAD_SIZE);
}

void compute_value_head(Model *model, float input[EMBED_SIZE], int head, float output[HEAD_SIZE])
{
    vec_zero(output, HEAD_SIZE);
    for (int j = 0; j < EMBED_SIZE; j++)
        fma_scale_accumulate(output, model->Wv[head][j], input[j], HEAD_SIZE);
}

void compute_attention_scores_head(
    float queries[CONTEXT_SIZE][HEAD_SIZE],
    float keys[CONTEXT_SIZE][HEAD_SIZE],
    int length,
    float scores[CONTEXT_SIZE][CONTEXT_SIZE])
{
    float inv_sqrt_head = 1.0f / sqrtf((float)HEAD_SIZE);

    for (int i = 0; i < length; i++)
        for (int j = 0; j < length; j++)
        {
            if (j <= i)
                scores[i][j] = dot_avx(queries[i], keys[j], HEAD_SIZE) * inv_sqrt_head;
            else
                scores[i][j] = -INFINITY;
        }
}

void compute_attention_output_head(
    float scores[CONTEXT_SIZE][CONTEXT_SIZE],
    float values[CONTEXT_SIZE][HEAD_SIZE],
    int length,
    float output[CONTEXT_SIZE][HEAD_SIZE])
{
    for (int i = 0; i < length; i++)
    {
        vec_zero(output[i], HEAD_SIZE);
        for (int j = 0; j < length; j++)
            fma_scale_accumulate(output[i], values[j], scores[i][j], HEAD_SIZE);
    }
}

/* -------------------- Multi-head attention (with cache) -------------------- */

void multi_head_attention(
    Model *model,
    float input[CONTEXT_SIZE][EMBED_SIZE],
    int length,
    float output[CONTEXT_SIZE][EMBED_SIZE],
    ForwardCache *cache)
{
    float head_outputs[NUM_HEADS][CONTEXT_SIZE][HEAD_SIZE];
    float concatenated[CONTEXT_SIZE][EMBED_SIZE];

    if (cache != NULL) cache->length = length;

    for (int h = 0; h < NUM_HEADS; h++)
    {
        float queries[CONTEXT_SIZE][HEAD_SIZE];
        float keys[CONTEXT_SIZE][HEAD_SIZE];
        float values[CONTEXT_SIZE][HEAD_SIZE];
        float scores[CONTEXT_SIZE][CONTEXT_SIZE];

        for (int pos = 0; pos < length; pos++)
        {
            compute_query_head(model, input[pos], h, queries[pos]);
            compute_key_head(model, input[pos], h, keys[pos]);
            compute_value_head(model, input[pos], h, values[pos]);
        }

        compute_attention_scores_head(queries, keys, length, scores);

        if (cache != NULL)
            for (int i = 0; i < length; i++)
                for (int j = 0; j < length; j++)
                    cache->scores[h][i][j] = scores[i][j];

        for (int i = 0; i < length; i++)
            softmax_masked(scores[i], length);

        if (cache != NULL)
            for (int i = 0; i < length; i++)
                for (int j = 0; j < length; j++)
                    cache->weights[h][i][j] = scores[i][j];

        compute_attention_output_head(scores, values, length, head_outputs[h]);

        if (cache != NULL)
            for (int pos = 0; pos < length; pos++)
                for (int d = 0; d < HEAD_SIZE; d++)
                {
                    cache->queries[h][pos][d] = queries[pos][d];
                    cache->keys[h][pos][d] = keys[pos][d];
                    cache->values[h][pos][d] = values[pos][d];
                }
    }

    for (int pos = 0; pos < length; pos++)
        for (int h = 0; h < NUM_HEADS; h++)
            for (int d = 0; d < HEAD_SIZE; d++)
                concatenated[pos][h * HEAD_SIZE + d] = head_outputs[h][pos][d];

    if (cache != NULL)
        for (int pos = 0; pos < length; pos++)
            for (int i = 0; i < EMBED_SIZE; i++)
                cache->concatenated[pos][i] = concatenated[pos][i];

    for (int pos = 0; pos < length; pos++)
    {
        vec_zero(output[pos], EMBED_SIZE);
        for (int j = 0; j < EMBED_SIZE; j++)
            fma_scale_accumulate(output[pos], model->Wo[j], concatenated[pos][j], EMBED_SIZE);
    }

    if (cache != NULL)
        for (int pos = 0; pos < length; pos++)
            for (int i = 0; i < EMBED_SIZE; i++)
                cache->attention_output[pos][i] = output[pos][i];
}

/* -------------------- Layer norm -------------------- */

void layer_norm(
    float input[EMBED_SIZE],
    float gamma[EMBED_SIZE],
    float beta[EMBED_SIZE],
    float output[EMBED_SIZE])
{
    /* Vectorized mean reduction. */
    __m256 vsum = _mm256_setzero_ps();
    int i = 0;
    for (; i + 8 <= EMBED_SIZE; i += 8)
        vsum = _mm256_add_ps(vsum, _mm256_loadu_ps(&input[i]));
    float mean = hsum256_ps(vsum);
    for (; i < EMBED_SIZE; i++)
        mean += input[i];
    mean /= EMBED_SIZE;

    /* Vectorized variance reduction: sum((x - mean)^2) via FMA. */
    __m256 vmean = _mm256_set1_ps(mean);
    __m256 vvar = _mm256_setzero_ps();
    i = 0;
    for (; i + 8 <= EMBED_SIZE; i += 8)
    {
        __m256 diff = _mm256_sub_ps(_mm256_loadu_ps(&input[i]), vmean);
        vvar = _mm256_fmadd_ps(diff, diff, vvar);
    }
    float variance = hsum256_ps(vvar);
    for (; i < EMBED_SIZE; i++)
    {
        float diff = input[i] - mean;
        variance += diff * diff;
    }
    variance /= EMBED_SIZE;

    /* Vectorized elementwise combine: gamma * (x - mean) * inv_std + beta. */
    float inv_std = 1.0f / sqrtf(variance + 1e-5f);
    __m256 vinv_std = _mm256_set1_ps(inv_std);
    i = 0;
    for (; i + 8 <= EMBED_SIZE; i += 8)
    {
        __m256 vx = _mm256_loadu_ps(&input[i]);
        __m256 vg = _mm256_loadu_ps(&gamma[i]);
        __m256 vb = _mm256_loadu_ps(&beta[i]);
        __m256 t = _mm256_mul_ps(_mm256_sub_ps(vx, vmean), vinv_std);
        __m256 vout = _mm256_fmadd_ps(vg, t, vb);
        _mm256_storeu_ps(&output[i], vout);
    }
    for (; i < EMBED_SIZE; i++)
        output[i] = gamma[i] * (input[i] - mean) * inv_std + beta[i];
}

/* -------------------- Feed-forward -------------------- */

void feed_forward(Model *model, float input[EMBED_SIZE], float output[EMBED_SIZE])
{
    float hidden[FFN_SIZE];
    feed_forward_forward(model, input, output, hidden);
}

void feed_forward_forward(Model *model, float input[EMBED_SIZE], float output[EMBED_SIZE], float hidden_pre_relu[FFN_SIZE])
{
    /* First layer: y = x @ W1 + b1 via broadcast-FMA over FFN_SIZE. */
    for (int i = 0; i < FFN_SIZE; i++)
        hidden_pre_relu[i] = model->b1[i];

    for (int j = 0; j < EMBED_SIZE; j++)
        fma_scale_accumulate(hidden_pre_relu, model->W1[j], input[j], FFN_SIZE);

    /* ReLU computed once per hidden unit (not once per output element). */
    float activated[FFN_SIZE];
    int k = 0;
    __m256 vzero = _mm256_setzero_ps();
    for (; k + 8 <= FFN_SIZE; k += 8)
    {
        __m256 v = _mm256_loadu_ps(&hidden_pre_relu[k]);
        v = _mm256_max_ps(v, vzero);
        _mm256_storeu_ps(&activated[k], v);
    }
    for (; k < FFN_SIZE; k++)
        activated[k] = hidden_pre_relu[k] > 0.0f ? hidden_pre_relu[k] : 0.0f;

    /* Second layer: y = relu(h) @ W2 + b2 via broadcast-FMA over EMBED_SIZE. */
    for (int i = 0; i < EMBED_SIZE; i++)
        output[i] = model->b2[i];

    for (int j = 0; j < FFN_SIZE; j++)
        fma_scale_accumulate(output, model->W2[j], activated[j], EMBED_SIZE);
}

/* -------------------- Transformer block -------------------- */

void transformer_block(
    Model *model,
    float input[CONTEXT_SIZE][EMBED_SIZE],
    int length,
    float output[CONTEXT_SIZE][EMBED_SIZE],
    TransformerBlockCache *cache)
{
    float attention_output[CONTEXT_SIZE][EMBED_SIZE];
    float residual1[CONTEXT_SIZE][EMBED_SIZE];
    float norm1[CONTEXT_SIZE][EMBED_SIZE];
    float ff_output[CONTEXT_SIZE][EMBED_SIZE];
    float residual2[CONTEXT_SIZE][EMBED_SIZE];

    ForwardCache *attn_cache = (cache != NULL) ? &cache->attn_cache : NULL;
    multi_head_attention(model, input, length, attention_output, attn_cache);

    for (int pos = 0; pos < length; pos++)
        for (int i = 0; i < EMBED_SIZE; i++)
            residual1[pos][i] = input[pos][i] + attention_output[pos][i];

    for (int pos = 0; pos < length; pos++)
        layer_norm(residual1[pos], model->ln1_gamma, model->ln1_beta, norm1[pos]);

    for (int pos = 0; pos < length; pos++)
    {
        if (cache != NULL)
            feed_forward_forward(model, norm1[pos], ff_output[pos], cache->ffn_hidden[pos]);
        else
            feed_forward(model, norm1[pos], ff_output[pos]);
    }

    for (int pos = 0; pos < length; pos++)
        for (int i = 0; i < EMBED_SIZE; i++)
            residual2[pos][i] = norm1[pos][i] + ff_output[pos][i];

    for (int pos = 0; pos < length; pos++)
        layer_norm(residual2[pos], model->ln2_gamma, model->ln2_beta, output[pos]);

    if (cache != NULL)
    {
        cache->length = length;
        for (int pos = 0; pos < length; pos++)
            for (int i = 0; i < EMBED_SIZE; i++)
            {
                cache->input[pos][i] = input[pos][i];
                cache->residual1[pos][i] = residual1[pos][i];
                cache->norm1[pos][i] = norm1[pos][i];
                cache->residual2[pos][i] = residual2[pos][i];
                cache->output[pos][i] = output[pos][i];
            }
    }
}

/* -------------------- Logits & sampling -------------------- */

void compute_logits(Model *model, float hidden[EMBED_SIZE], float logits[VOCAB_SIZE])
{
    vec_zero(logits, VOCAB_SIZE);
    for (int i = 0; i < EMBED_SIZE; i++)
        fma_scale_accumulate(logits, model->output_projection[i], hidden[i], VOCAB_SIZE);
}

int sample_token(float logits[VOCAB_SIZE], float temperature)
{
    /* expf is scalar (no portable AVX2 exp); this loop dominates the cost
       here, so vectorizing the sum around it buys little. Normalization
       is skipped entirely: r < cumsum/sum  <=>  r*sum < cumsum. */
    float probs[VOCAB_SIZE];
    float sum = 0.0f;
    for (int i = 0; i < VOCAB_SIZE; i++)
    {
        probs[i] = expf(logits[i] / temperature);
        sum += probs[i];
    }

    float r = (float)rand() / (float)RAND_MAX;
    float target = r * sum;
    float cumsum = 0.0f;
    for (int i = 0; i < VOCAB_SIZE; i++)
    {
        cumsum += probs[i];
        if (target < cumsum)
            return i;
    }
    return VOCAB_SIZE - 1;
}

int sample_token_topk(float logits[VOCAB_SIZE], float temperature, int k)
{
    if (k > VOCAB_SIZE) k = VOCAB_SIZE;
    if (k <= 0) k = 1;

    int indices[VOCAB_SIZE];
    float logits_copy[VOCAB_SIZE];
    for (int i = 0; i < VOCAB_SIZE; i++) {
        indices[i] = i;
        logits_copy[i] = logits[i];
    }

    /* Partial selection sort: only the first k slots need to end up
       sorted descending, so the outer loop stops after k passes
       instead of VOCAB_SIZE - 1. O(n*k) instead of O(n^2) - a large
       win whenever k << VOCAB_SIZE. */
    for (int i = 0; i < k; i++) {
        int max_idx = i;
        for (int j = i + 1; j < VOCAB_SIZE; j++)
            if (logits_copy[j] > logits_copy[max_idx])
                max_idx = j;
        if (max_idx != i) {
            float tf = logits_copy[i];
            logits_copy[i] = logits_copy[max_idx];
            logits_copy[max_idx] = tf;
            int ti = indices[i];
            indices[i] = indices[max_idx];
            indices[max_idx] = ti;
        }
    }

    /* Compact size-k arrays instead of full VOCAB_SIZE probs: avoids the
       O(VOCAB_SIZE) zero-fill and the O(VOCAB_SIZE) scans the old
       normalize/sample loops did over mostly-zero entries. */
    float probs[k];
    float sum = 0.0f;
    for (int i = 0; i < k; i++) {
        probs[i] = expf(logits_copy[i] / temperature);
        sum += probs[i];
    }

    /* Same trick as sample_token(): r < cumsum/sum <=> r*sum < cumsum,
       so no separate normalization pass is needed. */
    float r = (float)rand() / (float)RAND_MAX;
    float target = r * sum;
    float cumsum = 0.0f;
    for (int i = 0; i < k; i++) {
        cumsum += probs[i];
        if (target < cumsum)
            return indices[i];
    }
    return indices[k - 1];
}