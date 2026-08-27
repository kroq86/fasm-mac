#define T 8
#define M 16
#define H 4
#define D 4
#define F 32
#define TRANSFORMER_EXECUTOR_NO_MAIN
#include "tensor_transformer_executor_spike.h"
#undef TRANSFORMER_EXECUTOR_NO_MAIN

#include <ctype.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum { CLASS_AUTH, CLASS_DATABASE, CLASS_CODE, CLASS_COUNT, MAX_WORDS = 2048, MAX_SAMPLES = 384 };

typedef struct { float x[T * M]; unsigned label; } Sample;

static uint32_t hash_word(const char *s) {
    uint32_t h = 2166136261u;
    while (*s) { h ^= (unsigned char)*s++; h *= 16777619u; }
    return h;
}

static void encode_word(const char *word, float *out) {
    uint32_t h = hash_word(word);
    for (unsigned j = 0; j < M; j++) {
        unsigned byte = (h >> (j * 8)) & 255u;
        out[j] = ((float)byte / 127.5f) - 1.0f;
    }
}

static unsigned read_words(const char *path, char words[MAX_WORDS][48]) {
    FILE *f = fopen(path, "rb");
    if (!f) return 0;
    unsigned count = 0, at = 0;
    int ch;
    char word[48];
    while ((ch = fgetc(f)) != EOF && count < MAX_WORDS) {
        if (isalnum((unsigned char)ch) || ch == '_') {
            if (at + 1 < sizeof word) word[at++] = (char)tolower((unsigned char)ch);
        } else if (at) {
            word[at] = 0;
            strcpy(words[count++], word);
            at = 0;
        }
    }
    if (at && count < MAX_WORDS) { word[at] = 0; strcpy(words[count++], word); }
    fclose(f);
    return count;
}

static unsigned add_document_samples(Sample *samples, unsigned count, const char *path, unsigned label) {
    char words[MAX_WORDS][48];
    unsigned n = read_words(path, words);
    if (n < T) return count;
    /* Overlapping windows preserve word order while the stride keeps the
       checked-in fixture bounded and makes every source contribute. */
    for (unsigned i = 0; i + T <= n && count < MAX_SAMPLES; i += 2) {
        Sample *s = &samples[count++];
        memset(s, 0, sizeof *s);
        s->label = label;
        for (unsigned t = 0; t < T; t++) encode_word(words[i + t], &s->x[t * M]);
    }
    return count;
}

static void make_target(unsigned label, float target[T * M]) {
    float code[M], mean = 0, variance = 0;
    for (unsigned j = 0; j < M; j++) {
        code[j] = sinf((float)(label + 1) * (j + 1) * 1.37f) + cosf((float)(label + 2) * (j + 1) * .73f);
        mean += code[j];
    }
    mean /= M;
    for (unsigned j = 0; j < M; j++) { float d = code[j] - mean; variance += d * d; }
    float inv = 1.0f / sqrtf(variance / M + 1e-5f);
    for (unsigned j = 0; j < M; j++) code[j] = (code[j] - mean) * inv;
    for (unsigned t = 0; t < T; t++) memcpy(&target[t * M], code, M * sizeof(float));
}

static float loss_seed(Block *b, unsigned label, float seed[T * M]) {
    float target[T * M], loss = 0;
    make_target(label, target);
    for (unsigned i = 0; i < T * M; i++) {
        float e = b->out[i] - target[i];
        loss += e * e;
        seed[i] = 2 * e / (T * M);
    }
    return loss / (T * M);
}

static unsigned predict(Block *b) {
    unsigned best = 0;
    float best_loss = INFINITY;
    for (unsigned label = 0; label < CLASS_COUNT; label++) {
        float target[T * M], loss = 0;
        make_target(label, target);
        for (unsigned i = 0; i < T * M; i++) { float e = b->out[i] - target[i]; loss += e * e; }
        if (loss < best_loss) { best_loss = loss; best = label; }
    }
    return best;
}

static unsigned evaluate_held_out(Block *b, const Sample *samples, unsigned count, int show) {
    static const char *names[] = {"auth", "database", "code"};
    unsigned correct = 0, shown = 0;
    for (unsigned i = 0; i < count; i++) {
        if (i % 5 != 0) continue;
        memcpy(b->x, samples[i].x, sizeof b->x);
        forward(b);
        unsigned got = predict(b);
        correct += got == samples[i].label;
        if (show && shown++ < 6) printf("held_out=%u expected=%s predicted=%s\n", i, names[samples[i].label], names[got]);
    }
    return correct;
}

int run_corpus(int argc, char **argv) {
    if (argc != 2) { fprintf(stderr, "usage: tensorctl corpus PATH_TO_TINY_REPO\n"); return 2; }
    char path[1024];
    Sample samples[MAX_SAMPLES];
    unsigned count = 0;
    snprintf(path, sizeof path, "%s/docs/auth.md", argv[1]);
    count = add_document_samples(samples, count, path, CLASS_AUTH);
    snprintf(path, sizeof path, "%s/docs/db.md", argv[1]);
    count = add_document_samples(samples, count, path, CLASS_DATABASE);
    snprintf(path, sizeof path, "%s/src/middleware.go", argv[1]);
    count = add_document_samples(samples, count, path, CLASS_CODE);
    if (count < 12) { fprintf(stderr, "tensorctl corpus: fixture is missing or too small\n"); return 2; }

    Block b;
    Scratch scratch = {0};
    float seed[T * M];
    initialize(&b, seed);
    uint32_t generation = 1;
    ExecStep steps[21];
    Context ctx[21];
    const unsigned epochs = 120;
    unsigned train_count = 0, test_count = 0;
    for (unsigned i = 0; i < count; i++) {
        if (i % 5) train_count++;
        else test_count++;
    }
    unsigned correct_before = evaluate_held_out(&b, samples, count, 0);
    for (unsigned epoch = 0; epoch < epochs; epoch++) {
        for (unsigned i = 0; i < count; i++) {
            unsigned selected = (i * 37u + epoch * 17u) % count;
            if (selected % 5 == 0) continue; /* deterministic held-out windows */
            Sample *s = &samples[selected];
            memcpy(b.x, s->x, sizeof b.x);
            forward(&b);
            (void)loss_seed(&b, s->label, seed);
            unsigned n = emit(&b, &scratch, seed, &generation, steps, ctx);
            if (n != 21 || tensor_transformer_steps_execute(steps, n)) return 3;
            const float lr = .0025f;
            for (int k = 0; k < M * QW; k++) b.wq[k] -= lr * b.dwq[k];
            for (int k = 0; k < M * M; k++) b.wo[k] -= lr * b.dwo[k];
            for (int k = 0; k < M * F; k++) b.w1[k] -= lr * b.dw1[k];
            for (int k = 0; k < F * M; k++) b.w2[k] -= lr * b.dw2[k];
        }
    }

    unsigned correct = evaluate_held_out(&b, samples, count, 1);
    float accuracy_before = test_count ? (float)correct_before / test_count : 0;
    float accuracy = test_count ? (float)correct / test_count : 0;
    printf("corpus: files=3 windows=%u train=%u test=%u transformer=T%d-M%d-H%d-D%d-F%d executor=x86_64-assembly epochs=%u accuracy=%.3f->%.3f (%u/%u)\n",
           count, train_count, test_count, T, M, H, D, F, epochs, accuracy_before, accuracy, correct, test_count);
    return accuracy >= .70f ? 0 : 1;
}
