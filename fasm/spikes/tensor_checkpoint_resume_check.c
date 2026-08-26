#include <math.h>
#include <stdio.h>
#include <string.h>

enum { N = 6, STEPS = 80, SPLIT = 31 };
static const float xs[N] = {-2.f, -1.f, -.25f, .5f, 1.5f, 3.f};
static const float ys[N] = {-3.8f, -1.9f, -.475f, 1.05f, 3.15f, 6.2f};

typedef struct { float w, b, vw, vb; } State;

static void step(State *s, unsigned epoch, float momentum) {
    float gw = 0, gb = 0;
    /* Epoch-dependent rotation proves that the resume point is part of the
       execution state even though this full-batch sum is deterministic. */
    for (unsigned k = 0; k < N; k++) {
        unsigned i = (k + epoch) % N;
        float error = s->w * xs[i] + s->b - ys[i];
        gw += error * xs[i]; gb += error;
    }
    gw /= N; gb /= N;
    s->vw = momentum * s->vw + gw;
    s->vb = momentum * s->vb + gb;
    s->w -= .08f * s->vw; s->b -= .08f * s->vb;
}

static void train(State *s, unsigned begin, unsigned end, float momentum) {
    for (unsigned epoch = begin; epoch < end; epoch++) step(s, epoch, momentum);
}

static int same(const State *a, const State *b) { return memcmp(a, b, sizeof *a) == 0; }
static int same_params(const State *a, const State *b) { return a->w == b->w && a->b == b->b; }

int main(void) {
    State plain_full = {.w=.2f,.b=-.1f}, plain_split = plain_full;
    train(&plain_full, 0, STEPS, 0);
    train(&plain_split, 0, SPLIT, 0);
    State plain_loaded = {.w=plain_split.w,.b=plain_split.b};
    train(&plain_loaded, SPLIT, STEPS, 0);
    if (!same_params(&plain_full, &plain_loaded)) return 1;

    State momentum_full = {.w=.2f,.b=-.1f}, momentum_split = momentum_full;
    train(&momentum_full, 0, STEPS, .85f);
    train(&momentum_split, 0, SPLIT, .85f);
    State params_only = {.w=momentum_split.w,.b=momentum_split.b};
    train(&params_only, SPLIT, STEPS, .85f);
    if (same_params(&momentum_full, &params_only)) return 2;

    State full_state = momentum_split;
    train(&full_state, SPLIT, STEPS, .85f);
    if (!same(&momentum_full, &full_state)) return 3;

    printf("tensor checkpoint resume spike passed: plain_sgd=params-only-exact momentum=params-only-diverges momentum+velocity=exact step=%u required_state=params,optimizer,epoch\n", SPLIT);
    return 0;
}
