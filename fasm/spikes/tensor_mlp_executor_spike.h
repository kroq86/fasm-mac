#ifndef TENSOR_MLP_EXECUTOR_SPIKE_H
#define TENSOR_MLP_EXECUTOR_SPIKE_H
/* Experimental MLP block reusing the transformer's generic executor ABI;
 * not a stable core ABI. tensor_transformer_steps_execute (the assembly
 * executor) contains no transformer-specific logic: it is a 32-byte
 * {run, context, kind, flags} step interpreter. This block is the
 * generality test — a 2->4->1 XOR MLP, unrelated math to attention/
 * LayerNorm/FFN, driven through the exact same executor loop. */
#include <math.h>
#include <stdint.h>
#include <string.h>

enum { IN = 2, HID = 4, OUT = 1 };
typedef struct {
    float x[IN], w1[IN * HID], b1[HID], z1[HID], h1[HID], w2[HID * OUT], b2[OUT], out[OUT];
    float dx[IN], dw1[IN * HID], db1[HID], dw2[HID * OUT], db2[OUT];
} Block;

static void mlp_forward(Block *b) {
    for (int j = 0; j < HID; j++) {
        float s = b->b1[j];
        for (int i = 0; i < IN; i++) s += b->x[i] * b->w1[i * HID + j];
        b->z1[j] = s;
        b->h1[j] = s > 0 ? s : 0;
    }
    for (int k = 0; k < OUT; k++) {
        float s = b->b2[k];
        for (int j = 0; j < HID; j++) s += b->h1[j] * b->w2[j * OUT + k];
        b->out[k] = s;
    }
}

/* Monolithic reference backward (dL/dout given as seed), used both as a
 * finite-difference oracle and as the cross-check for the scheduled path. */
static void mlp_backward(Block *b, const float *seed) {
    memset(b->dx, 0, sizeof b->dx);
    memset(b->dw1, 0, sizeof b->dw1);
    memset(b->db1, 0, sizeof b->db1);
    memset(b->dw2, 0, sizeof b->dw2);
    memset(b->db2, 0, sizeof b->db2);
    float dh1[HID] = {0};
    for (int k = 0; k < OUT; k++) {
        b->db2[k] += seed[k];
        for (int j = 0; j < HID; j++) {
            b->dw2[j * OUT + k] += b->h1[j] * seed[k];
            dh1[j] += b->w2[j * OUT + k] * seed[k];
        }
    }
    float dz1[HID];
    for (int j = 0; j < HID; j++) dz1[j] = b->z1[j] > 0 ? dh1[j] : 0;
    for (int j = 0; j < HID; j++) {
        b->db1[j] += dz1[j];
        for (int i = 0; i < IN; i++) {
            b->dw1[i * HID + j] += b->x[i] * dz1[j];
            b->dx[i] += b->w1[i * HID + j] * dz1[j];
        }
    }
}

static float mlp_initv(int i, int s) { return (float)(((i * 37 + s * 17) % 29) - 14) / 41.0f; }

/* Same hand-picked starting weights as fasm/examples/xor_tensor_train.asm
 * (w1_data/b1_data/w2_data/b2_data), so a training-dynamics difference
 * can't be blamed on unlucky random init when cross-checking against it. */
static void mlp_initialize(Block *b) {
    memset(b, 0, sizeof *b);
    static const float w1[IN * HID] = {0.5f, -0.7f, 0.3f, 0.8f, -0.4f, 0.6f, 0.9f, -0.2f};
    static const float b1[HID] = {0.1f, 0.1f, -0.1f, 0.0f};
    static const float w2[HID * OUT] = {0.7f, -0.5f, 0.6f, -0.8f};
    memcpy(b->w1, w1, sizeof w1);
    memcpy(b->b1, b1, sizeof b1);
    memcpy(b->w2, w2, sizeof w2);
}

/* --- generic executor plumbing: same shapes as tensor_transformer_executor_spike.h --- */
typedef struct { int (*run)(void *); void *context; uint32_t kind, flags; uint64_t reserved; } ExecStep;
extern int tensor_transformer_steps_execute(const ExecStep *, uint64_t);

enum { ZERO_ACTION, REVERSE_ACTION };
typedef struct { float dh1[HID], dz1[HID]; } Scratch;
typedef struct { Block *b; Scratch *s; const float *seed; uint32_t kind, index, generation, *live_generation; } Context;

static int mlp_zero(Context *c) {
    switch (c->index) {
        case 0: memset(c->b->dx, 0, sizeof c->b->dx); break;
        case 1: memset(c->b->dw1, 0, sizeof c->b->dw1); break;
        case 2: memset(c->b->db1, 0, sizeof c->b->db1); break;
        case 3: memset(c->b->dw2, 0, sizeof c->b->dw2); break;
        case 4: memset(c->b->db2, 0, sizeof c->b->db2); break;
        default: return -1;
    }
    return 0;
}

static int mlp_reverse(Context *c) {
    Block *b = c->b;
    Scratch *s = c->s;
    switch (c->index) {
        case 0: /* output bias + matmul2 backward, produces dh1 */
            memset(s->dh1, 0, sizeof s->dh1);
            for (int k = 0; k < OUT; k++) {
                b->db2[k] += c->seed[k];
                for (int j = 0; j < HID; j++) {
                    b->dw2[j * OUT + k] += b->h1[j] * c->seed[k];
                    s->dh1[j] += b->w2[j * OUT + k] * c->seed[k];
                }
            }
            break;
        case 1: /* relu backward */
            for (int j = 0; j < HID; j++) s->dz1[j] = b->z1[j] > 0 ? s->dh1[j] : 0;
            break;
        case 2: /* hidden bias + matmul1 backward */
            for (int j = 0; j < HID; j++) {
                b->db1[j] += s->dz1[j];
                for (int i = 0; i < IN; i++) {
                    b->dw1[i * HID + j] += b->x[i] * s->dz1[j];
                    b->dx[i] += b->w1[i * HID + j] * s->dz1[j];
                }
            }
            break;
        default: return -1;
    }
    return 0;
}

static int mlp_execute(void *opaque) {
    Context *c = opaque;
    if (*c->live_generation != c->generation) return -4;
    if (c->kind == ZERO_ACTION) return mlp_zero(c);
    return mlp_reverse(c);
}

static unsigned mlp_emit(Block *b, Scratch *s, const float *seed, uint32_t *generation, ExecStep *steps, Context *ctx) {
    unsigned at = 0;
    for (unsigned i = 0; i < 5; i++) {
        ctx[at] = (Context){b, s, seed, ZERO_ACTION, i, *generation, generation};
        steps[at] = (ExecStep){mlp_execute, &ctx[at], ZERO_ACTION, 0, 0};
        at++;
    }
    for (unsigned i = 0; i < 3; i++) {
        ctx[at] = (Context){b, s, seed, REVERSE_ACTION, i, *generation, generation};
        steps[at] = (ExecStep){mlp_execute, &ctx[at], REVERSE_ACTION, 0, 0};
        at++;
    }
    return at;
}
#endif
