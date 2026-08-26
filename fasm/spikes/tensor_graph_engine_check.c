/* Experimental generic graph-driven engine; not a stable core ABI.
 *
 * tensor_compiler_planner_check.c already proved a Node graph (matmul/bias/
 * relu/mse, MB/MBR fusion) is structurally sound and general enough to
 * describe an MLP — but it never executed anything: no data buffers, no
 * kernels, no backward, no executor. This connects that same op set to real
 * numbers: forward and backward are both dispatched generically by
 * node->op through a small table, not hand-written per model, and both
 * directions run through the exact same shared assembly executor
 * (tensor_transformer_steps_execute) that every other spike in this repo
 * uses. Fusing steps (the original spike's STEP_MB/STEP_MBR) is a natural
 * next refinement once this baseline is proven — deliberately out of scope
 * here so the core claim ("one engine, not per-model hand code") isn't
 * entangled with an optimization on top of it.
 *
 * Proof of "one engine, not MLP-code + Transformer-code": the exact 2-4-1
 * XOR MLP this repo already trains three other ways (fasm/examples/
 * xor_tensor_train.asm, tensor_mlp_train_check.c, tensorctl) is trained a
 * fourth time here with no forward()/backward() function at all for this
 * specific shape — only a Node array and a generic interpreter that would
 * describe any other matmul/bias/relu/mse graph identically.
 *
 * ExecStep is redeclared here (not included from tensor_mlp_executor_spike.h
 * or tensor_transformer_executor_spike.h) because those headers define
 * types with the same names for their own models — this file must not
 * share a translation unit with either. The struct layout is the ABI the
 * assembly executor actually reads; it must stay byte-identical.
 */
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

enum { LEAF, MATMUL, RELU, MSE, BIAS, OP_COUNT };
enum { PARAM = 1, CONSTANT = 2, INPUT = 4, TEMP = 8 };
#define NONE UINT32_MAX

typedef struct {
    uint32_t op, lhs, rhs, flags;
    uint32_t rows, cols; /* shape of this node's own value */
    uint8_t trainable;
    float *data, *grad; /* grad NULL means "no gradient needed for this node" */
} Node;

typedef struct { int (*run)(void *); void *context; uint32_t kind, flags; uint64_t reserved; } ExecStep;
extern int tensor_transformer_steps_execute(const ExecStep *, uint64_t);

/* --- generic kernels: dispatched by node->op, identical code path for any
 * graph built from this op set, not specific to this MLP's shape --- */
static void k_matmul_fwd(Node *self, Node *lhs, Node *rhs) {
    uint32_t r = lhs->rows, k = lhs->cols, c = rhs->cols;
    memset(self->data, 0, (size_t)r * c * sizeof(float));
    for (uint32_t i = 0; i < r; i++) for (uint32_t p = 0; p < k; p++) {
        float x = lhs->data[i * k + p];
        for (uint32_t j = 0; j < c; j++) self->data[i * c + j] += x * rhs->data[p * c + j];
    }
}
static void k_matmul_bwd(Node *self, Node *lhs, Node *rhs) {
    uint32_t r = lhs->rows, k = lhs->cols, c = rhs->cols;
    for (uint32_t i = 0; i < r; i++) for (uint32_t j = 0; j < c; j++) {
        float g = self->grad[i * c + j];
        for (uint32_t p = 0; p < k; p++) {
            if (lhs->grad) lhs->grad[i * k + p] += g * rhs->data[p * c + j];
            if (rhs->grad) rhs->grad[p * c + j] += g * lhs->data[i * k + p];
        }
    }
}
static void k_bias_fwd(Node *self, Node *lhs, Node *rhs) {
    for (uint32_t i = 0; i < self->rows; i++) for (uint32_t j = 0; j < self->cols; j++)
        self->data[i * self->cols + j] = lhs->data[i * self->cols + j] + rhs->data[j];
}
static void k_bias_bwd(Node *self, Node *lhs, Node *rhs) {
    for (uint32_t i = 0; i < self->rows; i++) for (uint32_t j = 0; j < self->cols; j++) {
        float g = self->grad[i * self->cols + j];
        if (lhs->grad) lhs->grad[i * self->cols + j] += g;
        if (rhs->grad) rhs->grad[j] += g;
    }
}
static void k_relu_fwd(Node *self, Node *lhs, Node *rhs) {
    (void)rhs;
    for (uint32_t i = 0; i < self->rows * self->cols; i++) self->data[i] = lhs->data[i] > 0 ? lhs->data[i] : 0;
}
static void k_relu_bwd(Node *self, Node *lhs, Node *rhs) {
    (void)rhs;
    for (uint32_t i = 0; i < self->rows * self->cols; i++) if (lhs->grad) lhs->grad[i] += lhs->data[i] > 0 ? self->grad[i] : 0;
}
static void k_mse_fwd(Node *self, Node *lhs, Node *rhs) {
    uint32_t n = lhs->rows * lhs->cols;
    float s = 0;
    for (uint32_t i = 0; i < n; i++) { float e = lhs->data[i] - rhs->data[i]; s += e * e; }
    self->data[0] = s / n;
}
static void k_mse_bwd(Node *self, Node *lhs, Node *rhs) {
    uint32_t n = lhs->rows * lhs->cols;
    float g = self->grad[0];
    for (uint32_t i = 0; i < n; i++) { float e = lhs->data[i] - rhs->data[i]; if (lhs->grad) lhs->grad[i] += g * 2 * e / n; }
}
typedef void (*KernelFn)(Node *, Node *, Node *);
static const KernelFn FWD[OP_COUNT] = {[MATMUL] = k_matmul_fwd, [BIAS] = k_bias_fwd, [RELU] = k_relu_fwd, [MSE] = k_mse_fwd};
static const KernelFn BWD[OP_COUNT] = {[MATMUL] = k_matmul_bwd, [BIAS] = k_bias_bwd, [RELU] = k_relu_bwd, [MSE] = k_mse_bwd};

/* --- executor plumbing: one context type for both zeroing and node
 * execution, dispatch entirely driven by node->op, not by which model
 * this graph happens to represent --- */
typedef struct { Node *nodes; uint32_t index, generation, *live_generation; int backward; } NodeCtx;
static int node_execute(void *opaque) {
    NodeCtx *c = opaque;
    if (*c->live_generation != c->generation) return -4;
    Node *n = &c->nodes[c->index];
    Node *lhs = &c->nodes[n->lhs];
    Node *rhs = n->rhs != NONE ? &c->nodes[n->rhs] : NULL;
    (c->backward ? BWD : FWD)[n->op](n, lhs, rhs);
    return 0;
}
typedef struct { float *grad; uint32_t count, generation, *live_generation; } ZeroCtx;
static int zero_execute(void *opaque) {
    ZeroCtx *c = opaque;
    if (*c->live_generation != c->generation) return -4;
    memset(c->grad, 0, (size_t)c->count * sizeof(float));
    return 0;
}

/* --- the MLP as a Node graph: same op sequence (matmul->bias->relu->
 * matmul->bias->mse) tensor_compiler_planner_check.c already validated
 * structurally, now with this repo's real 2-4-1 XOR shapes and data --- */
enum { N_X, N_W1, N_B1, N_W2, N_B2, N_TARGET, N_MM1, N_BIAS1, N_RELU1, N_MM2, N_BIAS2, N_LOSS, N_COUNT };
enum { IN = 2, HID = 4, OUT = 1 };

int main(void) {
    float x[IN], w1[IN * HID] = {0.5f, -0.7f, 0.3f, 0.8f, -0.4f, 0.6f, 0.9f, -0.2f}, b1[HID] = {0.1f, 0.1f, -0.1f, 0.0f};
    float w2[HID * OUT] = {0.7f, -0.5f, 0.6f, -0.8f}, b2[OUT] = {0}, target[OUT];
    float mm1[HID], bias1[HID], relu1[HID], mm2[OUT], bias2[OUT], loss[1];
    float gw1[IN * HID], gb1[HID], gw2[HID * OUT], gb2[OUT];
    float gmm1[HID], gbias1[HID], grelu1[HID], gmm2[OUT], gbias2[OUT], gloss[1];

    Node nodes[N_COUNT] = {
        [N_X] = {LEAF, NONE, NONE, INPUT, 1, IN, 0, x, NULL}, /* d(loss)/d(x) isn't needed for training, so no grad buffer at all */
        [N_W1] = {LEAF, NONE, NONE, PARAM, IN, HID, 1, w1, gw1},
        [N_B1] = {LEAF, NONE, NONE, PARAM, 1, HID, 1, b1, gb1},
        [N_W2] = {LEAF, NONE, NONE, PARAM, HID, OUT, 1, w2, gw2},
        [N_B2] = {LEAF, NONE, NONE, PARAM, 1, OUT, 1, b2, gb2},
        [N_TARGET] = {LEAF, NONE, NONE, CONSTANT, 1, OUT, 0, target, NULL},
        [N_MM1] = {MATMUL, N_X, N_W1, TEMP, 1, HID, 0, mm1, gmm1},
        [N_BIAS1] = {BIAS, N_MM1, N_B1, TEMP, 1, HID, 0, bias1, gbias1},
        [N_RELU1] = {RELU, N_BIAS1, NONE, TEMP, 1, HID, 0, relu1, grelu1},
        [N_MM2] = {MATMUL, N_RELU1, N_W2, TEMP, 1, OUT, 0, mm2, gmm2},
        [N_BIAS2] = {BIAS, N_MM2, N_B2, TEMP, 1, OUT, 0, bias2, gbias2},
        [N_LOSS] = {MSE, N_BIAS2, N_TARGET, TEMP, 1, 1, 0, loss, gloss},
    };

    /* --- forward+backward, both generically executed --- */
    static const float data[4][IN] = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};
    static const float labels[4] = {0, 1, 1, 0};
    uint32_t generation = 1;
    ExecStep steps[16];
    NodeCtx fctx[6], bctx[6];
    ZeroCtx zctx[9];
    /* Every node with a grad buffer that backward *accumulates into* (+=)
     * must be zeroed before each step, not just the four trainable params
     * — the five intermediate carriers (MM1/BIAS1/RELU1/MM2/BIAS2) would
     * otherwise accumulate across every previous training step forever.
     * (This is exactly the bug this file hit on the first run: predictions
     * diverged to a constant ~-12.9 because those five were never reset.) */
    Node *zeroable[9] = {&nodes[N_W1], &nodes[N_B1], &nodes[N_W2], &nodes[N_B2],
                          &nodes[N_MM1], &nodes[N_BIAS1], &nodes[N_RELU1], &nodes[N_MM2], &nodes[N_BIAS2]};
    unsigned forward_nodes[6] = {N_MM1, N_BIAS1, N_RELU1, N_MM2, N_BIAS2, N_LOSS};

    /* --- correctness first: finite-difference oracle on this exact
     * generic engine, same pattern every other spike in this repo uses --- */
    x[0] = 0; x[1] = 1; target[0] = 1;
    unsigned at = 0;
    for (unsigned i = 0; i < 9; i++) { zctx[i] = (ZeroCtx){zeroable[i]->grad, zeroable[i]->rows * zeroable[i]->cols, generation, &generation}; steps[at] = (ExecStep){zero_execute, &zctx[i], 0, 0, 0}; at++; }
    for (unsigned i = 0; i < 6; i++) { fctx[i] = (NodeCtx){nodes, forward_nodes[i], generation, &generation, 0}; steps[at] = (ExecStep){node_execute, &fctx[i], 0, 0, 0}; at++; }
    if (tensor_transformer_steps_execute(steps, at)) return 1;
    nodes[N_LOSS].grad[0] = 1.0f;
    unsigned bat = 0;
    ExecStep bsteps[6];
    for (int i = 5; i >= 0; i--) { bctx[bat] = (NodeCtx){nodes, forward_nodes[i], generation, &generation, 1}; bsteps[bat] = (ExecStep){node_execute, &bctx[bat], 0, 0, 0}; bat++; }
    if (tensor_transformer_steps_execute(bsteps, bat)) return 2;

    float eps = 1e-3f, old = w1[3], analytic = gw1[3];
    w1[3] = old + eps;
    for (unsigned i = 0; i < 6; i++) FWD[nodes[forward_nodes[i]].op](&nodes[forward_nodes[i]], &nodes[nodes[forward_nodes[i]].lhs], nodes[forward_nodes[i]].rhs != NONE ? &nodes[nodes[forward_nodes[i]].rhs] : NULL);
    float plus = loss[0];
    w1[3] = old - eps;
    for (unsigned i = 0; i < 6; i++) FWD[nodes[forward_nodes[i]].op](&nodes[forward_nodes[i]], &nodes[nodes[forward_nodes[i]].lhs], nodes[forward_nodes[i]].rhs != NONE ? &nodes[nodes[forward_nodes[i]].rhs] : NULL);
    float minus = loss[0];
    w1[3] = old;
    float numeric = (plus - minus) / (2 * eps);
    if (fabsf(numeric - analytic) > 5e-3f * fmaxf(1, fmaxf(fabsf(numeric), fabsf(analytic)))) {
        fprintf(stderr, "graph engine backward mismatch: analytic=%g numeric=%g\n", analytic, numeric);
        return 3;
    }

    /* --- train the exact XOR problem this repo already trains three other
     * ways, this time with zero hand-written forward()/backward() for this
     * shape --- */
    memcpy(w1, (float[]){0.5f, -0.7f, 0.3f, 0.8f, -0.4f, 0.6f, 0.9f, -0.2f}, sizeof w1);
    memcpy(b1, (float[]){0.1f, 0.1f, -0.1f, 0.0f}, sizeof b1);
    memcpy(w2, (float[]){0.7f, -0.5f, 0.6f, -0.8f}, sizeof w2);
    b2[0] = 0;
    float lr = 0.1f;
    generation = 100;
    for (unsigned epoch = 0; epoch < 4000; epoch++) {
        for (int p = 0; p < 4; p++) {
            memcpy(x, data[p], sizeof x);
            target[0] = labels[p];
            unsigned n = 0;
            for (unsigned i = 0; i < 9; i++) { zctx[i] = (ZeroCtx){zeroable[i]->grad, zeroable[i]->rows * zeroable[i]->cols, generation, &generation}; steps[n] = (ExecStep){zero_execute, &zctx[i], 0, 0, 0}; n++; }
            for (unsigned i = 0; i < 6; i++) { fctx[i] = (NodeCtx){nodes, forward_nodes[i], generation, &generation, 0}; steps[n] = (ExecStep){node_execute, &fctx[i], 0, 0, 0}; n++; }
            if (n != 15 || tensor_transformer_steps_execute(steps, n)) return 4;
            nodes[N_LOSS].grad[0] = 1.0f;
            unsigned bn = 0;
            for (int i = 5; i >= 0; i--) { bctx[bn] = (NodeCtx){nodes, forward_nodes[i], generation, &generation, 1}; bsteps[bn] = (ExecStep){node_execute, &bctx[bn], 0, 0, 0}; bn++; }
            if (bn != 6 || tensor_transformer_steps_execute(bsteps, bn)) return 5;
            for (int i = 0; i < IN * HID; i++) w1[i] -= lr * gw1[i];
            for (int i = 0; i < HID; i++) b1[i] -= lr * gb1[i];
            for (int i = 0; i < HID * OUT; i++) w2[i] -= lr * gw2[i];
            b2[0] -= lr * gb2[0];
        }
    }
    float predictions[4];
    int correct = 0;
    for (int p = 0; p < 4; p++) {
        memcpy(x, data[p], sizeof x);
        for (unsigned i = 0; i < 6; i++) FWD[nodes[forward_nodes[i]].op](&nodes[forward_nodes[i]], &nodes[nodes[forward_nodes[i]].lhs], nodes[forward_nodes[i]].rhs != NONE ? &nodes[nodes[forward_nodes[i]].rhs] : NULL);
        predictions[p] = bias2[0];
        correct += (predictions[p] >= .5f) == (labels[p] >= .5f);
    }
    printf("tensor graph engine passed: nodes=%u ops=matmul,bias,relu,mse dispatch=generic(by node->op) "
           "gradient_check=finite-difference task=xor epochs=4000 predictions=%.3f,%.3f,%.3f,%.3f expected=0,1,1,0 correct=%d/4 "
           "via_shared_executor=tensor_transformer_steps_execute\n",
           (unsigned)N_COUNT, predictions[0], predictions[1], predictions[2], predictions[3], correct);
    return correct == 4 ? 0 : 6;
}
