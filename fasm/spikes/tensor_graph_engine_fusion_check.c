/* Experimental generic fusion optimization pass; not a stable core ABI.
 *
 * The question this answers: can one generic optimization pass see
 * MATMUL->BIAS->RELU and replace it with a more efficient fused schedule,
 * regardless of which model that pattern occurs in — or does the compiler
 * itself need model-specific rules the moment it tries to optimize, not
 * just execute?
 *
 * consumers_of()/the MBR/MB grouping rule below are tensor_compiler_
 * planner_check.c's exact lowering logic (matmul followed by bias with
 * exactly one consumer, optionally followed by relu with exactly one
 * consumer), now applied to tensor_graph_engine_check.c's real MLP graph
 * with real numbers instead of that spike's structural-only Node array.
 * fused_execute() doesn't reimplement matmul+bias+relu — it just calls the
 * existing, separately-verified k_matmul/k_bias/k_relu kernels from within
 * one ExecStep instead of three, so correctness follows from kernels
 * already proven correct, not from new fused math.
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
    uint32_t rows, cols;
    uint8_t trainable;
    float *data, *grad;
} Node;

typedef struct { int (*run)(void *); void *context; uint32_t kind, flags; uint64_t reserved; } ExecStep;
extern int tensor_transformer_steps_execute(const ExecStep *, uint64_t);

/* --- same generic per-op kernels as tensor_graph_engine_check.c, verbatim --- */
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

/* --- the generic optimization pass: model-independent, only reads op/lhs/
 * rhs and how many other nodes consume each node --- */
static unsigned consumers_of(const Node *nodes, unsigned total, unsigned needle) {
    unsigned users = 0;
    for (unsigned i = 0; i < total; i++) {
        if (nodes[i].op == LEAF) continue;
        users += nodes[i].lhs == needle;
        users += nodes[i].rhs != NONE && nodes[i].rhs == needle;
    }
    return users;
}
enum { GROUP_PLAIN, GROUP_MB, GROUP_MBR };
typedef struct { uint8_t kind; uint32_t matmul, bias, relu, plain; } Group;
static unsigned build_schedule(Node *nodes, unsigned total, const unsigned *order, unsigned n, Group *groups) {
    unsigned g = 0, i = 0;
    while (i < n) {
        unsigned idx = order[i];
        if (nodes[idx].op == MATMUL && i + 1 < n && nodes[order[i + 1]].op == BIAS && nodes[order[i + 1]].lhs == idx && consumers_of(nodes, total, idx) == 1) {
            unsigned bias_idx = order[i + 1];
            if (i + 2 < n && nodes[order[i + 2]].op == RELU && nodes[order[i + 2]].lhs == bias_idx && consumers_of(nodes, total, bias_idx) == 1) {
                groups[g++] = (Group){GROUP_MBR, idx, bias_idx, order[i + 2], NONE};
                i += 3;
                continue;
            }
            groups[g++] = (Group){GROUP_MB, idx, bias_idx, NONE, NONE};
            i += 2;
            continue;
        }
        groups[g++] = (Group){GROUP_PLAIN, NONE, NONE, NONE, idx};
        i += 1;
    }
    return g;
}

/* --- executor plumbing: one fused ExecStep replaces 2-3 plain ones. The
 * "fusion" is schedule size, not new math — fused_execute just calls the
 * same per-op kernels the plain path calls, from inside one dispatch. --- */
typedef struct { Node *nodes; uint32_t index, generation, *live_generation; int backward; } PlainCtx;
static int plain_execute(void *opaque) {
    PlainCtx *c = opaque;
    if (*c->live_generation != c->generation) return -4;
    Node *n = &c->nodes[c->index];
    Node *lhs = &c->nodes[n->lhs];
    Node *rhs = n->rhs != NONE ? &c->nodes[n->rhs] : NULL;
    (c->backward ? BWD : FWD)[n->op](n, lhs, rhs);
    return 0;
}
typedef struct { Node *nodes; uint32_t matmul, bias, relu, generation, *live_generation; int backward; } FusedCtx;
static int fused_execute(void *opaque) {
    FusedCtx *c = opaque;
    if (*c->live_generation != c->generation) return -4;
    Node *nodes = c->nodes;
    Node *mm = &nodes[c->matmul], *mm_lhs = &nodes[mm->lhs], *mm_rhs = &nodes[mm->rhs];
    Node *bias = &nodes[c->bias], *bias_rhs = &nodes[bias->rhs];
    Node *relu = c->relu != NONE ? &nodes[c->relu] : NULL;
    if (!c->backward) {
        FWD[MATMUL](mm, mm_lhs, mm_rhs);
        FWD[BIAS](bias, mm, bias_rhs);
        if (relu) FWD[RELU](relu, bias, NULL);
    } else {
        if (relu) BWD[RELU](relu, bias, NULL);
        BWD[BIAS](bias, mm, bias_rhs);
        BWD[MATMUL](mm, mm_lhs, mm_rhs);
    }
    return 0;
}
typedef struct { float *grad; uint32_t count, generation, *live_generation; } ZeroCtx;
static int zero_execute(void *opaque) {
    ZeroCtx *c = opaque;
    if (*c->live_generation != c->generation) return -4;
    memset(c->grad, 0, (size_t)c->count * sizeof(float));
    return 0;
}

/* --- the exact same MLP graph as tensor_graph_engine_check.c --- */
enum { N_X, N_W1, N_B1, N_W2, N_B2, N_TARGET, N_MM1, N_BIAS1, N_RELU1, N_MM2, N_BIAS2, N_LOSS, N_COUNT };
enum { IN = 2, HID = 4, OUT = 1 };

int main(void) {
    float x[IN], w1[IN * HID] = {0.5f, -0.7f, 0.3f, 0.8f, -0.4f, 0.6f, 0.9f, -0.2f}, b1[HID] = {0.1f, 0.1f, -0.1f, 0.0f};
    float w2[HID * OUT] = {0.7f, -0.5f, 0.6f, -0.8f}, b2[OUT] = {0}, target[OUT];
    float mm1[HID], bias1[HID], relu1[HID], mm2[OUT], bias2[OUT], loss[1];
    float gw1[IN * HID], gb1[HID], gw2[HID * OUT], gb2[OUT];
    float gmm1[HID], gbias1[HID], grelu1[HID], gmm2[OUT], gbias2[OUT], gloss[1];

    Node nodes[N_COUNT] = {
        [N_X] = {LEAF, NONE, NONE, INPUT, 1, IN, 0, x, NULL},
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
    unsigned forward_nodes[6] = {N_MM1, N_BIAS1, N_RELU1, N_MM2, N_BIAS2, N_LOSS};
    Node *zeroable[9] = {&nodes[N_W1], &nodes[N_B1], &nodes[N_W2], &nodes[N_B2],
                          &nodes[N_MM1], &nodes[N_BIAS1], &nodes[N_RELU1], &nodes[N_MM2], &nodes[N_BIAS2]};

    /* --- the optimization pass itself: same code, applied once --- */
    Group groups[6];
    unsigned ng = build_schedule(nodes, N_COUNT, forward_nodes, 6, groups);
    /* Cross-check against tensor_compiler_planner_check.c's own locked
     * assertion on this exact op sequence: steps=3, MBR then MB. */
    if (ng != 3 || groups[0].kind != GROUP_MBR || groups[1].kind != GROUP_MB || groups[2].kind != GROUP_PLAIN) {
        fprintf(stderr, "fusion pass disagreed with tensor_compiler_planner_check.c's locked schedule: ng=%u kinds=%d,%d,%d\n",
                ng, groups[0].kind, ng > 1 ? groups[1].kind : -1, ng > 2 ? groups[2].kind : -1);
        return 1;
    }

    /* --- build fused forward+backward ExecSteps from the detected groups --- */
    uint32_t generation = 1;
    ExecStep steps[9 + 3], bsteps[3];
    PlainCtx pctx_f[3], pctx_b[3];
    FusedCtx fctx_f[3], fctx_b[3];

    unsigned run_forward = 0, run_backward = 0; /* filled below, reused per training step */
#define BUILD_FORWARD() do { \
        run_forward = 0; \
        for (unsigned i = 0; i < 9; i++) { \
            ZeroCtx zc = {zeroable[i]->grad, zeroable[i]->rows * zeroable[i]->cols, generation, &generation}; \
            static ZeroCtx zctx_storage[9]; zctx_storage[i] = zc; \
            steps[run_forward] = (ExecStep){zero_execute, &zctx_storage[i], 0, 0, 0}; run_forward++; \
        } \
        for (unsigned gi = 0; gi < ng; gi++) { \
            if (groups[gi].kind == GROUP_PLAIN) { \
                pctx_f[gi] = (PlainCtx){nodes, groups[gi].plain, generation, &generation, 0}; \
                steps[run_forward] = (ExecStep){plain_execute, &pctx_f[gi], 0, 0, 0}; \
            } else { \
                fctx_f[gi] = (FusedCtx){nodes, groups[gi].matmul, groups[gi].bias, groups[gi].relu, generation, &generation, 0}; \
                steps[run_forward] = (ExecStep){fused_execute, &fctx_f[gi], 0, 0, 0}; \
            } \
            run_forward++; \
        } \
    } while (0)
#define BUILD_BACKWARD() do { \
        run_backward = 0; \
        for (int gi = (int)ng - 1; gi >= 0; gi--) { \
            unsigned bi = (unsigned)((int)ng - 1 - gi); \
            if (groups[gi].kind == GROUP_PLAIN) { \
                pctx_b[bi] = (PlainCtx){nodes, groups[gi].plain, generation, &generation, 1}; \
                bsteps[bi] = (ExecStep){plain_execute, &pctx_b[bi], 0, 0, 0}; \
            } else { \
                fctx_b[bi] = (FusedCtx){nodes, groups[gi].matmul, groups[gi].bias, groups[gi].relu, generation, &generation, 1}; \
                bsteps[bi] = (ExecStep){fused_execute, &fctx_b[bi], 0, 0, 0}; \
            } \
            run_backward++; \
        } \
    } while (0)

    /* --- correctness: finite differences on the fused schedule itself --- */
    x[0] = 0; x[1] = 1; target[0] = 1;
    BUILD_FORWARD();
    if (run_forward != 9 + 3 || tensor_transformer_steps_execute(steps, run_forward)) return 2;
    nodes[N_LOSS].grad[0] = 1.0f;
    BUILD_BACKWARD();
    if (run_backward != 3 || tensor_transformer_steps_execute(bsteps, run_backward)) return 3;

    float eps = 1e-3f, old = w1[3], analytic = gw1[3];
    w1[3] = old + eps;
    BUILD_FORWARD();
    tensor_transformer_steps_execute(steps, run_forward);
    float plus = loss[0];
    w1[3] = old - eps;
    BUILD_FORWARD();
    tensor_transformer_steps_execute(steps, run_forward);
    float minus = loss[0];
    w1[3] = old;
    float numeric = (plus - minus) / (2 * eps);
    if (fabsf(numeric - analytic) > 5e-3f * fmaxf(1, fmaxf(fabsf(numeric), fabsf(analytic)))) {
        fprintf(stderr, "fused schedule backward mismatch: analytic=%g numeric=%g\n", numeric, analytic);
        return 4;
    }

    /* --- train the same XOR problem, same init, through the fused (9+3
     * action) schedule instead of the unfused (9+6) one --- */
    memcpy(w1, (float[]){0.5f, -0.7f, 0.3f, 0.8f, -0.4f, 0.6f, 0.9f, -0.2f}, sizeof w1);
    memcpy(b1, (float[]){0.1f, 0.1f, -0.1f, 0.0f}, sizeof b1);
    memcpy(w2, (float[]){0.7f, -0.5f, 0.6f, -0.8f}, sizeof w2);
    b2[0] = 0;
    static const float data[4][IN] = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};
    static const float labels[4] = {0, 1, 1, 0};
    float lr = 0.1f;
    generation = 100;
    for (unsigned epoch = 0; epoch < 4000; epoch++) {
        for (int p = 0; p < 4; p++) {
            memcpy(x, data[p], sizeof x);
            target[0] = labels[p];
            BUILD_FORWARD();
            if (run_forward != 12 || tensor_transformer_steps_execute(steps, run_forward)) return 5;
            nodes[N_LOSS].grad[0] = 1.0f;
            BUILD_BACKWARD();
            if (run_backward != 3 || tensor_transformer_steps_execute(bsteps, run_backward)) return 6;
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
        for (unsigned gi = 0; gi < ng; gi++) {
            if (groups[gi].kind == GROUP_PLAIN) { Node *n = &nodes[groups[gi].plain]; FWD[n->op](n, &nodes[n->lhs], n->rhs != NONE ? &nodes[n->rhs] : NULL); }
            else { fused_execute(&(FusedCtx){nodes, groups[gi].matmul, groups[gi].bias, groups[gi].relu, generation, &generation, 0}); }
        }
        predictions[p] = bias2[0];
        correct += (predictions[p] >= .5f) == (labels[p] >= .5f);
    }

    printf("tensor graph engine fusion passed: pass=generic(matmul+bias[+relu], consumer-count==1) "
           "unfused_forward_actions=6 fused_forward_actions=%u groups=MBR,MB,PLAIN "
           "matches_tensor_compiler_planner_check=yes gradient_check=finite-difference "
           "task=xor epochs=4000 predictions=%.3f,%.3f,%.3f,%.3f expected=0,1,1,0 correct=%d/4\n",
           ng, predictions[0], predictions[1], predictions[2], predictions[3], correct);
    return correct == 4 ? 0 : 7;
}
