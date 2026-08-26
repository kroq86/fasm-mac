/* Experimental generic graph-driven engine, extended to the Transformer
 * block; not a stable core ABI.
 *
 * tensor_graph_engine_check.c proved the generic op-dispatch engine on an
 * MLP. This is the real test of the abstraction, not "add Transformer
 * support": the same generic dispatch-by-node->op mechanism, extended with
 * four new op traits (ATTENTION, RESIDUAL, LAYERNORM, CONTIGUOUS) that this
 * block's real math needs, with a hard rule — no transformer_backward_emit()
 * or any transformer-specific compiler path. MATMUL, BIAS, and RELU are
 * reused completely unchanged from the MLP engine; only the four new ops
 * are new code, and they're dispatched through the exact same table.
 *
 * Same real shape as every other transformer spike in this repo: T=3
 * tokens, M=4 model width, H=2 heads, D=2 head dim, F=6 FFN width,
 * QW=3*M=12. The real Block (tensor_transformer_reference_spike.h) has no
 * bias terms in the FFN — just matmul->relu->matmul — so this graph
 * doesn't have BIAS nodes anywhere; the kernel still exists and is proven
 * separately by the MLP engine, it's simply unused by this particular
 * graph, which is itself part of the point: the engine doesn't care which
 * ops a given graph happens to use.
 */
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

enum { LEAF, MATMUL, RELU, MSE, BIAS, ATTENTION, RESIDUAL, LAYERNORM, CONTIGUOUS, OP_COUNT };
enum { PARAM = 1, CONSTANT = 2, INPUT = 4, TEMP = 8 };
#define NONE UINT32_MAX
enum { T = 3, M = 4, H = 2, D = 2, F = 6, QW = 3 * M };

typedef struct {
    uint32_t op, lhs, rhs, flags;
    uint32_t rows, cols;      /* shape of this node's own value */
    uint8_t trainable;
    float *data, *grad;       /* grad NULL means "no gradient needed for this node" */
    float *aux;                /* op-specific saved forward state for backward (attention: prob; layernorm: mean+invstd) */
} Node;

typedef struct { int (*run)(void *); void *context; uint32_t kind, flags; uint64_t reserved; } ExecStep;
extern int tensor_transformer_steps_execute(const ExecStep *, uint64_t);

static float initv(int i, int s) { return (float)(((i * 37 + s * 17) % 29) - 14) / 41.0f; }

/* --- MATMUL/BIAS/RELU/MSE: unchanged from tensor_graph_engine_check.c,
 * generalized over rows/cols already, so no changes were needed to reuse
 * them for this block's [T,*] shapes instead of the MLP's [1,*] shapes. --- */
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

/* --- new: RESIDUAL, same-shape elementwise add, no broadcast (unlike BIAS) --- */
static void k_residual_fwd(Node *self, Node *lhs, Node *rhs) {
    uint32_t n = self->rows * self->cols;
    for (uint32_t i = 0; i < n; i++) self->data[i] = lhs->data[i] + rhs->data[i];
}
static void k_residual_bwd(Node *self, Node *lhs, Node *rhs) {
    uint32_t n = self->rows * self->cols;
    for (uint32_t i = 0; i < n; i++) {
        float g = self->grad[i];
        if (lhs->grad) lhs->grad[i] += g;
        if (rhs->grad) rhs->grad[i] += g;
    }
}

/* --- new: LAYERNORM, per-row normalize over `cols` columns. aux holds
 * mean[rows] then invstd[rows], saved for backward — same formulas as
 * tensor_transformer_reference_spike.h's ln()/lnback(). --- */
static void k_layernorm_fwd(Node *self, Node *lhs, Node *rhs) {
    (void)rhs;
    float *mean = self->aux, *invstd = self->aux + self->rows;
    for (uint32_t i = 0; i < self->rows; i++) {
        float m = 0, v = 0;
        for (uint32_t j = 0; j < self->cols; j++) m += lhs->data[i * self->cols + j];
        m /= self->cols;
        for (uint32_t j = 0; j < self->cols; j++) { float d = lhs->data[i * self->cols + j] - m; v += d * d; }
        mean[i] = m;
        invstd[i] = 1.0f / sqrtf(v / self->cols + 1e-5f);
        for (uint32_t j = 0; j < self->cols; j++) self->data[i * self->cols + j] = (lhs->data[i * self->cols + j] - m) * invstd[i];
    }
}
static void k_layernorm_bwd(Node *self, Node *lhs, Node *rhs) {
    (void)rhs;
    if (!lhs->grad) return;
    float *mean = self->aux, *invstd = self->aux + self->rows;
    for (uint32_t i = 0; i < self->rows; i++) {
        float sum = 0, sumh = 0;
        for (uint32_t j = 0; j < self->cols; j++) {
            float hh = (lhs->data[i * self->cols + j] - mean[i]) * invstd[i];
            sum += self->grad[i * self->cols + j];
            sumh += self->grad[i * self->cols + j] * hh;
        }
        for (uint32_t j = 0; j < self->cols; j++) {
            float hh = (lhs->data[i * self->cols + j] - mean[i]) * invstd[i];
            lhs->grad[i * self->cols + j] += invstd[i] * (self->grad[i * self->cols + j] - (sum + hh * sumh) / self->cols);
        }
    }
}

/* --- new: ATTENTION, multi-head scaled dot-product attention over a
 * packed [T,QW] Q|K|V buffer (QW=3*M). aux holds prob[H*T*T], saved for
 * backward. Output is head[H*T*D] (H*D==M by construction). Same formulas
 * as tensor_transformer_reference_spike.h's forward()/reverse_one() case 9,
 * generalized only in that they're reached via node->op, not a hand-coded
 * case number in a model-specific switch. --- */
static void k_attention_fwd(Node *self, Node *lhs, Node *rhs) {
    (void)rhs;
    const float *qkv = lhs->data;
    float *prob = self->aux; /* [H*T*T] */
    float scale = 1.0f / sqrtf((float)D);
    for (int h = 0; h < H; h++) for (int i = 0; i < T; i++) {
        float score[T], mx = -INFINITY;
        for (int j = 0; j < T; j++) {
            float s = 0;
            for (int d = 0; d < D; d++) s += qkv[i * QW + h * D + d] * qkv[j * QW + M + h * D + d];
            s *= scale;
            score[j] = s;
            mx = fmaxf(mx, s);
        }
        float total = 0;
        for (int j = 0; j < T; j++) { float e = expf(score[j] - mx); prob[(h * T + i) * T + j] = e; total += e; }
        for (int j = 0; j < T; j++) prob[(h * T + i) * T + j] /= total;
        for (int d = 0; d < D; d++) {
            float s = 0;
            for (int j = 0; j < T; j++) s += prob[(h * T + i) * T + j] * qkv[j * QW + 2 * M + h * D + d];
            self->data[(h * T + i) * D + d] = s;
        }
    }
}
static void k_attention_bwd(Node *self, Node *lhs, Node *rhs) {
    (void)rhs;
    if (!lhs->grad) return;
    const float *qkv = lhs->data;
    const float *prob = self->aux;
    float dprob[H * T * T] = {0}, dscore[H * T * T] = {0};
    for (int h = 0; h < H; h++) for (int i = 0; i < T; i++) for (int j = 0; j < T; j++) for (int d = 0; d < D; d++) {
        float g = self->grad[(h * T + i) * D + d];
        dprob[(h * T + i) * T + j] += g * qkv[j * QW + 2 * M + h * D + d];
        lhs->grad[j * QW + 2 * M + h * D + d] += g * prob[(h * T + i) * T + j];
    }
    for (int h = 0; h < H; h++) for (int i = 0; i < T; i++) {
        float dot = 0;
        for (int j = 0; j < T; j++) dot += dprob[(h * T + i) * T + j] * prob[(h * T + i) * T + j];
        for (int j = 0; j < T; j++) dscore[(h * T + i) * T + j] = prob[(h * T + i) * T + j] * (dprob[(h * T + i) * T + j] - dot);
    }
    float scale = 1.0f / sqrtf((float)D);
    for (int h = 0; h < H; h++) for (int i = 0; i < T; i++) for (int j = 0; j < T; j++) for (int d = 0; d < D; d++) {
        float g = dscore[(h * T + i) * T + j] * scale;
        lhs->grad[i * QW + h * D + d] += g * qkv[j * QW + M + h * D + d];
        lhs->grad[j * QW + M + h * D + d] += g * qkv[i * QW + h * D + d];
    }
}

/* --- new: CONTIGUOUS, the merge-heads layout op: [H,T,D] -> [T,H*D].
 * Not a legal zero-copy reshape (tensor_qkv_layout_check.c already proved
 * this), so this is real data movement, dispatched the same generic way
 * as every other op. --- */
static void k_contiguous_fwd(Node *self, Node *lhs, Node *rhs) {
    (void)rhs;
    for (int i = 0; i < T; i++) for (int h = 0; h < H; h++) for (int d = 0; d < D; d++)
        self->data[i * M + h * D + d] = lhs->data[(h * T + i) * D + d];
}
static void k_contiguous_bwd(Node *self, Node *lhs, Node *rhs) {
    (void)rhs;
    if (!lhs->grad) return;
    for (int i = 0; i < T; i++) for (int h = 0; h < H; h++) for (int d = 0; d < D; d++)
        lhs->grad[(h * T + i) * D + d] += self->grad[i * M + h * D + d];
}

typedef void (*KernelFn)(Node *, Node *, Node *);
static const KernelFn FWD[OP_COUNT] = {
    [MATMUL] = k_matmul_fwd, [BIAS] = k_bias_fwd, [RELU] = k_relu_fwd, [MSE] = k_mse_fwd,
    [ATTENTION] = k_attention_fwd, [RESIDUAL] = k_residual_fwd, [LAYERNORM] = k_layernorm_fwd, [CONTIGUOUS] = k_contiguous_fwd,
};
static const KernelFn BWD[OP_COUNT] = {
    [MATMUL] = k_matmul_bwd, [BIAS] = k_bias_bwd, [RELU] = k_relu_bwd, [MSE] = k_mse_bwd,
    [ATTENTION] = k_attention_bwd, [RESIDUAL] = k_residual_bwd, [LAYERNORM] = k_layernorm_bwd, [CONTIGUOUS] = k_contiguous_bwd,
};

/* --- the exact same generic fusion-detection pass as
 * tensor_graph_engine_fusion_check.c (matmul+bias[+relu], consumer-count
 * ==1), applied here unmodified to a graph with no BIAS nodes at all. The
 * point isn't to fuse anything — it's that the SAME pass, with no
 * Transformer-specific carve-out, correctly finds nothing to fuse instead
 * of misfiring or needing a special case for this shape. */
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

/* --- the real Transformer block as a Node graph. Same shape as every
 * other transformer spike (T=3,M=4,H=2,D=2,F=6). No BIAS nodes: the real
 * Block has no FFN bias terms (matmul->relu->matmul only), matching
 * tensor_transformer_reference_spike.h exactly. --- */
enum {
    N_X, N_WQ, N_WO, N_W1, N_W2, N_TARGET,
    N_QKV, N_ATTN, N_MERGED, N_PROJ, N_SUM1, N_LN1, N_Z1, N_ACT, N_FF, N_SUM2, N_LN2, N_LOSS,
    N_COUNT
};

int main(void) {
    float x[T * M], wq[M * QW], wo[M * M], w1[M * F], w2[F * M], target[T * M];
    float qkv[T * QW], attn[H * T * D], merged[T * M], proj[T * M], sum1[T * M], ln1[T * M];
    float z1[T * F], act[T * F], ff[T * M], sum2[T * M], ln2[T * M], loss[1];
    float gwq[M * QW], gwo[M * M], gw1[M * F], gw2[F * M];
    float gqkv[T * QW], gattn[H * T * D], gmerged[T * M], gproj[T * M], gsum1[T * M], gln1[T * M];
    float gz1[T * F], gact[T * F], gff[T * M], gsum2[T * M], gln2[T * M], gloss[1];
    float attn_aux[H * T * T], ln1_aux[2 * T], ln2_aux[2 * T];

    for (int i = 0; i < T * M; i++) x[i] = initv(i, 1);
    for (int i = 0; i < M * QW; i++) wq[i] = initv(i, 2) * .4f;
    for (int i = 0; i < M * M; i++) wo[i] = initv(i, 3) * .4f;
    for (int i = 0; i < M * F; i++) w1[i] = initv(i, 4) * .5f;
    for (int i = 0; i < F * M; i++) w2[i] = initv(i, 5) * .5f;
    /* Per-row normalized, same as every other transformer training spike's
     * target_make(): the last op before the loss is LN2, whose output is
     * inherently zero-mean/unit-variance per row, so an unnormalized
     * target has an unreachable floor — not an engine bug (finite
     * differences below already confirm backward is exact), just a target
     * that doesn't match what LayerNorm-terminated output can produce. */
    for (int i = 0; i < T; i++) {
        float m = 0, v = 0;
        for (int j = 0; j < M; j++) { target[i * M + j] = sinf((float)((i + 1) * (j + 2)) * .7f) + cosf((float)(i - j) * .4f); m += target[i * M + j]; }
        m /= M;
        for (int j = 0; j < M; j++) { float d = target[i * M + j] - m; v += d * d; }
        float inv = 1.0f / sqrtf(v / M + 1e-5f);
        for (int j = 0; j < M; j++) target[i * M + j] = (target[i * M + j] - m) * inv;
    }

    Node nodes[N_COUNT] = {
        [N_X] = {LEAF, NONE, NONE, INPUT, T, M, 0, x, NULL, NULL},
        [N_WQ] = {LEAF, NONE, NONE, PARAM, M, QW, 1, wq, gwq, NULL},
        [N_WO] = {LEAF, NONE, NONE, PARAM, M, M, 1, wo, gwo, NULL},
        [N_W1] = {LEAF, NONE, NONE, PARAM, M, F, 1, w1, gw1, NULL},
        [N_W2] = {LEAF, NONE, NONE, PARAM, F, M, 1, w2, gw2, NULL},
        [N_TARGET] = {LEAF, NONE, NONE, CONSTANT, T, M, 0, target, NULL, NULL},
        [N_QKV] = {MATMUL, N_X, N_WQ, TEMP, T, QW, 0, qkv, gqkv, NULL},
        [N_ATTN] = {ATTENTION, N_QKV, NONE, TEMP, 1, H * T * D, 0, attn, gattn, attn_aux},
        [N_MERGED] = {CONTIGUOUS, N_ATTN, NONE, TEMP, T, M, 0, merged, gmerged, NULL},
        [N_PROJ] = {MATMUL, N_MERGED, N_WO, TEMP, T, M, 0, proj, gproj, NULL},
        [N_SUM1] = {RESIDUAL, N_X, N_PROJ, TEMP, T, M, 0, sum1, gsum1, NULL},
        [N_LN1] = {LAYERNORM, N_SUM1, NONE, TEMP, T, M, 0, ln1, gln1, ln1_aux},
        [N_Z1] = {MATMUL, N_LN1, N_W1, TEMP, T, F, 0, z1, gz1, NULL},
        [N_ACT] = {RELU, N_Z1, NONE, TEMP, T, F, 0, act, gact, NULL},
        [N_FF] = {MATMUL, N_ACT, N_W2, TEMP, T, M, 0, ff, gff, NULL},
        [N_SUM2] = {RESIDUAL, N_LN1, N_FF, TEMP, T, M, 0, sum2, gsum2, NULL},
        [N_LN2] = {LAYERNORM, N_SUM2, NONE, TEMP, T, M, 0, ln2, gln2, ln2_aux},
        [N_LOSS] = {MSE, N_LN2, N_TARGET, TEMP, 1, 1, 0, loss, gloss, NULL},
    };

    unsigned forward_nodes[12] = {N_QKV, N_ATTN, N_MERGED, N_PROJ, N_SUM1, N_LN1, N_Z1, N_ACT, N_FF, N_SUM2, N_LN2, N_LOSS};
    Node *zeroable[15] = {&nodes[N_WQ], &nodes[N_WO], &nodes[N_W1], &nodes[N_W2],
                           &nodes[N_QKV], &nodes[N_ATTN], &nodes[N_MERGED], &nodes[N_PROJ], &nodes[N_SUM1], &nodes[N_LN1],
                           &nodes[N_Z1], &nodes[N_ACT], &nodes[N_FF], &nodes[N_SUM2], &nodes[N_LN2]};

    /* --- cross-check: the exact same fusion pass tensor_graph_engine_
     * fusion_check.c ran on the MLP graph (where it found MBR+MB and
     * collapsed 6 actions to 3), unmodified, on this graph. This block has
     * matmul immediately followed by relu (N_Z1 -> N_ACT) with no bias in
     * between anywhere — the pass's rule is specifically matmul+bias
     * [+relu], so the correct, honest answer here is "nothing to fuse",
     * not a misfire and not a special case for this shape. */
    {
        Group groups[12];
        unsigned ng = build_schedule(nodes, N_COUNT, forward_nodes, 12, groups);
        int all_plain = 1;
        for (unsigned i = 0; i < ng; i++) if (groups[i].kind != GROUP_PLAIN) all_plain = 0;
        if (ng != 12 || !all_plain) {
            fprintf(stderr, "fusion pass found unexpected groups on the no-bias transformer graph: ng=%u all_plain=%d\n", ng, all_plain);
            return 8;
        }
        printf("fusion cross-check: same generic pass as the MLP graph, applied unmodified -> groups=%u all_plain=yes "
               "(no BIAS nodes in this graph, so matmul+relu at N_Z1->N_ACT correctly does NOT fuse under the matmul+bias[+relu] rule)\n", ng);
    }
    uint32_t generation = 1;
    ExecStep steps[27];
    NodeCtx fctx[12], bctx[12];
    ZeroCtx zctx[15];

    /* --- correctness: finite differences on this exact engine, same
     * oracle pattern as every other spike, probing one weight from each
     * of the four trainable groups (not all of them — same minimal-but-real
     * pattern the MLP engine used). --- */
    unsigned at = 0;
    for (unsigned i = 0; i < 15; i++) { zctx[i] = (ZeroCtx){zeroable[i]->grad, zeroable[i]->rows * zeroable[i]->cols, generation, &generation}; steps[at] = (ExecStep){zero_execute, &zctx[i], 0, 0, 0}; at++; }
    for (unsigned i = 0; i < 12; i++) { fctx[i] = (NodeCtx){nodes, forward_nodes[i], generation, &generation, 0}; steps[at] = (ExecStep){node_execute, &fctx[i], 0, 0, 0}; at++; }
    if (at != 27 || tensor_transformer_steps_execute(steps, at)) return 1;
    nodes[N_LOSS].grad[0] = 1.0f;
    ExecStep bsteps[12];
    unsigned bat = 0;
    for (int i = 11; i >= 0; i--) { bctx[bat] = (NodeCtx){nodes, forward_nodes[i], generation, &generation, 1}; bsteps[bat] = (ExecStep){node_execute, &bctx[bat], 0, 0, 0}; bat++; }
    if (tensor_transformer_steps_execute(bsteps, bat)) return 2;

    struct { float *param, *grad; int idx; const char *name; } probes[] = {
        {wq, gwq, 5, "wq"}, {wo, gwo, 3, "wo"}, {w1, gw1, 7, "w1"}, {w2, gw2, 2, "w2"},
    };
    for (unsigned p = 0; p < 4; p++) {
        float eps = 1e-3f, old = probes[p].param[probes[p].idx], analytic = probes[p].grad[probes[p].idx];
        probes[p].param[probes[p].idx] = old + eps;
        for (unsigned i = 0; i < 12; i++) FWD[nodes[forward_nodes[i]].op](&nodes[forward_nodes[i]], &nodes[nodes[forward_nodes[i]].lhs], nodes[forward_nodes[i]].rhs != NONE ? &nodes[nodes[forward_nodes[i]].rhs] : NULL);
        float plus = loss[0];
        probes[p].param[probes[p].idx] = old - eps;
        for (unsigned i = 0; i < 12; i++) FWD[nodes[forward_nodes[i]].op](&nodes[forward_nodes[i]], &nodes[nodes[forward_nodes[i]].lhs], nodes[forward_nodes[i]].rhs != NONE ? &nodes[nodes[forward_nodes[i]].rhs] : NULL);
        float minus = loss[0];
        probes[p].param[probes[p].idx] = old;
        float numeric = (plus - minus) / (2 * eps);
        if (fabsf(numeric - analytic) > 5e-3f * fmaxf(1, fmaxf(fabsf(numeric), fabsf(analytic)))) {
            fprintf(stderr, "graph engine transformer backward mismatch on %s: analytic=%g numeric=%g\n", probes[p].name, analytic, numeric);
            return 3;
        }
    }

    /* --- train the same synthetic sequence-pattern target every other
     * transformer training spike in this repo uses, through this graph and
     * zero transformer-specific compiler code. --- */
    for (int i = 0; i < M * QW; i++) wq[i] = initv(i, 2) * .4f;
    for (int i = 0; i < M * M; i++) wo[i] = initv(i, 3) * .4f;
    for (int i = 0; i < M * F; i++) w1[i] = initv(i, 4) * .5f;
    for (int i = 0; i < F * M; i++) w2[i] = initv(i, 5) * .5f;
    float lr = .02f;
    generation = 100;
    unsigned n = 0;
    for (unsigned i = 0; i < 15; i++) { zctx[i] = (ZeroCtx){zeroable[i]->grad, zeroable[i]->rows * zeroable[i]->cols, generation, &generation}; steps[n] = (ExecStep){zero_execute, &zctx[i], 0, 0, 0}; n++; }
    for (unsigned i = 0; i < 12; i++) { fctx[i] = (NodeCtx){nodes, forward_nodes[i], generation, &generation, 0}; steps[n] = (ExecStep){node_execute, &fctx[i], 0, 0, 0}; n++; }
    for (unsigned i = 0; i < 12; i++) FWD[nodes[forward_nodes[i]].op](&nodes[forward_nodes[i]], &nodes[nodes[forward_nodes[i]].lhs], nodes[forward_nodes[i]].rhs != NONE ? &nodes[nodes[forward_nodes[i]].rhs] : NULL);
    float initial = loss[0];

    for (unsigned epoch = 0; epoch < 12000; epoch++) {
        unsigned fn = 0;
        for (unsigned i = 0; i < 15; i++) { zctx[i] = (ZeroCtx){zeroable[i]->grad, zeroable[i]->rows * zeroable[i]->cols, generation, &generation}; steps[fn] = (ExecStep){zero_execute, &zctx[i], 0, 0, 0}; fn++; }
        for (unsigned i = 0; i < 12; i++) { fctx[i] = (NodeCtx){nodes, forward_nodes[i], generation, &generation, 0}; steps[fn] = (ExecStep){node_execute, &fctx[i], 0, 0, 0}; fn++; }
        if (fn != 27 || tensor_transformer_steps_execute(steps, fn)) return 4;
        nodes[N_LOSS].grad[0] = 1.0f;
        unsigned bn = 0;
        for (int i = 11; i >= 0; i--) { bctx[bn] = (NodeCtx){nodes, forward_nodes[i], generation, &generation, 1}; bsteps[bn] = (ExecStep){node_execute, &bctx[bn], 0, 0, 0}; bn++; }
        if (bn != 12 || tensor_transformer_steps_execute(bsteps, bn)) return 5;
        for (int i = 0; i < M * QW; i++) wq[i] -= lr * gwq[i];
        for (int i = 0; i < M * M; i++) wo[i] -= lr * gwo[i];
        for (int i = 0; i < M * F; i++) w1[i] -= lr * gw1[i];
        for (int i = 0; i < F * M; i++) w2[i] -= lr * gw2[i];
    }
    for (unsigned i = 0; i < 12; i++) FWD[nodes[forward_nodes[i]].op](&nodes[forward_nodes[i]], &nodes[nodes[forward_nodes[i]].lhs], nodes[forward_nodes[i]].rhs != NONE ? &nodes[nodes[forward_nodes[i]].rhs] : NULL);
    float final = loss[0];

    printf("tensor graph engine transformer passed: nodes=%u ops=matmul,attention,contiguous,residual,layernorm,relu,mse "
           "dispatch=generic(by node->op) new_ops=attention,residual,layernorm,contiguous reused_unchanged=matmul,bias,relu,mse "
           "gradient_check=finite-difference(wq,wo,w1,w2) epochs=12000 loss=%.6f->%.6f via_shared_executor=tensor_transformer_steps_execute\n",
           (unsigned)N_COUNT, initial, final);
    return final < initial * .25f ? 0 : 6;
}
