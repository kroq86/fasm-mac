#ifndef TENSOR_SEMANTIC_COMPILER_H
#define TENSOR_SEMANTIC_COMPILER_H
/* Canonical merged graph compiler; not a stable core ABI.
 *
 * Reconciles two independently-built compilers that had both gone green on
 * MLP training: tensor_semantic_training_compiler_check.c's compile()
 * skeleton (one entry point, automatic needs-grad, shape validation,
 * optimizer scheduled through the executor like everything else) plus
 * everything tensor_graph_engine_*_check.c had that skeleton didn't
 * (table-driven op dispatch instead of an if-chain, generic per-op saved
 * state for LAYERNORM/ATTENTION, a separate fusion pass, the extended
 * Transformer op set, and a real external-dataset workload). Both original
 * files are left untouched as regression oracles — this is new code, not
 * an edit to either.
 *
 * Hard rule that shaped every choice here: no forward_nodes[]/zeroable[]/
 * hand-enumerated trainable lists anywhere downstream. A model is exactly
 * a Node array; compile() derives the rest.
 *
 * ATTENTION/CONTIGUOUS need fixed T/M/H/D/F/QW dimensions for their
 * indexing (same as tensor_graph_engine_transformer_check.c); CONV needs
 * fixed HIN/WIN/CIN/COUT/KH/KW (valid convolution, stride 1; HOUT/WOUT are
 * derived); POOL needs its own PHIN/PWIN/PCIN/PPH/PPW (non-overlapping max
 * pool, stride==window; PHOUT/PWOUT are derived) — a separate namespace
 * from CONV's, not implicitly "whatever CONV just produced", so POOL stays
 * meaningful standing alone. Define whichever set your graph's ops need
 * before #include-ing this header — the defaults below are inert
 * placeholders. Adding CONV, and now POOL, touched none of compile()'s
 * algorithm: same enum-indexed table, same compile(), same validate()
 * structure (one new op-specific branch plus one entry in the existing
 * unary-op list, the existing branches untouched), same needs-grad/
 * zero-grad/backward-mask/optimizer logic, same executor.
 */
#include <math.h>
#include <stdint.h>
#include <string.h>

#ifndef T
#define T 1
#endif
#ifndef M
#define M 1
#endif
#ifndef H
#define H 1
#endif
#ifndef D
#define D 1
#endif
#ifndef F
#define F 1
#endif
#ifndef QW
#define QW 1
#endif
#ifndef HIN
#define HIN 1
#endif
#ifndef WIN
#define WIN 1
#endif
#ifndef CIN
#define CIN 1
#endif
#ifndef COUT
#define COUT 1
#endif
#ifndef KH
#define KH 1
#endif
#ifndef KW
#define KW 1
#endif
#define HOUT (HIN - KH + 1)
#define WOUT (WIN - KW + 1)
#ifndef PHIN
#define PHIN 1
#endif
#ifndef PWIN
#define PWIN 1
#endif
#ifndef PCIN
#define PCIN 1
#endif
#ifndef PPH
#define PPH 1
#endif
#ifndef PPW
#define PPW 1
#endif
#define PHOUT (PHIN / PPH)
#define PWOUT (PWIN / PPW)

enum { LEAF, MATMUL, BIAS_ADD, RELU, MSE, ATTENTION, RESIDUAL, LAYERNORM, CONTIGUOUS, CONV, POOL, OP_COUNT };
/* Storage/ownership role occupies the low nibble. RETAIN_GRAD is an
 * orthogonal semantic request: materialize this tensor's gradient even
 * when it is not trainable. It does not make a leaf an optimizer target. */
enum { INPUT = 1, PARAM = 2, CONSTANT = 4, TEMP = 8, RETAIN_GRAD = 16 };
enum { FORWARD, ZERO_GRAD, BACKWARD, OPTIMIZER };
#define NONE UINT32_MAX
enum { MAX_NODES = 64 };

typedef struct { float *data, *grad, *aux; uint32_t rows, cols, aux_count; } Tensor;
typedef struct { uint32_t op, lhs, rhs, flags; Tensor tensor; } Node;
typedef struct { int (*run)(void *); void *context; uint32_t kind, flags; uint64_t reserved; } ExecStep;
typedef struct { Node *graph; uint32_t node, action, grad_mask; float lr; } Context;
extern int tensor_transformer_steps_execute(const ExecStep *, uint64_t);

/* --- op dispatch: table-driven (Claude's contribution) instead of an
 * if-chain, so a new op is a new table entry, not an edit to a shared
 * dispatch function. Backward takes an explicit grad_mask bit per operand
 * (Codex's contribution) rather than a NULL-pointer sentinel — the mask is
 * *computed once, automatically*, from graph reachability in compile()
 * below, not decided per-graph by whoever wrote the model. --- */
typedef void (*FwdFn)(Node *, Node *, Node *);
typedef void (*BwdFn)(Node *, Node *, Node *, uint32_t);

static void k_matmul_fwd(Node *self, Node *a, Node *b) {
    Tensor *out = &self->tensor;
    for (uint32_t i = 0; i < a->tensor.rows; i++) for (uint32_t j = 0; j < b->tensor.cols; j++) {
        float s = 0;
        for (uint32_t k = 0; k < a->tensor.cols; k++) s += a->tensor.data[i * a->tensor.cols + k] * b->tensor.data[k * b->tensor.cols + j];
        out->data[i * out->cols + j] = s;
    }
}
static void k_matmul_bwd(Node *self, Node *a, Node *b, uint32_t mask) {
    Tensor *out = &self->tensor;
    for (uint32_t i = 0; i < a->tensor.rows; i++) for (uint32_t j = 0; j < b->tensor.cols; j++) {
        float g = out->grad[i * out->cols + j];
        for (uint32_t k = 0; k < a->tensor.cols; k++) {
            if (mask & 1) a->tensor.grad[i * a->tensor.cols + k] += g * b->tensor.data[k * b->tensor.cols + j];
            if (mask & 2) b->tensor.grad[k * b->tensor.cols + j] += a->tensor.data[i * a->tensor.cols + k] * g;
        }
    }
}
static void k_bias_fwd(Node *self, Node *a, Node *b) {
    Tensor *out = &self->tensor;
    for (uint32_t i = 0; i < out->rows; i++) for (uint32_t j = 0; j < out->cols; j++) out->data[i * out->cols + j] = a->tensor.data[i * out->cols + j] + b->tensor.data[j];
}
static void k_bias_bwd(Node *self, Node *a, Node *b, uint32_t mask) {
    Tensor *out = &self->tensor;
    for (uint32_t i = 0; i < out->rows; i++) for (uint32_t j = 0; j < out->cols; j++) {
        float g = out->grad[i * out->cols + j];
        if (mask & 1) a->tensor.grad[i * out->cols + j] += g;
        if (mask & 2) b->tensor.grad[j] += g;
    }
}
static void k_relu_fwd(Node *self, Node *a, Node *b) {
    (void)b;
    Tensor *out = &self->tensor;
    for (uint32_t i = 0; i < out->rows * out->cols; i++) out->data[i] = a->tensor.data[i] > 0 ? a->tensor.data[i] : 0;
}
static void k_relu_bwd(Node *self, Node *a, Node *b, uint32_t mask) {
    (void)b;
    Tensor *out = &self->tensor;
    if (mask & 1) for (uint32_t i = 0; i < out->rows * out->cols; i++) a->tensor.grad[i] += a->tensor.data[i] > 0 ? out->grad[i] : 0;
}
static void k_mse_fwd(Node *self, Node *a, Node *b) {
    float loss = 0;
    uint32_t n = a->tensor.rows * a->tensor.cols;
    for (uint32_t i = 0; i < n; i++) { float e = a->tensor.data[i] - b->tensor.data[i]; loss += e * e; }
    self->tensor.data[0] = loss / n;
}
static void k_mse_bwd(Node *self, Node *a, Node *b, uint32_t mask) {
    (void)self;
    uint32_t n = a->tensor.rows * a->tensor.cols;
    for (uint32_t i = 0; i < n; i++) {
        float g = 2 * (a->tensor.data[i] - b->tensor.data[i]) / n;
        if (mask & 1) a->tensor.grad[i] += g;
        if (mask & 2) b->tensor.grad[i] -= g;
    }
}

/* --- residual/layernorm/attention/contiguous: ported from
 * tensor_graph_engine_transformer_check.c, rewritten to take grad_mask
 * instead of checking grad-pointer-non-NULL, and using the generic `aux`
 * buffer (not transformer-specific Node fields) for saved forward state:
 * attention saves prob[H*T*T], layernorm saves mean[rows]+invstd[rows]. --- */
static void k_residual_fwd(Node *self, Node *a, Node *b) {
    uint32_t n = self->tensor.rows * self->tensor.cols;
    for (uint32_t i = 0; i < n; i++) self->tensor.data[i] = a->tensor.data[i] + b->tensor.data[i];
}
static void k_residual_bwd(Node *self, Node *a, Node *b, uint32_t mask) {
    uint32_t n = self->tensor.rows * self->tensor.cols;
    for (uint32_t i = 0; i < n; i++) {
        float g = self->tensor.grad[i];
        if (mask & 1) a->tensor.grad[i] += g;
        if (mask & 2) b->tensor.grad[i] += g;
    }
}
static void k_layernorm_fwd(Node *self, Node *a, Node *b) {
    (void)b;
    Tensor *out = &self->tensor;
    float *mean = out->aux, *invstd = out->aux + out->rows;
    for (uint32_t i = 0; i < out->rows; i++) {
        float m = 0, v = 0;
        for (uint32_t j = 0; j < out->cols; j++) m += a->tensor.data[i * out->cols + j];
        m /= out->cols;
        for (uint32_t j = 0; j < out->cols; j++) { float d = a->tensor.data[i * out->cols + j] - m; v += d * d; }
        mean[i] = m;
        invstd[i] = 1.0f / sqrtf(v / out->cols + 1e-5f);
        for (uint32_t j = 0; j < out->cols; j++) out->data[i * out->cols + j] = (a->tensor.data[i * out->cols + j] - m) * invstd[i];
    }
}
static void k_layernorm_bwd(Node *self, Node *a, Node *b, uint32_t mask) {
    (void)b;
    if (!(mask & 1)) return;
    Tensor *out = &self->tensor;
    float *mean = out->aux, *invstd = out->aux + out->rows;
    for (uint32_t i = 0; i < out->rows; i++) {
        float sum = 0, sumh = 0;
        for (uint32_t j = 0; j < out->cols; j++) {
            float hh = (a->tensor.data[i * out->cols + j] - mean[i]) * invstd[i];
            sum += out->grad[i * out->cols + j];
            sumh += out->grad[i * out->cols + j] * hh;
        }
        for (uint32_t j = 0; j < out->cols; j++) {
            float hh = (a->tensor.data[i * out->cols + j] - mean[i]) * invstd[i];
            a->tensor.grad[i * out->cols + j] += invstd[i] * (out->grad[i * out->cols + j] - (sum + hh * sumh) / out->cols);
        }
    }
}
static void k_attention_fwd(Node *self, Node *a, Node *b) {
    (void)b;
    const float *qkv = a->tensor.data;
    float *prob = self->tensor.aux;
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
            self->tensor.data[(h * T + i) * D + d] = s;
        }
    }
}
static void k_attention_bwd(Node *self, Node *a, Node *b, uint32_t mask) {
    (void)b;
    if (!(mask & 1)) return;
    const float *qkv = a->tensor.data;
    const float *prob = self->tensor.aux;
    float dprob[H * T * T] = {0}, dscore[H * T * T] = {0};
    for (int h = 0; h < H; h++) for (int i = 0; i < T; i++) for (int j = 0; j < T; j++) for (int d = 0; d < D; d++) {
        float g = self->tensor.grad[(h * T + i) * D + d];
        dprob[(h * T + i) * T + j] += g * qkv[j * QW + 2 * M + h * D + d];
        a->tensor.grad[j * QW + 2 * M + h * D + d] += g * prob[(h * T + i) * T + j];
    }
    for (int h = 0; h < H; h++) for (int i = 0; i < T; i++) {
        float dot = 0;
        for (int j = 0; j < T; j++) dot += dprob[(h * T + i) * T + j] * prob[(h * T + i) * T + j];
        for (int j = 0; j < T; j++) dscore[(h * T + i) * T + j] = prob[(h * T + i) * T + j] * (dprob[(h * T + i) * T + j] - dot);
    }
    float scale = 1.0f / sqrtf((float)D);
    for (int h = 0; h < H; h++) for (int i = 0; i < T; i++) for (int j = 0; j < T; j++) for (int d = 0; d < D; d++) {
        float g = dscore[(h * T + i) * T + j] * scale;
        a->tensor.grad[i * QW + h * D + d] += g * qkv[j * QW + M + h * D + d];
        a->tensor.grad[j * QW + M + h * D + d] += g * qkv[i * QW + h * D + d];
    }
}
static void k_contiguous_fwd(Node *self, Node *a, Node *b) {
    (void)b;
    for (int i = 0; i < T; i++) for (int h = 0; h < H; h++) for (int d = 0; d < D; d++)
        self->tensor.data[i * M + h * D + d] = a->tensor.data[(h * T + i) * D + d];
}
static void k_contiguous_bwd(Node *self, Node *a, Node *b, uint32_t mask) {
    (void)b;
    if (!(mask & 1)) return;
    for (int i = 0; i < T; i++) for (int h = 0; h < H; h++) for (int d = 0; d < D; d++)
        a->tensor.grad[(h * T + i) * D + d] += self->tensor.grad[i * M + h * D + d];
}

/* --- CONV: valid (no padding) stride-1 2D convolution. Input is a flat
 * [1, HIN*WIN*CIN] node (row-major h,w,c); weight is [COUT, CIN*KH*KW]
 * (one filter per row); output is a flat [1, HOUT*WOUT*COUT] node — flat on
 * both sides so a plain MATMUL can consume the output directly afterward
 * (a real reshape, not a broadcast trick: element (oh,ow,co) always lives
 * at the same offset (oh*WOUT+ow)*COUT+co regardless of which op treats
 * this buffer as [1,N] or [HOUT*WOUT,COUT]). No bias — same reason the
 * Transformer FFN has none: kept out to isolate this op as one new trait,
 * not a bundle of unrelated additions; BIAS_ADD already exists if a model
 * wants one on a [HOUT*WOUT,COUT]-shaped conv output. --- */
static void k_conv_fwd(Node *self, Node *a, Node *b) {
    for (int co = 0; co < COUT; co++) for (int oh = 0; oh < HOUT; oh++) for (int ow = 0; ow < WOUT; ow++) {
        float s = 0;
        for (int ci = 0; ci < CIN; ci++) for (int kh = 0; kh < KH; kh++) for (int kw = 0; kw < KW; kw++) {
            int ih = oh + kh, iw = ow + kw;
            s += a->tensor.data[(ih * WIN + iw) * CIN + ci] * b->tensor.data[co * (CIN * KH * KW) + (ci * KH + kh) * KW + kw];
        }
        self->tensor.data[(oh * WOUT + ow) * COUT + co] = s;
    }
}
static void k_conv_bwd(Node *self, Node *a, Node *b, uint32_t mask) {
    for (int co = 0; co < COUT; co++) for (int oh = 0; oh < HOUT; oh++) for (int ow = 0; ow < WOUT; ow++) {
        float g = self->tensor.grad[(oh * WOUT + ow) * COUT + co];
        for (int ci = 0; ci < CIN; ci++) for (int kh = 0; kh < KH; kh++) for (int kw = 0; kw < KW; kw++) {
            int ih = oh + kh, iw = ow + kw;
            uint32_t ai = (ih * WIN + iw) * CIN + ci, bi = co * (CIN * KH * KW) + (ci * KH + kh) * KW + kw;
            if (mask & 1) a->tensor.grad[ai] += g * b->tensor.data[bi];
            if (mask & 2) b->tensor.grad[bi] += g * a->tensor.data[ai];
        }
    }
}

/* --- POOL: non-overlapping max pool (stride == window, no padding).
 * Input is a flat [1, PHIN*PWIN*PCIN] node using the same (h,w,c) row-major
 * layout CONV's output already uses, so it drops in directly after a CONV
 * node with no reshape; output is flat [1, PHOUT*PWOUT*PCIN]. Own
 * dimension namespace (PHIN/PWIN/PCIN/PPH/PPW), not implicitly reusing
 * CONV's HOUT/WOUT/COUT, so POOL means the same thing whether or not a
 * CONV precedes it. Saved state is the argmax's flat input index per
 * output element, in `aux` — same generic saved-state mechanism
 * LAYERNORM/ATTENTION already use, not a new struct field. --- */
static void k_pool_fwd(Node *self, Node *a, Node *b) {
    (void)b;
    for (int c = 0; c < PCIN; c++) for (int ph = 0; ph < PHOUT; ph++) for (int pw = 0; pw < PWOUT; pw++) {
        float best = -INFINITY;
        int best_idx = 0;
        for (int dh = 0; dh < PPH; dh++) for (int dw = 0; dw < PPW; dw++) {
            int ih = ph * PPH + dh, iw = pw * PPW + dw;
            int idx = (ih * PWIN + iw) * PCIN + c;
            float v = a->tensor.data[idx];
            if (v > best) { best = v; best_idx = idx; }
        }
        int oidx = (ph * PWOUT + pw) * PCIN + c;
        self->tensor.data[oidx] = best;
        self->tensor.aux[oidx] = (float)best_idx;
    }
}
static void k_pool_bwd(Node *self, Node *a, Node *b, uint32_t mask) {
    (void)b;
    if (!(mask & 1)) return;
    for (int c = 0; c < PCIN; c++) for (int ph = 0; ph < PHOUT; ph++) for (int pw = 0; pw < PWOUT; pw++) {
        int oidx = (ph * PWOUT + pw) * PCIN + c;
        int idx = (int)self->tensor.aux[oidx];
        a->tensor.grad[idx] += self->tensor.grad[oidx];
    }
}

static const FwdFn FWD[OP_COUNT] = {
    [MATMUL] = k_matmul_fwd, [BIAS_ADD] = k_bias_fwd, [RELU] = k_relu_fwd, [MSE] = k_mse_fwd,
    [ATTENTION] = k_attention_fwd, [RESIDUAL] = k_residual_fwd, [LAYERNORM] = k_layernorm_fwd, [CONTIGUOUS] = k_contiguous_fwd,
    [CONV] = k_conv_fwd, [POOL] = k_pool_fwd,
};
static const BwdFn BWD[OP_COUNT] = {
    [MATMUL] = k_matmul_bwd, [BIAS_ADD] = k_bias_bwd, [RELU] = k_relu_bwd, [MSE] = k_mse_bwd,
    [ATTENTION] = k_attention_bwd, [RESIDUAL] = k_residual_bwd, [LAYERNORM] = k_layernorm_bwd, [CONTIGUOUS] = k_contiguous_bwd,
    [CONV] = k_conv_bwd, [POOL] = k_pool_bwd,
};

static int action(void *opaque) {
    Context *c = opaque;
    Node *n = &c->graph[c->node];
    Node *a = n->lhs == NONE ? NULL : &c->graph[n->lhs];
    Node *b = n->rhs == NONE ? NULL : &c->graph[n->rhs];
    Tensor *out = &n->tensor;
    if (c->action == ZERO_GRAD) { memset(out->grad, 0, (size_t)out->rows * out->cols * sizeof(float)); return 0; }
    if (c->action == OPTIMIZER) { for (uint32_t i = 0; i < out->rows * out->cols; i++) out->data[i] -= c->lr * out->grad[i]; return 0; }
    if (c->action == FORWARD) { FWD[n->op](n, a, b); return 0; }
    BWD[n->op](n, a, b, c->grad_mask);
    return 0;
}

/* --- shape validation (Codex's contribution), extended to the new ops.
 * ATTENTION/CONTIGUOUS check against the fixed T/M/H/D/QW this header was
 * built with; RESIDUAL/LAYERNORM check same-shape like RELU already did. --- */
static int validate(const Node *g, uint32_t n) {
    for (uint32_t i = 0; i < n; i++) {
        if (!g[i].tensor.data || !g[i].tensor.grad || !g[i].tensor.rows || !g[i].tensor.cols) return -1;
        if (g[i].op == LEAF) continue;
        if (g[i].lhs >= i ||
            (g[i].op != RELU && g[i].op != LAYERNORM && g[i].op != CONTIGUOUS && g[i].op != ATTENTION && g[i].op != POOL && g[i].rhs >= i))
            return -2;
        const Tensor *a = &g[g[i].lhs].tensor, *b = g[i].rhs == NONE ? NULL : &g[g[i].rhs].tensor, *o = &g[i].tensor;
        if (g[i].op == MATMUL && (a->cols != b->rows || o->rows != a->rows || o->cols != b->cols)) return -3;
        if (g[i].op == BIAS_ADD && (a->rows != o->rows || a->cols != o->cols || b->rows != 1 || b->cols != o->cols)) return -3;
        if (g[i].op == RELU && (a->rows != o->rows || a->cols != o->cols)) return -3;
        if (g[i].op == MSE && (a->rows != b->rows || a->cols != b->cols || o->rows != 1 || o->cols != 1)) return -3;
        if (g[i].op == RESIDUAL && (a->rows != o->rows || a->cols != o->cols || b->rows != o->rows || b->cols != o->cols)) return -3;
        if (g[i].op == LAYERNORM && (a->rows != o->rows || a->cols != o->cols || !o->aux || o->aux_count < 2 * o->rows)) return -3;
        if (g[i].op == CONTIGUOUS && a->rows * a->cols != o->rows * o->cols) return -3;
        if (g[i].op == ATTENTION && (o->rows * o->cols != (uint32_t)(H * T * D) || !o->aux || o->aux_count < (uint32_t)(H * T * T))) return -3;
        if (g[i].op == CONV && (a->rows != 1 || a->cols != (uint32_t)(HIN * WIN * CIN) || b->rows != (uint32_t)COUT ||
                                 b->cols != (uint32_t)(CIN * KH * KW) || o->rows != 1 || o->cols != (uint32_t)(HOUT * WOUT * COUT)))
            return -3;
        if (g[i].op == POOL && (a->rows != 1 || a->cols != (uint32_t)(PHIN * PWIN * PCIN) || o->rows != 1 ||
                                 o->cols != (uint32_t)(PHOUT * PWOUT * PCIN) || !o->aux || o->aux_count < (uint32_t)(PHOUT * PWOUT * PCIN)))
            return -3;
    }
    return 0;
}

/* --- the one entry point: given ANY validated graph, derive needs-grad
 * automatically, and emit the complete forward+zero+backward+optimizer
 * schedule. No forward_nodes[]/zeroable[] — a model is exactly `g`. --- */
static int compile(Node *g, uint32_t n, float lr, ExecStep *s, Context *c, uint32_t cap, uint32_t *out) {
    if (validate(g, n) || n > MAX_NODES) return -1;
    uint8_t needs[MAX_NODES] = {0};
    for (uint32_t i = 0; i < n; i++)
        needs[i] = g[i].op == LEAF ? ((g[i].flags & (PARAM | RETAIN_GRAD)) != 0)
                                   : ((g[i].flags & RETAIN_GRAD) || needs[g[i].lhs] ||
                                      (g[i].rhs != NONE && needs[g[i].rhs]));
    uint32_t at = 0;
    for (uint32_t i = 0; i < n; i++) if (g[i].op != LEAF) {
        if (at == cap) return -2;
        c[at] = (Context){g, i, FORWARD, 0, lr};
        s[at] = (ExecStep){action, &c[at], FORWARD, 0, 0};
        at++;
    }
    for (uint32_t i = 0; i < n; i++) if (needs[i] && g[i].op != MSE) {
        if (at == cap) return -2;
        c[at] = (Context){g, i, ZERO_GRAD, 0, lr};
        s[at] = (ExecStep){action, &c[at], ZERO_GRAD, 0, 0};
        at++;
    }
    for (uint32_t i = n; i-- > 0;) if (g[i].op != LEAF && needs[i]) {
        if (at == cap) return -2;
        uint32_t mask = needs[g[i].lhs] ? 1 : 0;
        if (g[i].rhs != NONE && needs[g[i].rhs]) mask |= 2;
        c[at] = (Context){g, i, BACKWARD, mask, lr};
        s[at] = (ExecStep){action, &c[at], BACKWARD, 0, 0};
        at++;
    }
    for (uint32_t i = 0; i < n; i++) if (g[i].op == LEAF && (g[i].flags & PARAM)) {
        if (at == cap) return -2;
        c[at] = (Context){g, i, OPTIMIZER, 0, lr};
        s[at] = (ExecStep){action, &c[at], OPTIMIZER, 0, 0};
        at++;
    }
    *out = at;
    return 0;
}

/* --- fusion: a separate, optional generic pass (Claude's contribution),
 * not built into compile() or into any model's logic. Detects
 * MATMUL->BIAS_ADD[->RELU] runs with exactly one consumer each (same rule
 * tensor_compiler_planner_check.c locked in statically) and reports groups
 * a caller can use to build a shorter fused schedule on top of the same
 * validated, needs-grad-derived graph compile() already establishes. --- */
enum { GROUP_PLAIN, GROUP_MB, GROUP_MBR };
typedef struct { uint8_t kind; uint32_t matmul, bias, relu, plain; } Group;
static unsigned consumers_of(const Node *g, unsigned n, unsigned needle) __attribute__((unused));
static unsigned consumers_of(const Node *g, unsigned n, unsigned needle) {
    unsigned users = 0;
    for (unsigned i = 0; i < n; i++) {
        if (g[i].op == LEAF) continue;
        users += g[i].lhs == needle;
        users += g[i].rhs != NONE && g[i].rhs == needle;
    }
    return users;
}
static unsigned detect_fusion(Node *g, unsigned n, Group *groups) __attribute__((unused));
static unsigned detect_fusion(Node *g, unsigned n, Group *groups) {
    unsigned gi = 0, i = 0;
    while (i < n) {
        if (g[i].op == LEAF) { i++; continue; }
        if (g[i].op == MATMUL && i + 1 < n && g[i + 1].op == BIAS_ADD && g[i + 1].lhs == i && consumers_of(g, n, i) == 1) {
            if (i + 2 < n && g[i + 2].op == RELU && g[i + 2].lhs == i + 1 && consumers_of(g, n, i + 1) == 1) {
                groups[gi++] = (Group){GROUP_MBR, i, i + 1, i + 2, NONE};
                i += 3;
                continue;
            }
            groups[gi++] = (Group){GROUP_MB, i, i + 1, NONE, NONE};
            i += 2;
            continue;
        }
        groups[gi++] = (Group){GROUP_PLAIN, NONE, NONE, NONE, i};
        i += 1;
    }
    return gi;
}

#endif
