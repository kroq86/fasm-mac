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

enum { LEAF, MATMUL, BIAS_ADD, RELU, MSE, ATTENTION, RESIDUAL, LAYERNORM, CONTIGUOUS, CONV, POOL, REDUCE_MEAN_ROWS, SIGMOID, BINARY_CROSS_ENTROPY, EMBED_LOOKUP, CAUSAL_ATTENTION, GELU, SOFTMAX_ROWS, OP_COUNT };
/* Storage/ownership role occupies the low nibble. RETAIN_GRAD is an
 * orthogonal semantic request: materialize this tensor's gradient even
 * when it is not trainable. It does not make a leaf an optimizer target. */
enum { INPUT = 1, PARAM = 2, CONSTANT = 4, TEMP = 8, RETAIN_GRAD = 16 };
enum { FORWARD, ZERO_GRAD, BACKWARD, OPTIMIZER };
#define NONE UINT32_MAX
enum { MAX_NODES = 64 };

typedef struct { float *data, *grad, *aux; uint32_t rows, cols, aux_count; } Tensor;
typedef struct { uint32_t op, lhs, rhs, flags; Tensor tensor; } Node;
/* Optional metadata for PARAM leaves. A NULL tensor.aux keeps the historical
 * multiplier 1.0. Only lr_multiplier is contractual today. */
typedef struct { float lr_multiplier; } ParameterOptimizationMetadata;
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

/* k_matmul_fwd stays the plain scalar ijk loop on every platform,
 * including arm64. An arm64 NEON variant (ikj loop order, vectorized over
 * output columns, ported unmodified from tensor_kernel_dispatch_spike.h's
 * already-verified native_matmul) was tried here and reverted: measured at
 * the exact M=1 shapes the killer-workload scaling matrix actually uses
 * (tensor_matmul_scaling_profile.c), it LOSES to this scalar loop by
 * 33-61% on the dominant shape (layer 1, K=784) across the entire
 * 25k-4M-param range — not a wash, a consistent regression. Mechanistic
 * reason: at M=1 (batch=1 inference), the ikj/vectorize-over-N pattern
 * re-reads and re-writes the *entire* output row from memory on every one
 * of the K iterations (no register-resident accumulator across K), which
 * is memory-traffic-bound; the scalar ijk loop keeps one accumulator in a
 * register across the whole K reduction per output element and only
 * writes it once. native_matmul's own shape sweep in
 * tensor_kernel_dispatch_profile.c tests M in {3,8,64}, where amortizing
 * the ikj pattern's re-read/re-write cost across multiple output rows pays
 * off — that's a real, different regime from this repo's actual inference
 * shape. Shipping the swap anyway because "NEON should help" would have
 * been exactly the kind of unmeasured assumption this project has
 * repeatedly had to walk back elsewhere; measuring first here caught it
 * before it shipped. A NEON strategy suited to M=1 (a K-reduction
 * vectorized per output element, keeping the accumulator in a register)
 * is a legitimate next experiment but a materially different kernel, not
 * a one-line swap, and isn't implemented here. */
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
static void k_sigmoid_fwd(Node *self, Node *a, Node *b) {
    (void)b;
    uint32_t n = self->tensor.rows * self->tensor.cols;
    for (uint32_t i = 0; i < n; i++) {
        float z = fmaxf(-60.0f, fminf(60.0f, a->tensor.data[i]));
        self->tensor.data[i] = 1.0f / (1.0f + expf(-z));
    }
}
static void k_sigmoid_bwd(Node *self, Node *a, Node *b, uint32_t mask) {
    (void)b;
    if (!(mask & 1)) return;
    uint32_t n = self->tensor.rows * self->tensor.cols;
    for (uint32_t i = 0; i < n; i++) {
        float y = self->tensor.data[i];
        a->tensor.grad[i] += self->tensor.grad[i] * y * (1.0f - y);
    }
}
/* Decoder-runtime Etap 2, operation 4/10: GELU. The tanh approximation
 * GPT-2's own reference code uses (not the exact erf-based GELU) --
 * matters because a later real-weight vertical slice needs to reproduce
 * GPT-2's own numerics, not a mathematically-cleaner but different
 * activation. Same constants (sqrt(2/pi), 0.044715) as the original GPT-2
 * and nanoGPT/llm.c reference implementations. */
static void k_gelu_fwd(Node *self, Node *a, Node *b) {
    (void)b;
    const float c0 = 0.7978845608028654f, c1 = 0.044715f;
    uint32_t n = self->tensor.rows * self->tensor.cols;
    for (uint32_t i = 0; i < n; i++) {
        float x = a->tensor.data[i];
        float u = c0 * (x + c1 * x * x * x);
        self->tensor.data[i] = 0.5f * x * (1.0f + tanhf(u));
    }
}
static void k_gelu_bwd(Node *self, Node *a, Node *b, uint32_t mask) {
    (void)b;
    if (!(mask & 1)) return;
    const float c0 = 0.7978845608028654f, c1 = 0.044715f;
    uint32_t n = self->tensor.rows * self->tensor.cols;
    for (uint32_t i = 0; i < n; i++) {
        float x = a->tensor.data[i];
        float u = c0 * (x + c1 * x * x * x);
        float t = tanhf(u);
        float sech2 = 1.0f - t * t;
        float du_dx = c0 * (1.0f + 3.0f * c1 * x * x);
        float dgelu = 0.5f * (1.0f + t) + 0.5f * x * sech2 * du_dx;
        a->tensor.grad[i] += self->tensor.grad[i] * dgelu;
    }
}
/* Decoder-runtime Etap 2, operation 5/10: numerically stable softmax as
 * its own reusable op (row-wise, last-axis) -- for the final
 * logits->vocab-probability step, distinct from the softmax already
 * baked inside k_attention_fwd/k_causal_attention_fwd's score
 * computation (that one is internal to attention and not reachable as a
 * standalone graph node). Same max-subtraction stability trick. Backward
 * reuses the exact softmax-Jacobian formula already validated inside
 * k_attention_bwd's dprob->dscore step (dscore=prob*(dprob-dot)) --
 * applied here row-wise instead of per (head,query) pair. */
static void k_softmax_rows_fwd(Node *self, Node *a, Node *b) {
    (void)b;
    uint32_t rows = self->tensor.rows, cols = self->tensor.cols;
    for (uint32_t i = 0; i < rows; i++) {
        float mx = -INFINITY;
        for (uint32_t j = 0; j < cols; j++) mx = fmaxf(mx, a->tensor.data[i * cols + j]);
        float total = 0;
        for (uint32_t j = 0; j < cols; j++) { float e = expf(a->tensor.data[i * cols + j] - mx); self->tensor.data[i * cols + j] = e; total += e; }
        for (uint32_t j = 0; j < cols; j++) self->tensor.data[i * cols + j] /= total;
    }
}
static void k_softmax_rows_bwd(Node *self, Node *a, Node *b, uint32_t mask) {
    (void)b;
    if (!(mask & 1)) return;
    uint32_t rows = self->tensor.rows, cols = self->tensor.cols;
    for (uint32_t i = 0; i < rows; i++) {
        float dot = 0;
        for (uint32_t j = 0; j < cols; j++) dot += self->tensor.grad[i * cols + j] * self->tensor.data[i * cols + j];
        for (uint32_t j = 0; j < cols; j++) {
            float y = self->tensor.data[i * cols + j];
            a->tensor.grad[i * cols + j] += y * (self->tensor.grad[i * cols + j] - dot);
        }
    }
}
static void k_bce_fwd(Node *self, Node *a, Node *b) {
    uint32_t n = a->tensor.rows * a->tensor.cols;
    float loss = 0;
    for (uint32_t i = 0; i < n; i++) {
        float p = fmaxf(1e-7f, fminf(1.0f - 1e-7f, a->tensor.data[i]));
        loss -= b->tensor.data[i] * logf(p) + (1.0f - b->tensor.data[i]) * logf(1.0f - p);
    }
    self->tensor.data[0] = loss / n;
}
static void k_bce_bwd(Node *self, Node *a, Node *b, uint32_t mask) {
    (void)self;
    uint32_t n = a->tensor.rows * a->tensor.cols;
    for (uint32_t i = 0; i < n; i++) {
        float p = fmaxf(1e-7f, fminf(1.0f - 1e-7f, a->tensor.data[i]));
        if (mask & 1) a->tensor.grad[i] += (p - b->tensor.data[i]) / (p * (1.0f - p) * n);
        if (mask & 2) b->tensor.grad[i] -= logf(p / (1.0f - p)) / n;
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

/* Sequence/classification bridge: reduce [rows, cols] to [1, cols].
 * This is a general tensor relation, not GunPoint-specific model logic. */
static void k_reduce_mean_rows_fwd(Node *self, Node *a, Node *b) {
    (void)b;
    for (uint32_t j = 0; j < a->tensor.cols; j++) {
        float sum = 0;
        for (uint32_t i = 0; i < a->tensor.rows; i++) sum += a->tensor.data[i * a->tensor.cols + j];
        self->tensor.data[j] = sum / a->tensor.rows;
    }
}
static void k_reduce_mean_rows_bwd(Node *self, Node *a, Node *b, uint32_t mask) {
    (void)b;
    if (!(mask & 1)) return;
    for (uint32_t i = 0; i < a->tensor.rows; i++)
        for (uint32_t j = 0; j < a->tensor.cols; j++)
            a->tensor.grad[i * a->tensor.cols + j] += self->tensor.grad[j] / a->tensor.rows;
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
/* Decoder-runtime Etap 2, operation 3/10: causal attention mask. Same
 * computation as k_attention_fwd, with future positions (j>i) masked to
 * -inf before softmax, so their softmax weight is exactly 0.0, not just
 * small -- token i never sees token i+1..T-1. No new backward kernel:
 * k_attention_bwd is reused as-is below, because every gradient term it
 * produces for a masked position is multiplied by that position's own
 * prob (which is exactly 0 here), so the existing math already zeroes
 * out future-position contributions without needing to know about the
 * mask itself. */
static void k_causal_attention_fwd(Node *self, Node *a, Node *b) {
    (void)b;
    const float *qkv = a->tensor.data;
    float *prob = self->tensor.aux;
    float scale = 1.0f / sqrtf((float)D);
    for (int h = 0; h < H; h++) for (int i = 0; i < T; i++) {
        float score[T], mx = -INFINITY;
        for (int j = 0; j < T; j++) {
            float s;
            if (j > i) { s = -INFINITY; }
            else {
                s = 0;
                for (int d = 0; d < D; d++) s += qkv[i * QW + h * D + d] * qkv[j * QW + M + h * D + d];
                s *= scale;
            }
            score[j] = s;
            mx = fmaxf(mx, s);
        }
        float total = 0;
        for (int j = 0; j < T; j++) { float e = j > i ? 0.0f : expf(score[j] - mx); prob[(h * T + i) * T + j] = e; total += e; }
        for (int j = 0; j < T; j++) prob[(h * T + i) * T + j] /= total;
        for (int d = 0; d < D; d++) {
            float s = 0;
            for (int j = 0; j <= i; j++) s += prob[(h * T + i) * T + j] * qkv[j * QW + 2 * M + h * D + d];
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

/* --- EMBED_LOOKUP: decoder-runtime step 1 (Etap 2 of the GPT-2 ladder).
 * Gathers rows from an embedding table by integer index -- `a` is the
 * table [VOCAB, DIM], `b` holds SEQ token ids as a [1, SEQ] row of
 * floats. Storing indices as float is exact for any realistic vocab:
 * float32 represents integers exactly up to 2^24, far past GPT-2's
 * 50257-token vocabulary, so there is no precision loss to round-trip
 * through. Indices never need a gradient (they are not learned, only
 * `a`'s rows are) -- expressed here simply by k_embed_bwd never touching
 * `b`, the same convention k_sigmoid_bwd/k_relu_bwd already use for an
 * unused second operand. Output is [SEQ, DIM], one row per token. */
static void k_embed_fwd(Node *self, Node *a, Node *b) {
    uint32_t seq = self->tensor.rows, dim = self->tensor.cols;
    for (uint32_t t = 0; t < seq; t++) {
        uint32_t idx = (uint32_t)(b->tensor.data[t] + 0.5f); /* round-to-nearest guards a stored-as-float integer */
        for (uint32_t d = 0; d < dim; d++) self->tensor.data[t * dim + d] = a->tensor.data[idx * dim + d];
    }
}
static void k_embed_bwd(Node *self, Node *a, Node *b, uint32_t mask) {
    if (!(mask & 1)) return;
    uint32_t seq = self->tensor.rows, dim = self->tensor.cols;
    for (uint32_t t = 0; t < seq; t++) {
        uint32_t idx = (uint32_t)(b->tensor.data[t] + 0.5f);
        for (uint32_t d = 0; d < dim; d++) a->tensor.grad[idx * dim + d] += self->tensor.grad[t * dim + d];
    }
}

static const FwdFn FWD[OP_COUNT] = {
    [MATMUL] = k_matmul_fwd, [BIAS_ADD] = k_bias_fwd, [RELU] = k_relu_fwd, [MSE] = k_mse_fwd,
    [ATTENTION] = k_attention_fwd, [RESIDUAL] = k_residual_fwd, [LAYERNORM] = k_layernorm_fwd, [CONTIGUOUS] = k_contiguous_fwd,
    [REDUCE_MEAN_ROWS] = k_reduce_mean_rows_fwd,
    [SIGMOID] = k_sigmoid_fwd, [BINARY_CROSS_ENTROPY] = k_bce_fwd,
    [CONV] = k_conv_fwd, [POOL] = k_pool_fwd,
    [EMBED_LOOKUP] = k_embed_fwd, [CAUSAL_ATTENTION] = k_causal_attention_fwd, [GELU] = k_gelu_fwd,
    [SOFTMAX_ROWS] = k_softmax_rows_fwd,
};
static const BwdFn BWD[OP_COUNT] = {
    [MATMUL] = k_matmul_bwd, [BIAS_ADD] = k_bias_bwd, [RELU] = k_relu_bwd, [MSE] = k_mse_bwd,
    [ATTENTION] = k_attention_bwd, [RESIDUAL] = k_residual_bwd, [LAYERNORM] = k_layernorm_bwd, [CONTIGUOUS] = k_contiguous_bwd,
    [REDUCE_MEAN_ROWS] = k_reduce_mean_rows_bwd,
    [SIGMOID] = k_sigmoid_bwd, [BINARY_CROSS_ENTROPY] = k_bce_bwd,
    [CONV] = k_conv_bwd, [POOL] = k_pool_bwd,
    [EMBED_LOOKUP] = k_embed_bwd, [CAUSAL_ATTENTION] = k_attention_bwd, [GELU] = k_gelu_bwd,
    [SOFTMAX_ROWS] = k_softmax_rows_bwd,
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
        if (g[i].op == LEAF) {
            if ((g[i].flags & PARAM) && g[i].tensor.aux) {
                const ParameterOptimizationMetadata *meta = (const ParameterOptimizationMetadata *)g[i].tensor.aux;
                if (g[i].tensor.aux_count != 1 || !isfinite(meta->lr_multiplier) || meta->lr_multiplier <= 0) return -4;
            }
            continue;
        }
        if (g[i].lhs >= i ||
            (g[i].op != RELU && g[i].op != LAYERNORM && g[i].op != CONTIGUOUS && g[i].op != ATTENTION && g[i].op != CAUSAL_ATTENTION && g[i].op != POOL && g[i].op != REDUCE_MEAN_ROWS && g[i].op != SIGMOID && g[i].op != GELU && g[i].op != SOFTMAX_ROWS && g[i].rhs >= i))
            return -2;
        const Tensor *a = &g[g[i].lhs].tensor, *b = g[i].rhs == NONE ? NULL : &g[g[i].rhs].tensor, *o = &g[i].tensor;
        if (g[i].op == MATMUL && (a->cols != b->rows || o->rows != a->rows || o->cols != b->cols)) return -3;
        if (g[i].op == BIAS_ADD && (a->rows != o->rows || a->cols != o->cols || b->rows != 1 || b->cols != o->cols)) return -3;
        if (g[i].op == RELU && (a->rows != o->rows || a->cols != o->cols)) return -3;
        if (g[i].op == MSE && (a->rows != b->rows || a->cols != b->cols || o->rows != 1 || o->cols != 1)) return -3;
        if (g[i].op == BINARY_CROSS_ENTROPY && (a->rows != b->rows || a->cols != b->cols || o->rows != 1 || o->cols != 1)) return -3;
        if (g[i].op == SIGMOID && (a->rows != o->rows || a->cols != o->cols)) return -3;
        if (g[i].op == GELU && (a->rows != o->rows || a->cols != o->cols)) return -3;
        if (g[i].op == SOFTMAX_ROWS && (a->rows != o->rows || a->cols != o->cols)) return -3;
        if (g[i].op == RESIDUAL && (a->rows != o->rows || a->cols != o->cols || b->rows != o->rows || b->cols != o->cols)) return -3;
        if (g[i].op == LAYERNORM && (a->rows != o->rows || a->cols != o->cols || !o->aux || o->aux_count < 2 * o->rows)) return -3;
        if (g[i].op == CONTIGUOUS && a->rows * a->cols != o->rows * o->cols) return -3;
        if (g[i].op == ATTENTION && (o->rows * o->cols != (uint32_t)(H * T * D) || !o->aux || o->aux_count < (uint32_t)(H * T * T))) return -3;
        if (g[i].op == CAUSAL_ATTENTION && (o->rows * o->cols != (uint32_t)(H * T * D) || !o->aux || o->aux_count < (uint32_t)(H * T * T))) return -3;
        if (g[i].op == CONV && (a->rows != 1 || a->cols != (uint32_t)(HIN * WIN * CIN) || b->rows != (uint32_t)COUT ||
                                 b->cols != (uint32_t)(CIN * KH * KW) || o->rows != 1 || o->cols != (uint32_t)(HOUT * WOUT * COUT)))
            return -3;
        if (g[i].op == POOL && (a->rows != 1 || a->cols != (uint32_t)(PHIN * PWIN * PCIN) || o->rows != 1 ||
                                 o->cols != (uint32_t)(PHOUT * PWOUT * PCIN) || !o->aux || o->aux_count < (uint32_t)(PHOUT * PWOUT * PCIN)))
            return -3;
        if (g[i].op == REDUCE_MEAN_ROWS && (o->rows != 1 || o->cols != a->cols)) return -3;
        if (g[i].op == EMBED_LOOKUP && (b->rows != 1 || b->cols != o->rows || a->cols != o->cols)) return -3;
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
        float multiplier = 1.0f;
        if (g[i].tensor.aux) multiplier = ((const ParameterOptimizationMetadata *)g[i].tensor.aux)->lr_multiplier;
        c[at] = (Context){g, i, OPTIMIZER, 0, lr * multiplier};
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
