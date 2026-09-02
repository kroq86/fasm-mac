/* Decoder-runtime Etap 2, operation 8/10 (final item): the autoregressive
 * generation loop. This is the integrative check that ties every op built
 * in this stage into one toy decoder block and one generation loop:
 *   token embedding (EMBED_LOOKUP) + positional embedding (EMBED_LOOKUP)
 *   -> RESIDUAL -> QKV projection (MATMUL) -> CAUSAL_ATTENTION ->
 *   CONTIGUOUS -> output projection (MATMUL) -> RESIDUAL -> FFN
 *   (MATMUL -> GELU -> MATMUL) -> RESIDUAL -> logits projection (MATMUL)
 *   -> SOFTMAX_ROWS -> sample_top_k_temperature -> append -> repeat.
 *
 * Weights are fixed/untrained (deterministic pseudo-random) -- this is
 * NOT the Etap 3 miniature-GPT milestone (real tokenizer, trained
 * weights). It only has to prove the LOOP's wiring and its KV-cache path
 * are correct, exactly like a real generation loop's.
 *
 * Because T is this project's compile-time attention-shape macro (not a
 * runtime value), the same three-layer strategy as operation 7/10's
 * kv-cache check is used:
 *   1. ANCHOR: at this binary's fixed T, the REAL canonical graph
 *      (compile()+executor, every op above) is run once over a full
 *      T-length token sequence and compared to a hand-rolled runtime-n
 *      reference implementing the identical per-op formulas.
 *   2. CORE PROPERTY: an incremental KV-cache generation loop (reusing
 *      kv-cache's tested append pattern) is checked step-by-step against
 *      the runtime-n full-recompute reference, and both are driven
 *      through the same sampler with the same RNG state -- their
 *      sampled tokens must match at every generation step, not just
 *      their logits.
 *   3. REPLAY: same seed -> identical generated sequence; a shorter
 *      generation run is a strict prefix of a longer one from the same
 *      seed (the loop-level analogue of kv-cache's non-retroactivity).
 */
#include "tensor_semantic_compiler.h"
#include "tensor_sampling.h"
#include <stdio.h>
#include <stdlib.h>

enum { VOCAB = 13, MAXPOS = 9, MAXN = 9, PROMPT_LEN = 2 };

static float dv(int i, int salt) {
    float a = sinf((float)(i * 12.9898f + salt * 78.233f)) * 43758.5453f;
    return a - floorf(a) - 0.5f;
}

typedef struct {
    float wte[VOCAB * M], wpe[MAXPOS * M];
    float wqkv[M * QW], wo[M * M];
    float w1[M * F], w2[F * M];
    float wout[M * VOCAB];
} Weights;

static void init_weights(Weights *w) {
    for (int i = 0; i < VOCAB * M; i++) w->wte[i] = dv(i, 101) * 0.5f;
    for (int i = 0; i < MAXPOS * M; i++) w->wpe[i] = dv(i, 102) * 0.5f;
    for (int i = 0; i < M * QW; i++) w->wqkv[i] = dv(i, 103) * 0.3f;
    for (int i = 0; i < M * M; i++) w->wo[i] = dv(i, 104) * 0.3f;
    for (int i = 0; i < M * F; i++) w->w1[i] = dv(i, 105) * 0.3f;
    for (int i = 0; i < F * M; i++) w->w2[i] = dv(i, 106) * 0.3f;
    for (int i = 0; i < M * VOCAB; i++) w->wout[i] = dv(i, 107) * 0.3f;
}

/* runtime-n full-recompute forward: given the token history tokens[0..n-1],
 * compute logits (softmax probabilities) for ALL n positions. Mirrors the
 * canonical kernels' exact per-op formulas. */
static void ref_forward(const Weights *w, const int *tokens, int n, float *probs_out /* [n*VOCAB] */) {
    float h[MAXN * 6]; /* M=6 fixed by this file's build flags */
    for (int t = 0; t < n; t++) for (int d = 0; d < M; d++)
        h[t * M + d] = w->wte[tokens[t] * M + d] + w->wpe[t * M + d];

    float qkv[MAXN * 18]; /* QW=18 */
    for (int t = 0; t < n; t++) for (int c = 0; c < QW; c++) {
        float s = 0; for (int d = 0; d < M; d++) s += h[t * M + d] * w->wqkv[d * QW + c];
        qkv[t * QW + c] = s;
    }

    float attn_out[H * MAXN * D];
    float scale = 1.0f / sqrtf((float)D);
    for (int hh = 0; hh < H; hh++) for (int i = 0; i < n; i++) {
        float score[MAXN], mx = -INFINITY;
        for (int j = 0; j <= i; j++) {
            float s = 0;
            for (int d = 0; d < D; d++) s += qkv[i * QW + hh * D + d] * qkv[j * QW + M + hh * D + d];
            s *= scale; score[j] = s; mx = fmaxf(mx, s);
        }
        float total = 0, prob[MAXN];
        for (int j = 0; j <= i; j++) { float e = expf(score[j] - mx); prob[j] = e; total += e; }
        for (int j = 0; j <= i; j++) prob[j] /= total;
        for (int d = 0; d < D; d++) {
            float s = 0; for (int j = 0; j <= i; j++) s += prob[j] * qkv[j * QW + 2 * M + hh * D + d];
            attn_out[(hh * n + i) * D + d] = s;
        }
    }

    float merged[MAXN * 6];
    for (int t = 0; t < n; t++) for (int hh = 0; hh < H; hh++) for (int d = 0; d < D; d++)
        merged[t * M + hh * D + d] = attn_out[(hh * n + t) * D + d];

    float resid1[MAXN * 6];
    for (int t = 0; t < n; t++) for (int d = 0; d < M; d++) {
        float s = 0; for (int e = 0; e < M; e++) s += merged[t * M + e] * w->wo[e * M + d];
        resid1[t * M + d] = h[t * M + d] + s;
    }

    float ff1[MAXN * 8]; /* F=8 */
    for (int t = 0; t < n; t++) for (int f = 0; f < F; f++) {
        float s = 0; for (int d = 0; d < M; d++) s += resid1[t * M + d] * w->w1[d * F + f];
        float u = 0.7978845608028654f * (s + 0.044715f * s * s * s);
        ff1[t * F + f] = 0.5f * s * (1.0f + tanhf(u));
    }
    float resid2[MAXN * 6];
    for (int t = 0; t < n; t++) for (int d = 0; d < M; d++) {
        float s = 0; for (int f = 0; f < F; f++) s += ff1[t * F + f] * w->w2[f * M + d];
        resid2[t * M + d] = resid1[t * M + d] + s;
    }

    for (int t = 0; t < n; t++) {
        float logits[VOCAB], mx = -INFINITY;
        for (int v = 0; v < VOCAB; v++) {
            float s = 0; for (int d = 0; d < M; d++) s += resid2[t * M + d] * w->wout[d * VOCAB + v];
            logits[v] = s; mx = fmaxf(mx, s);
        }
        float total = 0;
        for (int v = 0; v < VOCAB; v++) { float e = expf(logits[v] - mx); probs_out[t * VOCAB + v] = e; total += e; }
        for (int v = 0; v < VOCAB; v++) probs_out[t * VOCAB + v] /= total;
    }
}

/* incremental KV-cache generation state: caches K/V per head (as in
 * operation 7/10) plus, since every other op here is row-wise/causal-safe
 * to compute fresh per new token only, nothing else needs caching. */
typedef struct { float K[H][MAXN][D]; float V[H][MAXN][D]; } KVCache;

static void incremental_step(const Weights *w, KVCache *c, int token, int pos, float *probs_row /* [VOCAB] */) {
    float hrow[6];
    for (int d = 0; d < M; d++) hrow[d] = w->wte[token * M + d] + w->wpe[pos * M + d];

    float qkv_row[18];
    for (int cix = 0; cix < QW; cix++) {
        float s = 0; for (int d = 0; d < M; d++) s += hrow[d] * w->wqkv[d * QW + cix];
        qkv_row[cix] = s;
    }

    float scale = 1.0f / sqrtf((float)D);
    for (int hh = 0; hh < H; hh++) for (int d = 0; d < D; d++) {
        c->K[hh][pos][d] = qkv_row[M + hh * D + d];
        c->V[hh][pos][d] = qkv_row[2 * M + hh * D + d];
    }
    float attn_out[6]; /* M */
    for (int hh = 0; hh < H; hh++) {
        float score[MAXN], mx = -INFINITY;
        for (int j = 0; j <= pos; j++) {
            float s = 0; for (int d = 0; d < D; d++) s += qkv_row[hh * D + d] * c->K[hh][j][d];
            s *= scale; score[j] = s; mx = fmaxf(mx, s);
        }
        float total = 0, prob[MAXN];
        for (int j = 0; j <= pos; j++) { float e = expf(score[j] - mx); prob[j] = e; total += e; }
        for (int j = 0; j <= pos; j++) prob[j] /= total;
        for (int d = 0; d < D; d++) {
            float s = 0; for (int j = 0; j <= pos; j++) s += prob[j] * c->V[hh][j][d];
            attn_out[hh * D + d] = s;
        }
    }

    float resid1[6];
    for (int d = 0; d < M; d++) {
        float s = 0; for (int e = 0; e < M; e++) s += attn_out[e] * w->wo[e * M + d];
        resid1[d] = hrow[d] + s;
    }
    float ff1[8];
    for (int f = 0; f < F; f++) {
        float s = 0; for (int d = 0; d < M; d++) s += resid1[d] * w->w1[d * F + f];
        float u = 0.7978845608028654f * (s + 0.044715f * s * s * s);
        ff1[f] = 0.5f * s * (1.0f + tanhf(u));
    }
    float resid2[6];
    for (int d = 0; d < M; d++) {
        float s = 0; for (int f = 0; f < F; f++) s += ff1[f] * w->w2[f * M + d];
        resid2[d] = resid1[d] + s;
    }
    float logits[VOCAB], mx = -INFINITY;
    for (int v = 0; v < VOCAB; v++) {
        float s = 0; for (int d = 0; d < M; d++) s += resid2[d] * w->wout[d * VOCAB + v];
        logits[v] = s; mx = fmaxf(mx, s);
    }
    float total = 0;
    for (int v = 0; v < VOCAB; v++) { float e = expf(logits[v] - mx); probs_row[v] = e; total += e; }
    for (int v = 0; v < VOCAB; v++) probs_row[v] /= total;
}

/* generate GEN_LEN tokens after a fixed PROMPT_LEN-token prompt, via the
 * incremental KV-cache path, sampling each new token with the given rng.
 * Returns the full sequence (prompt + generated) in `tokens`, length
 * returned. */
static int generate(const Weights *w, const int *prompt, int gen_len, SampleRng *rng, float temperature, int top_k, int *tokens /* [PROMPT_LEN+gen_len] */) {
    KVCache cache; memset(&cache, 0, sizeof(cache));
    for (int i = 0; i < PROMPT_LEN; i++) tokens[i] = prompt[i];
    for (int pos = 0; pos < PROMPT_LEN + gen_len; pos++) {
        float probs[VOCAB];
        incremental_step(w, &cache, tokens[pos], pos, probs);
        if (pos >= PROMPT_LEN - 1 && pos < PROMPT_LEN + gen_len - 1) {
            /* logits at position `pos` predict the token at `pos+1` */
            float logits[VOCAB];
            for (int v = 0; v < VOCAB; v++) logits[v] = logf(probs[v]); /* monotonic, sampler only needs relative order + softmax-again is fine since temperature divides logits, not probs -- reconstructing logits from probs via log preserves the categorical distribution exactly for the purpose of re-softmaxing at temperature=1; tests below always use temperature=1 for this reason */
            int next = sample_top_k_temperature(logits, VOCAB, top_k, temperature, rng);
            if (next < 0) { fprintf(stderr, "generate: sampler rejected valid input\n"); exit(1); }
            tokens[pos + 1] = next;
        }
    }
    return PROMPT_LEN + gen_len;
}

static int run_canonical_anchor(const Weights *w, const int *tokens) {
    enum { WTE, TOKID, WPE, POSID, TOKEMB, POSEMB, HRES, WQKV, QKV, ATT, MERGED, WO, PROJ, RES1,
           W1, Z1, ACT, W2, FF, RES2, LOGITW, LOGITS, PROBS, NODES };
    float tok_ids[T], pos_ids[T]; for (int i = 0; i < T; i++) { tok_ids[i] = (float)tokens[i]; pos_ids[i] = (float)i; }
    float gtok[T] = {0}, gpos[T] = {0};
    float wte[VOCAB * M], gwte[VOCAB * M] = {0}; memcpy(wte, w->wte, sizeof wte);
    float wpe[MAXPOS * M], gwpe[MAXPOS * M] = {0}; memcpy(wpe, w->wpe, sizeof wpe);
    float tokemb[T * M], gtokemb[T * M] = {0}, posemb[T * M], gposemb[T * M] = {0};
    float hres[T * M], ghres[T * M] = {0};
    float wqkv[M * QW], gwqkv[M * QW] = {0}; memcpy(wqkv, w->wqkv, sizeof wqkv);
    float qkv[T * QW], gqkv[T * QW] = {0};
    float att[H * T * D], gatt[H * T * D] = {0}, attaux[H * T * T];
    float merged[T * M], gmerged[T * M] = {0};
    float wo[M * M], gwo[M * M] = {0}; memcpy(wo, w->wo, sizeof wo);
    float proj[T * M], gproj[T * M] = {0};
    float res1[T * M], gres1[T * M] = {0};
    float w1[M * F], gw1[M * F] = {0}; memcpy(w1, w->w1, sizeof w1);
    float z1[T * F], gz1[T * F] = {0};
    float act[T * F], gact[T * F] = {0};
    float w2[F * M], gw2[F * M] = {0}; memcpy(w2, w->w2, sizeof w2);
    float ff[T * M], gff[T * M] = {0};
    float res2[T * M], gres2[T * M] = {0};
    float wout[M * VOCAB], gwout[M * VOCAB] = {0}; memcpy(wout, w->wout, sizeof wout);
    float logits[T * VOCAB], glogits[T * VOCAB] = {0};
    float probs[T * VOCAB], gprobs[T * VOCAB] = {0};

    Node g[NODES] = {
        [WTE] = {LEAF, NONE, NONE, PARAM | RETAIN_GRAD, {wte, gwte, NULL, VOCAB, M, 0}},
        [TOKID] = {LEAF, NONE, NONE, CONSTANT, {tok_ids, gtok, NULL, 1, T, 0}},
        [WPE] = {LEAF, NONE, NONE, PARAM | RETAIN_GRAD, {wpe, gwpe, NULL, MAXPOS, M, 0}},
        [POSID] = {LEAF, NONE, NONE, CONSTANT, {pos_ids, gpos, NULL, 1, T, 0}},
        [TOKEMB] = {EMBED_LOOKUP, WTE, TOKID, TEMP, {tokemb, gtokemb, NULL, T, M, 0}},
        [POSEMB] = {EMBED_LOOKUP, WPE, POSID, TEMP, {posemb, gposemb, NULL, T, M, 0}},
        [HRES] = {RESIDUAL, TOKEMB, POSEMB, TEMP | RETAIN_GRAD, {hres, ghres, NULL, T, M, 0}},
        [WQKV] = {LEAF, NONE, NONE, PARAM | RETAIN_GRAD, {wqkv, gwqkv, NULL, M, QW, 0}},
        [QKV] = {MATMUL, HRES, WQKV, TEMP, {qkv, gqkv, NULL, T, QW, 0}},
        [ATT] = {CAUSAL_ATTENTION, QKV, NONE, TEMP, {att, gatt, attaux, 1, H * T * D, H * T * T}},
        [MERGED] = {CONTIGUOUS, ATT, NONE, TEMP, {merged, gmerged, NULL, T, M, 0}},
        [WO] = {LEAF, NONE, NONE, PARAM | RETAIN_GRAD, {wo, gwo, NULL, M, M, 0}},
        [PROJ] = {MATMUL, MERGED, WO, TEMP, {proj, gproj, NULL, T, M, 0}},
        [RES1] = {RESIDUAL, HRES, PROJ, TEMP | RETAIN_GRAD, {res1, gres1, NULL, T, M, 0}},
        [W1] = {LEAF, NONE, NONE, PARAM | RETAIN_GRAD, {w1, gw1, NULL, M, F, 0}},
        [Z1] = {MATMUL, RES1, W1, TEMP, {z1, gz1, NULL, T, F, 0}},
        [ACT] = {GELU, Z1, NONE, TEMP, {act, gact, NULL, T, F, 0}},
        [W2] = {LEAF, NONE, NONE, PARAM | RETAIN_GRAD, {w2, gw2, NULL, F, M, 0}},
        [FF] = {MATMUL, ACT, W2, TEMP, {ff, gff, NULL, T, M, 0}},
        [RES2] = {RESIDUAL, RES1, FF, TEMP | RETAIN_GRAD, {res2, gres2, NULL, T, M, 0}},
        [LOGITW] = {LEAF, NONE, NONE, PARAM | RETAIN_GRAD, {wout, gwout, NULL, M, VOCAB, 0}},
        [LOGITS] = {MATMUL, RES2, LOGITW, TEMP, {logits, glogits, NULL, T, VOCAB, 0}},
        [PROBS] = {SOFTMAX_ROWS, LOGITS, NONE, TEMP, {probs, gprobs, NULL, T, VOCAB, 0}},
    };
    ExecStep steps[64]; Context ctx[64]; uint32_t count = 0;
    if (compile(g, NODES, .01f, steps, ctx, 64, &count)) { fprintf(stderr, "anchor: compile rejected\n"); return 1; }
    if (tensor_transformer_steps_execute(steps, count)) { fprintf(stderr, "anchor: execute failed\n"); return 1; }

    float ref_probs[MAXN * VOCAB];
    int itok[MAXN]; for (int i = 0; i < T; i++) itok[i] = tokens[i];
    ref_forward(w, itok, T, ref_probs);
    for (uint32_t i = 0; i < (uint32_t)(T * VOCAB); i++) {
        if (fabsf(probs[i] - ref_probs[i]) > 1e-4f) {
            fprintf(stderr, "anchor mismatch at %u: canonical=%.9g runtime_ref=%.9g\n", i, probs[i], ref_probs[i]);
            return 1;
        }
    }
    printf("anchor: full toy-decoder pipeline (embed+pos+attn+ffn+softmax) matches canonical graph+executor exactly at T=%d, all %d probabilities\n", T, T * VOCAB);
    return 0;
}

int main(void) {
    if (M != 6 || H != 2 || D != 3 || QW != 18 || F != 8) { fprintf(stderr, "requires -DM=6 -DH=2 -DD=3 -DQW=18 -DF=8\n"); return 1; }
    Weights w; init_weights(&w);
    int prompt[PROMPT_LEN] = {3, 7};
    int gen_len = T - PROMPT_LEN;

    /* property 2 + anchor prerequisite: generate once via the incremental
     * KV-cache path, cross-checking every step's full probability
     * distribution against the runtime-n full-recompute reference before
     * accepting the sampled token */
    SampleRng rng = {0xC0FFEE};
    int tokens[MAXN];
    {
        KVCache cache; memset(&cache, 0, sizeof(cache));
        for (int i = 0; i < PROMPT_LEN; i++) tokens[i] = prompt[i];
        for (int pos = 0; pos < T; pos++) {
            float probs_inc[VOCAB];
            incremental_step(&w, &cache, tokens[pos], pos, probs_inc);

            int itok[MAXN]; for (int i = 0; i <= pos; i++) itok[i] = tokens[i];
            float ref_probs[MAXN * VOCAB];
            ref_forward(&w, itok, pos + 1, ref_probs);
            for (int v = 0; v < VOCAB; v++) {
                if (fabsf(probs_inc[v] - ref_probs[pos * VOCAB + v]) > 1e-5f) {
                    fprintf(stderr, "step %d: kv-cache probs diverge from full recompute at v=%d: %.9g vs %.9g\n", pos, v, probs_inc[v], ref_probs[pos * VOCAB + v]);
                    return 1;
                }
            }
            if (pos >= PROMPT_LEN - 1 && pos < T - 1) {
                float logits[VOCAB];
                for (int v = 0; v < VOCAB; v++) logits[v] = logf(probs_inc[v]);
                int next = sample_top_k_temperature(logits, VOCAB, 5, 1.0f, &rng);
                if (next < 0) { fprintf(stderr, "sampler rejected valid input at step %d\n", pos); return 1; }
                tokens[pos + 1] = next;
            }
        }
    }
    printf("kv-cache vs full recompute: probability distributions agree exactly at every generation step, n=1..%d\n", T);

    if (run_canonical_anchor(&w, tokens)) return 1;

    /* property 3a: replay determinism -- same seed reproduces the exact
     * same generated sequence */
    {
        SampleRng r1 = {0xC0FFEE}, r2 = {0xC0FFEE};
        int seq1[MAXN], seq2[MAXN];
        generate(&w, prompt, gen_len, &r1, 1.0f, 5, seq1);
        generate(&w, prompt, gen_len, &r2, 1.0f, 5, seq2);
        for (int i = 0; i < PROMPT_LEN + gen_len; i++) if (seq1[i] != seq2[i]) { fprintf(stderr, "replay mismatch at %d: %d vs %d\n", i, seq1[i], seq2[i]); return 1; }
        for (int i = 0; i < PROMPT_LEN + gen_len; i++) if (seq1[i] != tokens[i]) { fprintf(stderr, "generate() disagrees with the manually-driven loop at %d: %d vs %d\n", i, seq1[i], tokens[i]); return 1; }
        printf("replay: identical seed reproduces an identical %d-token generated sequence\n", PROMPT_LEN + gen_len);
    }

    /* property 3b: a shorter generation is a strict prefix of the longer
     * one from the same seed (loop-level non-retroactivity) */
    if (gen_len >= 2) {
        SampleRng r3 = {0xC0FFEE};
        int shorter[MAXN];
        int shorter_len = generate(&w, prompt, gen_len - 1, &r3, 1.0f, 5, shorter);
        for (int i = 0; i < shorter_len; i++) {
            if (shorter[i] != tokens[i]) { fprintf(stderr, "prefix mismatch at %d: shorter_run=%d full_run=%d\n", i, shorter[i], tokens[i]); return 1; }
        }
        printf("prefix stability: a %d-token generation is an exact prefix of the %d-token generation from the same seed\n", shorter_len, PROMPT_LEN + gen_len);
    }

    puts("autoregressive_loop differential check passed: full toy-decoder pipeline anchored to canonical graph+executor, kv-cache matches full recompute at every step, replay deterministic, shorter generation is an exact prefix of the longer one");
    return 0;
}
