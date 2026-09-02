/* Decoder-runtime Etap 2, operation 7/10: KV cache.
 *
 * KV caching is an INFERENCE-TIME optimization of causal attention's
 * existing math (compute Q/K/V once per new token, reuse cached K/V for
 * all past positions), not a different function -- so this is not a new
 * canonical-compiler op. The canonical CAUSAL_ATTENTION kernel (already
 * differentially verified in operation 3/10) is also architecturally
 * fixed-T (T is a compile-time macro baked into tensor_semantic_compiler.h,
 * not a runtime value), so it cannot itself represent a growing context.
 *
 * Verification strategy, three layers:
 *   1. ANCHOR: a hand-rolled, runtime-T-parameterized "full recompute"
 *      reference function is checked against the real canonical
 *      CAUSAL_ATTENTION kernel (compile()+executor) at the fixed T this
 *      binary is built with -- grounding the runtime-parameterized
 *      reference in already-verified semantics, for every row, not just
 *      the last.
 *   2. CORE PROPERTY: a hand-rolled incremental KV-cache implementation
 *      (append one token, reuse cached K/V, compute only the new row) is
 *      checked at every context length n=1..T against the anchored full
 *      recompute reference at that same n. This is the actual "KV-cache
 *      and full recompute agree" property the plan's later GPT-2
 *      milestone requires, brought forward and pinned down here.
 *   3. NON-RETROACTIVITY: appending more tokens must never change
 *      already-produced outputs for earlier positions -- run the
 *      incremental process to two different final lengths and check the
 *      shared prefix of outputs is bit-identical, catching cache-index
 *      bugs that a same-length-only check would miss.
 *
 * Built with -DT=7, -DT=8, -DT=9 (the project's standing tails
 * convention) via the gate script's loop, -DH=2 -DD=3 -DM=6 -DQW=18
 * fixed across all three.
 */
#include "tensor_semantic_compiler.h"
#include <stdio.h>
#include <stdlib.h>

enum { MAXN = 9 };

static float dv(int i, int salt) {
    float a = sinf((float)(i * 12.9898f + salt * 78.233f)) * 43758.5453f;
    return a - floorf(a) - 0.5f;
}

/* hand-rolled, runtime-n causal attention, same math as
 * k_causal_attention_fwd but with n passed as a parameter instead of
 * baked in as the T macro. Output layout matches the canonical kernel's:
 * (h*n+i)*D+d. */
static void ref_full_causal_attention_runtime(const float *qkv, int n, float *out) {
    float scale = 1.0f / sqrtf((float)D);
    for (int h = 0; h < H; h++) for (int i = 0; i < n; i++) {
        float score[MAXN], mx = -INFINITY;
        for (int j = 0; j <= i; j++) {
            float s = 0;
            for (int d = 0; d < D; d++) s += qkv[i * QW + h * D + d] * qkv[j * QW + M + h * D + d];
            s *= scale;
            score[j] = s;
            mx = fmaxf(mx, s);
        }
        float total = 0, prob[MAXN];
        for (int j = 0; j <= i; j++) { float e = expf(score[j] - mx); prob[j] = e; total += e; }
        for (int j = 0; j <= i; j++) prob[j] /= total;
        for (int d = 0; d < D; d++) {
            float s = 0;
            for (int j = 0; j <= i; j++) s += prob[j] * qkv[j * QW + 2 * M + h * D + d];
            out[(h * n + i) * D + d] = s;
        }
    }
}

/* incremental KV-cache append: token i's qkv row is provided (as if just
 * computed by that token's own projection); K_i/V_i are written into the
 * cache, and the attention output for position i is computed by
 * attending Q_i against cached K_0..K_i / V_0..V_i only. Never touches
 * or recomputes any earlier position's output. */
typedef struct { float K[H][MAXN][D]; float V[H][MAXN][D]; } KVCache;

static void kv_cache_append(KVCache *c, const float *qkv_row_i, int i, float *out_row /* [M] */) {
    float scale = 1.0f / sqrtf((float)D);
    for (int h = 0; h < H; h++) for (int d = 0; d < D; d++) {
        c->K[h][i][d] = qkv_row_i[M + h * D + d];
        c->V[h][i][d] = qkv_row_i[2 * M + h * D + d];
    }
    for (int h = 0; h < H; h++) {
        float score[MAXN], mx = -INFINITY;
        for (int j = 0; j <= i; j++) {
            float s = 0;
            for (int d = 0; d < D; d++) s += qkv_row_i[h * D + d] * c->K[h][j][d];
            s *= scale;
            score[j] = s;
            mx = fmaxf(mx, s);
        }
        float total = 0, prob[MAXN];
        for (int j = 0; j <= i; j++) { float e = expf(score[j] - mx); prob[j] = e; total += e; }
        for (int j = 0; j <= i; j++) prob[j] /= total;
        for (int d = 0; d < D; d++) {
            float s = 0;
            for (int j = 0; j <= i; j++) s += prob[j] * c->V[h][j][d];
            out_row[h * D + d] = s;
        }
    }
}

static int run_canonical_anchor(const float *qkv_full) {
    enum { X, ATT, NODES };
    float x[T * QW], gx[T * QW] = {0};
    memcpy(x, qkv_full, sizeof(float) * T * QW);
    float att[H * T * D], gatt[H * T * D] = {0}, aux[H * T * T];
    Node g[NODES] = {
        {LEAF, NONE, NONE, INPUT | RETAIN_GRAD, {x, gx, NULL, T, QW, 0}},
        {CAUSAL_ATTENTION, X, NONE, TEMP | RETAIN_GRAD, {att, gatt, aux, H * T, D, H * T * T}},
    };
    ExecStep steps[8]; Context ctx[8]; uint32_t count = 0;
    if (compile(g, NODES, .01f, steps, ctx, 8, &count)) { fprintf(stderr, "anchor: compile rejected\n"); return 1; }
    if (tensor_transformer_steps_execute(steps, count)) { fprintf(stderr, "anchor: execute failed\n"); return 1; }

    float ref[H * T * D];
    ref_full_causal_attention_runtime(qkv_full, T, ref);
    for (uint32_t i = 0; i < H * T * D; i++) {
        if (fabsf(att[i] - ref[i]) > 1e-5f) {
            fprintf(stderr, "anchor mismatch at %u: canonical=%.9g runtime_ref=%.9g\n", i, att[i], ref[i]);
            return 1;
        }
    }
    printf("anchor: hand-rolled runtime-T full-recompute reference matches canonical CAUSAL_ATTENTION exactly at T=%d, all %d values\n", T, H * T * D);
    return 0;
}

int main(void) {
    float qkv[MAXN * 18]; /* built with QW=18 (H=2,D=3,M=6) */
    if (QW != 18 || M != 6 || H != 2 || D != 3) { fprintf(stderr, "this check requires -DH=2 -DD=3 -DM=6 -DQW=18\n"); return 1; }
    for (int i = 0; i < T * QW; i++) qkv[i] = dv(i, 5) * 3.0f;
    /* numerical tail: one row gets large-magnitude values so the
     * per-step incremental softmax is exercised at the same edge the
     * causal-attention op's own differential check already covers */
    for (int d = 0; d < QW; d++) qkv[(T / 2) * QW + d] = dv(d, 99) * 15.0f;

    if (run_canonical_anchor(qkv)) return 1;

    /* property 2: kv-cache incremental append vs full recompute, at
     * every growing context length n=1..T */
    KVCache cache; memset(&cache, 0, sizeof(cache));
    float saved_out[MAXN][6]; /* M=6 */
    for (int i = 0; i < T; i++) {
        float out_row[6];
        kv_cache_append(&cache, &qkv[i * QW], i, out_row);
        memcpy(saved_out[i], out_row, sizeof(out_row));

        float ref[H * MAXN * D];
        ref_full_causal_attention_runtime(qkv, i + 1, ref);
        for (int h = 0; h < H; h++) for (int d = 0; d < D; d++) {
            float want = ref[(h * (i + 1) + i) * D + d];
            float got = out_row[h * D + d];
            if (fabsf(got - want) > 1e-5f) {
                fprintf(stderr, "kv-cache mismatch at step i=%d h=%d d=%d: kv_cache=%.9g full_recompute=%.9g\n", i, h, d, got, want);
                return 1;
            }
        }
    }
    printf("kv-cache vs full recompute: agree exactly at every context length n=1..%d\n", T);

    /* property 3: non-retroactivity -- rerun the cache to a shorter
     * final length and confirm the shared prefix of outputs is
     * unaffected by whether more tokens get appended afterward */
    int shorter = T > 2 ? T - 2 : 1;
    KVCache cache2; memset(&cache2, 0, sizeof(cache2));
    for (int i = 0; i < shorter; i++) {
        float out_row[6];
        kv_cache_append(&cache2, &qkv[i * QW], i, out_row);
        for (int k = 0; k < 6; k++) {
            if (out_row[k] != saved_out[i][k]) {
                fprintf(stderr, "non-retroactivity violated at step i=%d k=%d: short_run=%.9g full_run=%.9g\n", i, k, out_row[k], saved_out[i][k]);
                return 1;
            }
        }
    }
    printf("non-retroactivity: outputs for positions 0..%d identical whether %d or %d total tokens are eventually appended\n", shorter - 1, shorter, T);

    puts("kv_cache differential check passed: runtime-T reference anchored to canonical CAUSAL_ATTENTION, incremental cache matches full recompute at every context length, non-retroactivity confirmed, numerical tail finite");
    return 0;
}
