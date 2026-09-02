/* Decoder-runtime roadmap item 4: differential check for the new
 * CAUSAL_ATTENTION_CACHED canonical op (tensor_semantic_compiler.h) --
 * the planner-visible, in-graph alternative to a hand-written KV cache
 * (fasm/spikes/tensor_kv_cache_differential_check.c's earlier toy spike,
 * and tensor_gpt2_forward.h's real-model runtime-n forward pass).
 *
 * Three-way cross-check, each step incrementing the SAME cache tensor
 * through repeated compile()+execute() calls (T=1 graph every time):
 *   1. vs the hand-rolled runtime-n full-recompute reference this project
 *      already anchored against the real canonical CAUSAL_ATTENTION op at
 *      T=7/8/9 (tensor_kv_cache_differential_check.c's
 *      ref_full_causal_attention_runtime, reproduced here identically --
 *      not shared by #include, to keep this file's oracle independent of
 *      that file's own bookkeeping).
 *   2. vs that same file's hand-rolled INCREMENTAL kv_cache_append
 *      function (already itself proven equal to #1 at every step) --
 *      confirming the new canonical op matches not just the math but the
 *      toy spike's own incremental bookkeeping pattern.
 *   3. Structural: the cache Tensor's `aux_count` field (read as "pos" by the
 *      kernel, incremented as its one documented side effect) is checked
 *      to advance by exactly 1 per call, and non-retroactivity (a
 *      shorter run's outputs are unaffected by whether more tokens are
 *      appended afterward) is checked the same way the toy spike checked
 *      it.
 *   4. Fail-closed: validate() must reject a cache already at MAXCACHE
 *      capacity (aux_count >= rows==MAXCACHE) -- exercised directly, not just
 *      documented.
 *
 * Built at T=7/8/9 (repurposed here as the number of incremental steps,
 * not a real T macro use -- CAUSAL_ATTENTION_CACHED's own graphs are
 * always T=1), H=2, D=3, M=6, QW=18, MAXCACHE=9 (>= the largest step
 * count used), matching this project's existing causal-attention/kv-cache
 * checks' shape convention.
 *
 * Two real design bugs this check caught before it passed (both in the
 * new op, tensor_semantic_compiler.h -- caught here, fixed there):
 *   1. validate()'s blanket "every tensor has rows>=1" rule rejects a
 *      brand-new empty cache (rows=0) before the op's own validate()
 *      case ever runs. Fixed by making the cache tensor's `rows` field
 *      always the FIXED CAPACITY (MAXCACHE, never zero) and moving the
 *      mutable "current position" counter to `aux_count` instead.
 *   2. The cache leaf was first flagged PARAM (seemed natural: "a leaf
 *      the kernel writes into"), but validate() treats any PARAM leaf
 *      with a non-null `aux` as carrying `ParameterOptimizationMetadata`
 *      (expects `aux_count==1` and a valid `lr_multiplier`) -- an
 *      unrelated existing convention this cache's aux (the V-buffer)
 *      collided with, rejected with validate() rc=-4. Fixed by using
 *      INPUT instead: this cache is inference-only state, never an
 *      optimizer target, so INPUT (no backward, no optimizer step) is
 *      the correct flag regardless of the PARAM collision.
 */
#include "tensor_semantic_compiler.h"
#include <stdio.h>
#include <stdlib.h>

enum { MAXN = 9 };

static float dv(int i, int salt) {
    float a = sinf((float)(i * 12.9898f + salt * 78.233f)) * 43758.5453f;
    return a - floorf(a) - 0.5f;
}

static void ref_full_causal_attention_runtime(const float *qkv, int n, float *out) {
    float scale = 1.0f / sqrtf((float)D);
    for (int h = 0; h < H; h++) for (int i = 0; i < n; i++) {
        float score[MAXN], mx = -INFINITY;
        for (int j = 0; j <= i; j++) {
            float s = 0;
            for (int d = 0; d < D; d++) s += qkv[i * QW + h * D + d] * qkv[j * QW + M + h * D + d];
            s *= scale; score[j] = s; mx = fmaxf(mx, s);
        }
        float total = 0, prob[MAXN];
        for (int j = 0; j <= i; j++) { float e = expf(score[j] - mx); prob[j] = e; total += e; }
        for (int j = 0; j <= i; j++) prob[j] /= total;
        for (int d = 0; d < D; d++) {
            float s = 0; for (int j = 0; j <= i; j++) s += prob[j] * qkv[j * QW + 2 * M + h * D + d];
            out[(h * n + i) * D + d] = s;
        }
    }
}

typedef struct { float K[H][MAXN][D]; float V[H][MAXN][D]; } ToyKVCache;
static void toy_kv_cache_append(ToyKVCache *c, const float *qkv_row_i, int i, float *out_row) {
    float scale = 1.0f / sqrtf((float)D);
    for (int h = 0; h < H; h++) for (int d = 0; d < D; d++) {
        c->K[h][i][d] = qkv_row_i[M + h * D + d];
        c->V[h][i][d] = qkv_row_i[2 * M + h * D + d];
    }
    for (int h = 0; h < H; h++) {
        float score[MAXN], mx = -INFINITY;
        for (int j = 0; j <= i; j++) {
            float s = 0; for (int d = 0; d < D; d++) s += qkv_row_i[h * D + d] * c->K[h][j][d];
            s *= scale; score[j] = s; mx = fmaxf(mx, s);
        }
        float total = 0, prob[MAXN];
        for (int j = 0; j <= i; j++) { float e = expf(score[j] - mx); prob[j] = e; total += e; }
        for (int j = 0; j <= i; j++) prob[j] /= total;
        for (int d = 0; d < D; d++) {
            float s = 0; for (int j = 0; j <= i; j++) s += prob[j] * c->V[h][j][d];
            out_row[h * D + d] = s;
        }
    }
}

/* one T=1 compile()+execute() call against the real canonical op */
static int canonical_cached_step(const float *qkv_row, Tensor *cache_tensor, float *out_row) {
    enum { X, CACHE, ATT, NODES };
    float x[QW], gx[QW] = {0};
    memcpy(x, qkv_row, sizeof(float) * QW);
    float gcache_data[MAXCACHE * M] = {0};
    float out[M], gout[M] = {0};
    Node g[NODES] = {
        [X] = {LEAF, NONE, NONE, INPUT, {x, gx, NULL, 1, QW, 0}},
        [CACHE] = {LEAF, NONE, NONE, INPUT, {cache_tensor->data, gcache_data, cache_tensor->aux, (uint32_t)MAXCACHE, (uint32_t)M, cache_tensor->aux_count}},
        [ATT] = {CAUSAL_ATTENTION_CACHED, X, CACHE, TEMP, {out, gout, NULL, 1, M, 0}},
    };
    ExecStep steps[8]; Context ctx[8]; uint32_t count = 0;
    if (compile(g, NODES, .01f, steps, ctx, 8, &count)) { fprintf(stderr, "compile rejected\n"); return -1; }
    if (tensor_transformer_steps_execute(steps, count)) { fprintf(stderr, "execute failed\n"); return -1; }
    memcpy(out_row, out, sizeof(float) * M);
    cache_tensor->aux_count = g[CACHE].tensor.aux_count; /* the kernel's side effect: advanced pos */
    return 0;
}

int main(void) {
    if (H != 2 || D != 3 || M != 6 || QW != 18 || MAXCACHE < T) { fprintf(stderr, "requires -DH=2 -DD=3 -DM=6 -DQW=18 -DMAXCACHE>=T\n"); return 1; }

    float qkv[MAXN * 18];
    for (int i = 0; i < T * QW; i++) qkv[i] = dv(i, 5) * 3.0f;
    for (int d = 0; d < QW; d++) qkv[(T / 2) * QW + d] = dv(d, 99) * 15.0f; /* numerical tail */

    static float cache_K[MAXCACHE * 6], cache_V[MAXCACHE * 6];
    Tensor cache_tensor = {cache_K, NULL, cache_V, MAXCACHE, M, 0};
    ToyKVCache toy; memset(&toy, 0, sizeof toy);

    for (int i = 0; i < T; i++) {
        float canon_out[6], toy_out[6];
        if (canonical_cached_step(&qkv[i * QW], &cache_tensor, canon_out)) return 1;
        toy_kv_cache_append(&toy, &qkv[i * QW], i, toy_out);

        float ref[H * MAXN * D];
        ref_full_causal_attention_runtime(qkv, i + 1, ref);

        for (int h = 0; h < H; h++) for (int d = 0; d < D; d++) {
            float want_ref = ref[(h * (i + 1) + i) * D + d];
            float want_toy = toy_out[h * D + d];
            float got = canon_out[h * D + d];
            if (fabsf(got - want_ref) > 1e-5f) { fprintf(stderr, "step %d h=%d d=%d: canonical=%.9g full_recompute_ref=%.9g\n", i, h, d, got, want_ref); return 1; }
            if (fabsf(got - want_toy) > 1e-5f) { fprintf(stderr, "step %d h=%d d=%d: canonical=%.9g toy_incremental=%.9g\n", i, h, d, got, want_toy); return 1; }
        }
        if (cache_tensor.aux_count != (uint32_t)(i + 1)) { fprintf(stderr, "step %d: cache.aux_count=%u, expected %d (side-effect advance failed)\n", i, cache_tensor.aux_count, i + 1); return 1; }
    }
    printf("CAUSAL_ATTENTION_CACHED: agrees with the full-recompute reference AND the toy incremental spike at every step, n=1..%d; cache.aux_count advances correctly\n", T);

    /* non-retroactivity: a shorter run's outputs must be unaffected by
     * whether more tokens are appended afterward */
    {
        int shorter = T > 2 ? T - 2 : 1;
        static float cache_K2[MAXCACHE * 6], cache_V2[MAXCACHE * 6];
        Tensor cache2 = {cache_K2, NULL, cache_V2, MAXCACHE, M, 0};
        float saved[MAXN][6];
        for (int i = 0; i < shorter; i++) {
            float out_row[6];
            if (canonical_cached_step(&qkv[i * QW], &cache2, out_row)) return 1;
            memcpy(saved[i], out_row, sizeof out_row);
        }
        /* recompute the FULL T-length run fresh for a fair prefix comparison */
        static float cache_K3[MAXCACHE * 6], cache_V3[MAXCACHE * 6];
        Tensor cache3 = {cache_K3, NULL, cache_V3, MAXCACHE, M, 0};
        float full_out[MAXN][6];
        for (int i = 0; i < T; i++) {
            float out_row[6];
            if (canonical_cached_step(&qkv[i * QW], &cache3, out_row)) return 1;
            memcpy(full_out[i], out_row, sizeof out_row);
        }
        for (int i = 0; i < shorter; i++) for (int k = 0; k < 6; k++) {
            if (saved[i][k] != full_out[i][k]) { fprintf(stderr, "non-retroactivity violated at step %d k=%d\n", i, k); return 1; }
        }
        printf("non-retroactivity: outputs for positions 0..%d identical whether %d or %d total tokens are eventually appended\n", shorter - 1, shorter, T);
    }

    /* persistent-graph pattern: compile() ONCE, then just overwrite the
     * qkv input buffer and call tensor_transformer_steps_execute() again
     * for each new token -- no Node array rebuild, no external copy of
     * the position counter in/out (unlike canonical_cached_step above,
     * which rebuilds the graph every call for this file's own
     * per-call-differencing test convenience). This is the pattern the
     * real GPT-2 integration (tensor_gpt2_cached_generation_differential_check.c)
     * actually uses, so it needs its own direct verification here: must
     * produce IDENTICAL results to the rebuild-every-call pattern already
     * proven above. */
    {
        enum { X, CACHE, ATT, NODES };
        static float px[18], pgx[18] = {0};
        static float pcache_data[MAXCACHE * 6], pcache_grad[MAXCACHE * 6] = {0}, pcache_aux[MAXCACHE * 6];
        static float pout[6], pgout[6] = {0};
        Node pg[NODES] = {
            [X] = {LEAF, NONE, NONE, INPUT, {px, pgx, NULL, 1, QW, 0}},
            [CACHE] = {LEAF, NONE, NONE, INPUT, {pcache_data, pcache_grad, pcache_aux, (uint32_t)MAXCACHE, (uint32_t)M, 0}},
            [ATT] = {CAUSAL_ATTENTION_CACHED, X, CACHE, TEMP, {pout, pgout, NULL, 1, M, 0}},
        };
        ExecStep psteps[8]; Context pctx[8]; uint32_t pcount = 0;
        if (compile(pg, NODES, .01f, psteps, pctx, 8, &pcount)) { fprintf(stderr, "persistent-graph: compile rejected\n"); return 1; }

        for (int i = 0; i < T; i++) {
            memcpy(px, &qkv[i * QW], sizeof(float) * QW);
            if (tensor_transformer_steps_execute(psteps, pcount)) { fprintf(stderr, "persistent-graph: execute failed at step %d\n", i); return 1; }
            if (pg[CACHE].tensor.aux_count != (uint32_t)(i + 1)) { fprintf(stderr, "persistent-graph: aux_count=%u expected %d\n", pg[CACHE].tensor.aux_count, i + 1); return 1; }

            float ref[H * MAXN * D];
            ref_full_causal_attention_runtime(qkv, i + 1, ref);
            for (int h = 0; h < H; h++) for (int d = 0; d < D; d++) {
                float want = ref[(h * (i + 1) + i) * D + d];
                float got = pout[h * D + d];
                if (fabsf(got - want) > 1e-5f) { fprintf(stderr, "persistent-graph step %d h=%d d=%d: got=%.9g want=%.9g\n", i, h, d, got, want); return 1; }
            }
        }
        printf("persistent-graph pattern (compile once, execute many, no external position copy): matches the full-recompute reference exactly at every step, n=1..%d -- the actual pattern used by the real GPT-2 integration\n", T);
    }

    /* fail-closed: cache at capacity must be rejected */
    {
        enum { X, CACHE, ATT, NODES };
        float x[QW] = {0}, gx[QW] = {0};
        static float full_K[MAXCACHE * 6], full_V[MAXCACHE * 6], gfull[MAXCACHE * 6] = {0};
        float out[6], gout[6] = {0};
        Node g[NODES] = {
            [X] = {LEAF, NONE, NONE, INPUT, {x, gx, NULL, 1, QW, 0}},
            [CACHE] = {LEAF, NONE, NONE, INPUT, {full_K, gfull, full_V, (uint32_t)MAXCACHE, (uint32_t)M, (uint32_t)MAXCACHE}},
            [ATT] = {CAUSAL_ATTENTION_CACHED, X, CACHE, TEMP, {out, gout, NULL, 1, M, 0}},
        };
        ExecStep steps[8]; Context ctx[8]; uint32_t count = 0;
        int rc = compile(g, NODES, .01f, steps, ctx, 8, &count);
        if (rc == 0) { fprintf(stderr, "expected compile rejection for a full cache (rows=MAXCACHE), got success\n"); return 1; }
        printf("fail-closed: a cache already at MAXCACHE capacity is correctly rejected by validate()\n");
    }

    puts("causal_attention_cached differential check passed: canonical CAUSAL_ATTENTION_CACHED matches the full-recompute reference and the toy incremental spike exactly at every step, cache-state side effect correct, non-retroactivity confirmed, capacity-exceeded fails closed");
    return 0;
}
