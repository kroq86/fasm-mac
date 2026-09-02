/* Focused diagnostic: run the REAL canonical decode path (setup once,
 * execute many, same as tensor_gpt2_cached_generation_differential_check.c
 * and bench_sustained.c) for many sequential steps, and after EVERY step
 * compare the cache's just-written K/V row (layer 0) against an
 * independently, freshly computed expected K/V for that exact position
 * (same formula, computed from scratch, no persisted state) -- to find
 * the first position where the live incremental cache's content diverges
 * from what it should contain.
 */
#include "tensor_semantic_compiler.h"
#include "tensor_gpt2_forward.h"
#include <stdio.h>
#include <stdlib.h>

#if M != 768 || H != 12 || D != 64 || QW != 2304 || MAXCACHE < 128
#error "requires -DM=768 -DH=12 -DD=64 -DQW=2304 -DMAXCACHE>=128"
#endif

typedef struct {
    Node g[3]; ExecStep steps[8]; Context ctx[8]; uint32_t count;
    float x[768 * 3], gx[768 * 3];
    float cache_data[MAXCACHE * 768], cache_grad[MAXCACHE * 768], cache_aux[MAXCACHE * 768];
    float out[768], gout[768];
} LayerAttnGraph;
static LayerAttnGraph lag0;
enum { LAG_X, LAG_CACHE, LAG_ATT, LAG_NODES };
static int setup(LayerAttnGraph *lag) {
    memset(lag, 0, sizeof *lag);
    lag->g[LAG_X] = (Node){LEAF, NONE, NONE, INPUT, {lag->x, lag->gx, NULL, 1, QW, 0}};
    lag->g[LAG_CACHE] = (Node){LEAF, NONE, NONE, INPUT, {lag->cache_data, lag->cache_grad, lag->cache_aux, (uint32_t)MAXCACHE, (uint32_t)M, 0}};
    lag->g[LAG_ATT] = (Node){CAUSAL_ATTENTION_CACHED, LAG_X, LAG_CACHE, TEMP, {lag->out, lag->gout, NULL, 1, M, 0}};
    return compile(lag->g, LAG_NODES, .01f, lag->steps, lag->ctx, 8, &lag->count);
}

int main(int argc, char **argv) {
    const char *safetensors_path = argc > 1 ? argv[1] : "/Users/ll/.cache/huggingface/hub/models--gpt2/snapshots/607a30d783dfa663caf39e06633721c8d4cfcd7e/model.safetensors";
    static const char *EXPECT_SHA = "248dfc3911869ec493c76e65bf2fcf7f615828b0254c12b473182f0f81d3a707";
    static Gpt2Weights w;
    if (gpt2_load_weights(&w, safetensors_path, EXPECT_SHA)) { fprintf(stderr, "load failed\n"); return 1; }
    if (setup(&lag0)) { fprintf(stderr, "setup failed\n"); return 1; }

    int nsteps = argc > 2 ? atoi(argv[2]) : 40;
    static int tokens[256];
    for (int i = 0; i < nsteps; i++) tokens[i] = (i * 37 + 101) % 50257; /* same deterministic sequence style as bench_algorithm.c */

    const Gpt2BlockWeights *bw = &w.blk[0];

    for (int pos = 0; pos < nsteps; pos++) {
        /* live path: exactly what gpt2_decode_step_cached does for layer 0 */
        float h[768];
        for (int m = 0; m < 768; m++) h[m] = w.wte[tokens[pos] * 768 + m] + w.wpe[pos * 768 + m];
        float ln1[768]; gpt2_layernorm_raw(h, 1, 768, ln1);
        float qkv[2304]; gpt2_matmul_bias(ln1, 1, 768, bw->attn_w, bw->attn_b, 2304, qkv);
        memcpy(lag0.x, qkv, sizeof(float) * 2304);
        if (tensor_transformer_steps_execute(lag0.steps, lag0.count)) { fprintf(stderr, "execute failed at pos=%d\n", pos); return 1; }

        /* what the kernel SHOULD have just written into cache_data/cache_aux at row `pos` */
        float expected_k[768], expected_v[768];
        memcpy(expected_k, qkv + 768, sizeof expected_k);   /* K portion of this step's qkv, offset M=768 */
        memcpy(expected_v, qkv + 2 * 768, sizeof expected_v); /* V portion, offset 2M=1536 */

        float max_k_diff = 0, max_v_diff = 0;
        for (int d = 0; d < 768; d++) {
            float dk = fabsf(lag0.cache_data[pos * 768 + d] - expected_k[d]);
            float dv = fabsf(lag0.cache_aux[pos * 768 + d] - expected_v[d]);
            if (dk > max_k_diff) max_k_diff = dk;
            if (dv > max_v_diff) max_v_diff = dv;
        }
        if (max_k_diff > 0 || max_v_diff > 0) {
            printf("DRIFT at pos=%d: max_k_diff=%.9g max_v_diff=%.9g aux_count_after=%u\n", pos, max_k_diff, max_v_diff, lag0.g[LAG_CACHE].tensor.aux_count);
            /* keep going to see if it's isolated or persistent/growing */
        } else if (pos % 8 == 0 || pos == nsteps - 1) {
            printf("ok pos=%d aux_count_after=%u\n", pos, lag0.g[LAG_CACHE].tensor.aux_count);
        }
    }
    puts("diag_cache_drift done");
    return 0;
}
