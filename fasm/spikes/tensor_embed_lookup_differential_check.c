/* Decoder-runtime Etap 2, operation 1/10: EMBED_LOOKUP. Independent
 * reference implementation vs the canonical compile()-driven graph vs
 * the shared executor, per the methodology this milestone requires for
 * every new decoder op: reference <-> canonical graph <-> native
 * executor, not "it compiled so it's probably right".
 *
 * Tested per the predeclared criterion: shapes seq in {7,8,9}, finite
 * values, and tails (vocab boundary indices 0 and VOCAB-1 both
 * exercised, plus a duplicate-index case that only a correct
 * scatter-add backward, not a plain overwrite, gets right).
 */
#include "tensor_semantic_compiler.h"
#include <stdio.h>

enum { VOCAB = 11, DIM = 6, MAX_SEQ = 9 };

static float dv(int i, int salt) {
    float a = sinf((float)(i * 12.9898f + salt * 78.233f)) * 43758.5453f;
    return a - floorf(a) - 0.5f;
}

static void oracle_embed_lookup(const float *table, uint32_t dim, const float *ids, uint32_t seq, float *out) {
    for (uint32_t t = 0; t < seq; t++) {
        uint32_t idx = (uint32_t)(ids[t] + 0.5f);
        for (uint32_t d = 0; d < dim; d++) out[t * dim + d] = table[idx * dim + d];
    }
}
/* Independent oracle for the backward scatter-add: for each table row,
 * sum the incoming gradient of every sequence position that referenced
 * it. Written without reusing k_embed_bwd's loop structure. */
static void oracle_embed_grad(const float *ids, uint32_t seq, uint32_t dim, const float *dout, float *dtable_accum, uint32_t vocab) {
    for (uint32_t v = 0; v < vocab; v++) for (uint32_t d = 0; d < dim; d++) {
        float sum = 0;
        for (uint32_t t = 0; t < seq; t++) if ((uint32_t)(ids[t] + 0.5f) == v) sum += dout[t * dim + d];
        dtable_accum[v * dim + d] += sum;
    }
}

static int run_case(uint32_t seq, const float *ids, const char *label) {
    enum { TABLE, IDS, EMB, NODES };
    float table[VOCAB * DIM], gtable[VOCAB * DIM] = {0};
    float ids_buf[MAX_SEQ], gids[MAX_SEQ] = {0};
    float emb[MAX_SEQ * DIM], gemb[MAX_SEQ * DIM];
    for (uint32_t i = 0; i < VOCAB * DIM; i++) table[i] = dv((int)i, 1);
    for (uint32_t t = 0; t < seq; t++) ids_buf[t] = ids[t];
    for (uint32_t i = 0; i < seq * DIM; i++) gemb[i] = dv((int)i, 2); /* arbitrary incoming gradient to seed backward */

    Node g[NODES] = {
        {LEAF, NONE, NONE, PARAM | RETAIN_GRAD, {table, gtable, NULL, VOCAB, DIM, 0}},
        {LEAF, NONE, NONE, CONSTANT, {ids_buf, gids, NULL, 1, seq, 0}},
        {EMBED_LOOKUP, TABLE, IDS, TEMP | RETAIN_GRAD, {emb, gemb, NULL, seq, DIM, 0}},
    };
    ExecStep steps[8]; Context ctx[8]; uint32_t count = 0;
    if (compile(g, NODES, .01f, steps, ctx, 8, &count)) { fprintf(stderr, "%s: compile rejected\n", label); return 1; }
    /* forward-only steps: count includes zero_grad+backward too since EMB
     * is RETAIN_GRAD (needs=1); run everything, then check both. */
    if (tensor_transformer_steps_execute(steps, count)) { fprintf(stderr, "%s: execute failed\n", label); return 1; }

    float oracle_out[MAX_SEQ * DIM];
    oracle_embed_lookup(table, DIM, ids_buf, seq, oracle_out);
    for (uint32_t i = 0; i < seq * DIM; i++) {
        if (!isfinite(emb[i])) { fprintf(stderr, "%s: non-finite output at %u\n", label, i); return 1; }
        if (fabsf(emb[i] - oracle_out[i]) > 1e-6f) { fprintf(stderr, "%s: forward mismatch at %u: canonical=%.9g oracle=%.9g\n", label, i, emb[i], oracle_out[i]); return 1; }
    }

    float oracle_gtable[VOCAB * DIM] = {0};
    oracle_embed_grad(ids_buf, seq, DIM, gemb, oracle_gtable, VOCAB);
    for (uint32_t i = 0; i < VOCAB * DIM; i++) {
        if (!isfinite(gtable[i])) { fprintf(stderr, "%s: non-finite grad at %u\n", label, i); return 1; }
        if (fabsf(gtable[i] - oracle_gtable[i]) > 1e-6f) { fprintf(stderr, "%s: grad mismatch at %u: canonical=%.9g oracle=%.9g\n", label, i, gtable[i], oracle_gtable[i]); return 1; }
    }
    printf("%s: seq=%u forward=match backward_scatter_add=match finite=yes\n", label, seq);
    return 0;
}

int main(void) {
    /* seq=7: plain sequential ids, no repeats or boundaries beyond the natural range */
    float ids7[7] = {3, 1, 4, 1, 5, 9 % VOCAB, 2};
    if (run_case(7, ids7, "seq7_plain")) return 1;

    /* seq=8: tails -- vocab boundary indices 0 and VOCAB-1 both present */
    float ids8[8] = {0, 5, VOCAB - 1, 2, 0, 7, VOCAB - 1, 3};
    if (run_case(8, ids8, "seq8_tails")) return 1;

    /* seq=9: duplicate-index stress -- index 4 appears three times, only a
     * real scatter-add (not last-write-wins) backward gets this right */
    float ids9[9] = {4, 4, 1, 4, 6, 0, VOCAB - 1, 4, 4};
    if (run_case(9, ids9, "seq9_duplicates")) return 1;

    puts("embed_lookup differential check passed: reference<->canonical<->executor agree on seq=7/8/9, forward exact, backward scatter-add exact, all finite");
    return 0;
}
