/* Decoder-runtime Etap 2, operation 2/10: positional embeddings. No new
 * kernel -- GPT-2-style positional embedding is EMBED_LOOKUP (against a
 * SECOND, position-indexed table `wpe`) composed with RESIDUAL (elementwise
 * add to the token embedding), reusing both already-tested ops rather than
 * adding a model-specific one. What's new and actually needs its own
 * differential test here is the COMPOSITION: two independent PARAM tables
 * (wte, wpe) joined at a RESIDUAL branch point, and whether backward
 * correctly threads gradient to BOTH through that diamond, not just one.
 *
 * Tested per the predeclared criterion: shapes seq in {7,8,9}, finite
 * values, tails (seq=9 reaches the last row of a 9-row position table,
 * and both vocab boundary token ids and repeated token ids are exercised
 * simultaneously with the position sum, not just in isolation as in the
 * embed_lookup-only check).
 */
#include "tensor_semantic_compiler.h"
#include <stdio.h>

enum { VOCAB = 11, MAX_POS = 9, DIM = 6, MAX_SEQ = 9 };

static float dv(int i, int salt) {
    float a = sinf((float)(i * 12.9898f + salt * 78.233f)) * 43758.5453f;
    return a - floorf(a) - 0.5f;
}

static void oracle_pos_embed(const float *wte, const float *token_ids, const float *wpe, uint32_t seq, uint32_t dim, float *out) {
    for (uint32_t t = 0; t < seq; t++) {
        uint32_t tid = (uint32_t)(token_ids[t] + 0.5f);
        for (uint32_t d = 0; d < dim; d++) out[t * dim + d] = wte[tid * dim + d] + wpe[t * dim + d]; /* position id is literally t */
    }
}
static void oracle_grad(const float *token_ids, uint32_t seq, uint32_t dim, const float *dout, float *dwte, uint32_t vocab, float *dwpe) {
    for (uint32_t v = 0; v < vocab; v++) for (uint32_t d = 0; d < dim; d++) {
        float sum = 0;
        for (uint32_t t = 0; t < seq; t++) if ((uint32_t)(token_ids[t] + 0.5f) == v) sum += dout[t * dim + d];
        dwte[v * dim + d] += sum;
    }
    for (uint32_t t = 0; t < seq; t++) for (uint32_t d = 0; d < dim; d++) dwpe[t * dim + d] += dout[t * dim + d]; /* each position row used at most once */
}

static int run_case(uint32_t seq, const float *token_ids, const char *label) {
    enum { WTE, TOK_IDS, WPE, POS_IDS, TOK_EMB, POS_EMB, HID, NODES };
    float wte[VOCAB * DIM], gwte[VOCAB * DIM] = {0};
    float wpe[MAX_POS * DIM], gwpe[MAX_POS * DIM] = {0};
    float tok_ids_buf[MAX_SEQ], gtok[MAX_SEQ] = {0};
    float pos_ids_buf[MAX_SEQ], gpos[MAX_SEQ] = {0};
    float tok_emb[MAX_SEQ * DIM], gtok_emb[MAX_SEQ * DIM] = {0};
    float pos_emb[MAX_SEQ * DIM], gpos_emb[MAX_SEQ * DIM] = {0};
    float h[MAX_SEQ * DIM], gh[MAX_SEQ * DIM];
    for (uint32_t i = 0; i < VOCAB * DIM; i++) wte[i] = dv((int)i, 11);
    for (uint32_t i = 0; i < MAX_POS * DIM; i++) wpe[i] = dv((int)i, 22);
    for (uint32_t t = 0; t < seq; t++) { tok_ids_buf[t] = token_ids[t]; pos_ids_buf[t] = (float)t; }
    for (uint32_t i = 0; i < seq * DIM; i++) gh[i] = dv((int)i, 33); /* seeded incoming gradient at the sum */

    Node g[NODES] = {
        {LEAF, NONE, NONE, PARAM | RETAIN_GRAD, {wte, gwte, NULL, VOCAB, DIM, 0}},
        {LEAF, NONE, NONE, CONSTANT, {tok_ids_buf, gtok, NULL, 1, seq, 0}},
        {LEAF, NONE, NONE, PARAM | RETAIN_GRAD, {wpe, gwpe, NULL, MAX_POS, DIM, 0}},
        {LEAF, NONE, NONE, CONSTANT, {pos_ids_buf, gpos, NULL, 1, seq, 0}},
        {EMBED_LOOKUP, WTE, TOK_IDS, TEMP, {tok_emb, gtok_emb, NULL, seq, DIM, 0}},
        {EMBED_LOOKUP, WPE, POS_IDS, TEMP, {pos_emb, gpos_emb, NULL, seq, DIM, 0}},
        {RESIDUAL, TOK_EMB, POS_EMB, TEMP | RETAIN_GRAD, {h, gh, NULL, seq, DIM, 0}},
    };
    ExecStep steps[16]; Context ctx[16]; uint32_t count = 0;
    if (compile(g, NODES, .01f, steps, ctx, 16, &count)) { fprintf(stderr, "%s: compile rejected\n", label); return 1; }
    if (tensor_transformer_steps_execute(steps, count)) { fprintf(stderr, "%s: execute failed\n", label); return 1; }

    float oracle_h[MAX_SEQ * DIM];
    oracle_pos_embed(wte, tok_ids_buf, wpe, seq, DIM, oracle_h);
    for (uint32_t i = 0; i < seq * DIM; i++) {
        if (!isfinite(h[i])) { fprintf(stderr, "%s: non-finite h at %u\n", label, i); return 1; }
        if (fabsf(h[i] - oracle_h[i]) > 1e-6f) { fprintf(stderr, "%s: forward mismatch at %u: canonical=%.9g oracle=%.9g\n", label, i, h[i], oracle_h[i]); return 1; }
    }

    float oracle_dwte[VOCAB * DIM] = {0}, oracle_dwpe[MAX_POS * DIM] = {0};
    oracle_grad(tok_ids_buf, seq, DIM, gh, oracle_dwte, VOCAB, oracle_dwpe);
    for (uint32_t i = 0; i < VOCAB * DIM; i++) {
        if (!isfinite(gwte[i])) { fprintf(stderr, "%s: non-finite dwte at %u\n", label, i); return 1; }
        if (fabsf(gwte[i] - oracle_dwte[i]) > 1e-6f) { fprintf(stderr, "%s: dwte mismatch at %u: canonical=%.9g oracle=%.9g\n", label, i, gwte[i], oracle_dwte[i]); return 1; }
    }
    for (uint32_t i = 0; i < seq * DIM; i++) {
        if (!isfinite(gwpe[i])) { fprintf(stderr, "%s: non-finite dwpe at %u\n", label, i); return 1; }
        if (fabsf(gwpe[i] - oracle_dwpe[i]) > 1e-6f) { fprintf(stderr, "%s: dwpe mismatch at %u: canonical=%.9g oracle=%.9g\n", label, i, gwpe[i], oracle_dwpe[i]); return 1; }
    }
    printf("%s: seq=%u forward=match dwte_through_diamond=match dwpe_through_diamond=match finite=yes\n", label, seq);
    return 0;
}

int main(void) {
    float ids7[7] = {3, 1, 4, 1, 5, 2, 6};
    if (run_case(7, ids7, "seq7_plain")) return 1;

    /* tails: token id boundaries (0 and VOCAB-1), position table's own
     * last row exercised at t=7 within an 8-length sequence */
    float ids8[8] = {0, 5, VOCAB - 1, 2, 0, 7, VOCAB - 1, 3};
    if (run_case(8, ids8, "seq8_tails")) return 1;

    /* seq=9 == MAX_POS: exercises wpe's own last row (position 8), plus
     * repeated token ids interacting with the position sum */
    float ids9[9] = {4, 4, 1, 4, 6, 0, VOCAB - 1, 4, 4};
    if (run_case(9, ids9, "seq9_full_position_range")) return 1;

    puts("positional_embed differential check passed: reference<->canonical<->executor agree on seq=7/8/9, forward exact, dual-table diamond backward exact, all finite");
    return 0;
}
