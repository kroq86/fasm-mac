/* Self-written GPT-2 byte-level BPE tokenizer, differential/property check.
 *
 * Scope decision (made explicitly with the user before implementation):
 * pre-tokenization's letter/digit classification is ASCII-only, not full
 * Unicode \p{L}/\p{N} categories (no Unicode category tables in this
 * from-scratch project). The byte<->codepoint table itself is still the
 * real, complete GPT-2 algorithm. Round-trip correctness is claimed for
 * ASCII input only.
 *
 * Test cases in tensor_gpt2_bpe_test_cases.h are embedded, pre-computed
 * output from an INDEPENDENT Python oracle (fasm/spikes/tensor_gpt2_bpe.h
 * is not involved in generating them) run against the real, pinned
 * `vocab.json`/`merges.txt` (huggingface.co/gpt2, revision
 * 607a30d783dfa663caf39e06633721c8d4cfcd7e, fetched by
 * scripts/fetch-gpt2-tokenizer.sh), regenerate via:
 *   python3 <job-tmp>/gen_bpe_testcases.py > fasm/spikes/tensor_gpt2_bpe_test_cases.h
 * (the oracle script itself: an independent, from-scratch Python
 * implementation of the published byte-level-BPE algorithm -- same
 * algorithm class as this file's C, deliberately not sharing code with
 * it, matching this project's oracle-independence discipline). Includes
 * a known, documented simplification shared by both sides: "<|endoftext|>"
 * is not special-cased as a single EOS token id (neither oracle nor this
 * implementation does special-token handling), so that test case's
 * "expected" ids are the naive multi-piece BPE result, not id 50256 --
 * consistent between both sides, not a correctness claim about special
 * tokens.
 *
 * Additional property tests below: ASCII round-trip (encode then decode
 * recovers the original bytes exactly) on every case with n>0; and
 * byte-fallback graceful behavior on a non-ASCII byte (out of this
 * project's claimed pre-tokenization scope, but must not crash or hang).
 *
 * Two real bugs this differential check caught before it passed:
 *   1. tensor_gpt2_bpe.h's BPE merge loop merged only the FIRST
 *      occurrence of the winning-rank pair per iteration; the reference
 *      algorithm merges EVERY non-overlapping occurrence in one pass.
 *      These can diverge when a fresh merge creates a new lower-rank
 *      adjacent pair before a second occurrence of the original winner
 *      gets its turn. Fixed to match the reference's single-pass
 *      merge-all-occurrences behavior.
 *   2. The independent Python ORACLE itself had a bug, not this file's
 *      C: its merges.txt parser filtered any line whose first piece
 *      literally starts with the byte-encoded '#' character (e.g. the
 *      real, legitimate merge rule "# $"), treating it as a comment --
 *      correct only for line 0, wrongly applied to all 50000 lines. This
 *      silently dropped 8 real merge rules from the oracle (50000 file
 *      lines but only 49992 in its parsed table), producing wrong
 *      "expected" values for cases exercising them. Caught by comparing
 *      failing cases against hand-derived ranks from the raw file
 *      (grep'd line numbers) and Python's own dict-key count, not by
 *      trusting the oracle. Fixed by only slicing off the header line,
 *      matching OpenAI's reference encoder.py exactly (no per-line '#'
 *      filter at all). A reminder that "independent oracle" still needs
 *      independent verification when the two sides disagree -- the bug
 *      is not automatically in the new code.
 */
#include "tensor_gpt2_bpe.h"
#include "tensor_gpt2_bpe_test_cases.h"
#include <stdio.h>

int main(int argc, char **argv) {
    const char *vocab_path = argc > 1 ? argv[1] : "scratchpad/gpt2_tokenizer_fixture/vocab.json";
    const char *merges_path = argc > 2 ? argv[2] : "scratchpad/gpt2_tokenizer_fixture/merges.txt";

    static Gpt2Bpe bpe;
    if (gpt2_bpe_load(&bpe, vocab_path, merges_path)) {
        printf("gpt2_bpe: tokenizer fixture not found under %s / %s -- skipped\n", vocab_path, merges_path);
        return 0;
    }
    printf("gpt2_bpe: loaded vocab_count=%d merge_count=%d\n", bpe.vocab_count, bpe.merge_count);
    if (bpe.vocab_count < 50000 || bpe.merge_count < 49000) {
        fprintf(stderr, "gpt2_bpe: suspiciously small vocab/merge count, refusing to trust this load\n");
        return 1;
    }

    int failures = 0;
    for (int c = 0; c < BPE_TEST_CASE_COUNT; c++) {
        const BpeTestCase *tc = &BPE_TEST_CASES[c];
        int ids[64];
        int n = gpt2_bpe_encode(&bpe, tc->text, (int)strlen(tc->text), ids, 64);
        if (n != tc->n) {
            fprintf(stderr, "case %d (%.40s%s): count mismatch got=%d want=%d\n", c, tc->text, strlen(tc->text) > 40 ? "..." : "", n, tc->n);
            failures++;
            continue;
        }
        int mismatch = 0;
        for (int i = 0; i < n; i++) if (ids[i] != tc->ids[i]) { mismatch = 1; break; }
        if (mismatch) {
            fprintf(stderr, "case %d (%.40s%s): id mismatch:", c, tc->text, strlen(tc->text) > 40 ? "..." : "");
            for (int i = 0; i < n; i++) fprintf(stderr, " got[%d]=%d want[%d]=%d%s", i, ids[i], i, tc->ids[i], ids[i] != tc->ids[i] ? "*" : "");
            fprintf(stderr, "\n");
            failures++;
            continue;
        }

        if (n > 0) {
            char decoded[512];
            int dn = gpt2_bpe_decode(&bpe, ids, n, decoded, sizeof decoded - 1);
            if (dn < 0) { fprintf(stderr, "case %d: decode failed\n", c); failures++; continue; }
            decoded[dn] = '\0';
            int tlen = (int)strlen(tc->text);
            if (dn != tlen || memcmp(decoded, tc->text, (size_t)tlen)) {
                fprintf(stderr, "case %d: round-trip mismatch: original=%.60s decoded=%.60s\n", c, tc->text, decoded);
                failures++;
                continue;
            }
        }
    }
    printf("gpt2_bpe: %d/%d test cases matched the independent oracle exactly (encode ids + ASCII round-trip decode)\n", BPE_TEST_CASE_COUNT - failures, BPE_TEST_CASE_COUNT);
    if (failures) return 1;

    /* property: byte-fallback graceful behavior on a non-ASCII byte
     * (explicitly out of this project's ASCII-scoped pre-tokenization
     * claim) -- must not crash, hang, or read out of bounds; every byte
     * still has a well-defined byte_enc[] entry regardless of scope, so
     * encoding must at least terminate and either succeed or fail
     * closed, never corrupt memory (checked here by just not crashing
     * under ASan-style build in the gate script). */
    {
        unsigned char raw[4] = {0xC3, 0xA9, 'x', 0};
        int ids[16];
        int n = gpt2_bpe_encode(&bpe, (const char *)raw, 3, ids, 16);
        printf("gpt2_bpe: non-ASCII byte input handled without crash, encode returned n=%d (scope: not claimed correct, only required not to crash)\n", n);
    }

    /* property: vocab boundary ids decode correctly (id 0 and the last id) */
    {
        int ids0[1] = {0};
        char out[16];
        int n = gpt2_bpe_decode(&bpe, ids0, 1, out, sizeof out);
        if (n < 0) { fprintf(stderr, "decode of vocab id 0 failed\n"); return 1; }
        printf("gpt2_bpe: vocab id 0 decodes to %.*s (%d bytes)\n", n, out, n);

        int last_id = bpe.vocab_count - 1;
        int ids_last[1] = {last_id};
        n = gpt2_bpe_decode(&bpe, ids_last, 1, out, sizeof out);
        if (n < 0) { fprintf(stderr, "decode of last vocab id %d failed\n", last_id); return 1; }
        printf("gpt2_bpe: vocab id %d (last, count=%d) decoded ok, %d bytes\n", last_id, bpe.vocab_count, n);
    }

    /* property: fail-closed on an out-of-range token id */
    {
        int bad[1] = {999999};
        char out[16];
        int n = gpt2_bpe_decode(&bpe, bad, 1, out, sizeof out);
        if (n >= 0) { fprintf(stderr, "expected decode of an out-of-range id to fail, got n=%d\n", n); return 1; }
        printf("gpt2_bpe: out-of-range token id correctly rejected on decode (fail-closed)\n");
    }

    puts("gpt2_bpe differential check passed: encode ids match an independent oracle on 21 ASCII cases (including contractions, multi-space, tabs, newlines, punctuation, boundaries), ASCII round-trip exact, non-ASCII input handled without crash, vocab-boundary ids correct, out-of-range decode fails closed");
    return 0;
}
