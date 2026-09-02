/* Decoder-runtime Etap 2, operation 6/10: top-k/temperature sampling.
 *
 * This is inference-time control logic, not a differentiable canonical
 * graph op (sampling has no gradient), so there is no canonical-compiler
 * kernel pair here and no executor involvement -- the "reference vs
 * canonical vs executor" three-way split from operations 1-5 does not
 * apply. Instead this checks the properties an inference sampler must
 * actually have:
 *   1. temperature<=0 (greedy) always equals an independent argmax oracle,
 *      including a tie (first-index-wins) case and a negative-logits case.
 *   2. every top-k draw's returned token is a member of the true top-k
 *      set, verified against an independently computed top-k oracle, over
 *      many draws and several k values.
 *   3. same seed => same full draw sequence (determinism/replay).
 *   4. the categorical draw itself is verified against an independent
 *      double-precision softmax+cumulative-sum oracle on specific
 *      engineered logits at specific known uniform-draw values (not just
 *      "statistically plausible" -- exact index-boundary correctness).
 *   5. large-sample empirical frequency matches analytic probability for
 *      a small controlled distribution (statistical correctness, not
 *      just "returns a valid index").
 */
#include "tensor_sampling.h"
#include <stdio.h>
#include <stdlib.h>

static int oracle_argmax(const float *logits, int vocab) {
    int best = 0;
    for (int i = 1; i < vocab; i++) if (logits[i] > logits[best]) best = i;
    return best;
}

/* independent top-k membership oracle: counts how many entries strictly
 * exceed logits[idx], plus tie-break-by-lower-index bookkeeping, to
 * decide if idx is among the k highest under the same "first index wins
 * ties" convention as the implementation under test. */
static int oracle_in_top_k(const float *logits, int vocab, int k, int idx) {
    int rank = 0;
    for (int i = 0; i < vocab; i++) {
        if (logits[i] > logits[idx]) rank++;
        else if (logits[i] == logits[idx] && i < idx) rank++;
    }
    return rank < k;
}

int main(void) {
    /* 1. greedy correctness, including an exact tie and negative logits */
    {
        float a[7] = {0.1f, -3.0f, 5.0f, 5.0f, 2.0f, -9.0f, 4.999f}; /* tie at indices 2,3 */
        SampleRng rng = {12345};
        int got = sample_top_k_temperature(a, 7, 0, 0.0f, &rng);
        int want = oracle_argmax(a, 7);
        if (got != want) { fprintf(stderr, "greedy tie case: got=%d want=%d\n", got, want); return 1; }
        if (got != 2) { fprintf(stderr, "greedy tie case: expected first-index-wins tie break at 2, got %d\n", got); return 1; }

        float b[5] = {-1.0f, -0.5f, -8.0f, -0.2f, -4.0f};
        got = sample_argmax(b, 5); want = oracle_argmax(b, 5);
        if (got != want) { fprintf(stderr, "greedy negative case: got=%d want=%d\n", got, want); return 1; }
        printf("greedy: matches independent argmax oracle including tie-break and negative logits\n");
    }

    /* 2. top-k membership held over many draws, several (vocab,k) pairs */
    {
        int vocab = 11;
        float logits[11];
        for (int i = 0; i < vocab; i++) logits[i] = sinf((float)i * 3.7f) * 5.0f + (float)i * 0.3f;
        SampleRng rng = {999};
        int ks[] = {1, 3, 5, vocab};
        for (int kk = 0; kk < 4; kk++) {
            int k = ks[kk];
            for (int draw = 0; draw < 500; draw++) {
                int idx = sample_top_k_temperature(logits, vocab, k, 0.8f, &rng);
                if (!oracle_in_top_k(logits, vocab, k, idx)) {
                    fprintf(stderr, "top-k violation: k=%d idx=%d not in true top-%d\n", k, idx, k);
                    return 1;
                }
            }
        }
        printf("top-k: 2000 draws across k in {1,3,5,vocab}, every draw verified inside independent top-k oracle set\n");
    }

    /* 3. determinism / replay: same seed -> identical draw sequence */
    {
        int vocab = 9;
        float logits[9]; for (int i = 0; i < vocab; i++) logits[i] = cosf((float)i * 1.9f) * 3.0f;
        SampleRng rng1 = {424242}, rng2 = {424242};
        int seq1[50], seq2[50];
        for (int i = 0; i < 50; i++) seq1[i] = sample_top_k_temperature(logits, vocab, 5, 1.0f, &rng1);
        for (int i = 0; i < 50; i++) seq2[i] = sample_top_k_temperature(logits, vocab, 5, 1.0f, &rng2);
        for (int i = 0; i < 50; i++) if (seq1[i] != seq2[i]) { fprintf(stderr, "determinism violated at draw %d: %d vs %d\n", i, seq1[i], seq2[i]); return 1; }
        printf("determinism: same seed reproduces an identical 50-draw sequence\n");
    }

    /* 4. exact categorical-boundary correctness against an independent
     * double-precision oracle, using a hand-controlled uniform draw */
    {
        float logits[4] = {2.0f, 1.0f, 0.0f, -1.0f};
        int vocab = 4, k = 4;
        double mx = -1e300;
        for (int i = 0; i < vocab; i++) if (logits[i] > mx) mx = logits[i];
        double ex[4], total = 0;
        for (int i = 0; i < vocab; i++) { ex[i] = exp((double)logits[i] - mx); total += ex[i]; }
        double p[4]; for (int i = 0; i < vocab; i++) p[i] = ex[i] / total;
        double cum[4]; cum[0] = p[0]; for (int i = 1; i < vocab; i++) cum[i] = cum[i - 1] + p[i];

        /* engineer specific rng states whose first uniform draw we probe
         * for, by brute-force scanning small seeds for draws landing
         * cleanly inside each of the 4 probability buckets */
        for (int bucket = 0; bucket < vocab; bucket++) {
            double lo = bucket == 0 ? 0.0 : cum[bucket - 1];
            double hi = cum[bucket];
            double mid = (lo + hi) / 2.0;
            int found = 0;
            for (uint32_t seed = 1; seed < 200000 && !found; seed++) {
                SampleRng probe = {seed};
                float u = sample_rng_uniform01(&probe);
                if ((double)u > lo + 1e-4 && (double)u < hi - 1e-4) {
                    SampleRng rng = {seed};
                    int got = sample_top_k_temperature(logits, vocab, k, 1.0f, &rng);
                    if (got != bucket) { fprintf(stderr, "categorical boundary: seed=%u u=%.6f expected bucket=%d got=%d (cum=[%.4f,%.4f,%.4f,%.4f])\n", seed, u, bucket, got, cum[0], cum[1], cum[2], cum[3]); return 1; }
                    found = 1;
                    (void)mid;
                }
            }
            if (!found) { fprintf(stderr, "categorical boundary: could not find a seed landing in bucket %d (this indicates a search issue, not a sampler bug)\n", bucket); return 1; }
        }
        printf("categorical draw: exact bucket boundaries verified against independent double-precision softmax+CDF oracle for all 4 tokens\n");
    }

    /* 5. large-sample empirical frequency vs analytic probability */
    {
        float logits[3] = {1.0f, 0.0f, -1.0f}; /* softmax(temp=1) approx [0.665, 0.245, 0.090] */
        double mx = 1.0, ex[3], total = 0;
        for (int i = 0; i < 3; i++) { ex[i] = exp((double)logits[i] - mx); total += ex[i]; }
        double p[3]; for (int i = 0; i < 3; i++) p[i] = ex[i] / total;

        SampleRng rng = {777};
        long counts[3] = {0, 0, 0};
        int N = 200000;
        for (int i = 0; i < N; i++) counts[sample_top_k_temperature(logits, 3, 3, 1.0f, &rng)]++;
        for (int i = 0; i < 3; i++) {
            double freq = (double)counts[i] / N;
            if (fabs(freq - p[i]) > 0.01) { fprintf(stderr, "statistical check failed: token %d analytic=%.4f empirical=%.4f (N=%d)\n", i, p[i], freq, N); return 1; }
        }
        printf("statistical: N=%d draws, empirical frequencies match analytic softmax probabilities within 1%%\n", N);
    }

    puts("sampling differential/property check passed: greedy matches argmax oracle, top-k membership held over 2000 draws, deterministic replay confirmed, exact categorical boundaries match an independent oracle, empirical frequencies match analytic probabilities");
    return 0;
}
