/* Frozen contract tests for decoder top-k/temperature sampling.
 * The implementation returns -1 for invalid input and never touches logits or
 * RNG state in that case.  A zero RNG seed must be normalized to a usable
 * deterministic stream.  k<=0 or k>=vocab means the complete vocabulary.
 */
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "tensor_sampling.h"

static int failures;
#define CHECK(name, cond) do { int ok_ = !!(cond); printf("sampling_contract %-34s %s\n", name, ok_ ? "PASS" : "FAIL"); failures += !ok_; } while (0)

static void test_zero_seed(void) {
    SampleRng r = {0};
    uint32_t a = sample_rng_next(&r), b = sample_rng_next(&r);
    CHECK("zero_seed_not_absorbing", a != 0 && b != 0 && a != b);
}

static void test_full_vocab(void) {
    enum { V = 300, TRIALS = 4096 };
    float logits[V]; for (int i=0;i<V;i++) logits[i]=0.0f;
    int saw_tail_equal=0, saw_tail_over=0; SampleRng a={UINT32_C(0x12345678)},b={UINT32_C(0x87654321)};
    for (int draw=0;draw<TRIALS;draw++) {
        int x=sample_top_k_temperature(logits,V,V,1.0f,&a);
        int y=sample_top_k_temperature(logits,V,V+17,1.0f,&b);
        saw_tail_equal |= x>=256; saw_tail_over |= y>=256;
    }
    CHECK("k_equal_vocab_uses_full_vocab", saw_tail_equal);
    CHECK("k_over_vocab_uses_full_vocab", saw_tail_over);
}

static void test_topk_capacity(void) {
    enum { V=400 }; float logits[V]; for(int i=0;i<V;i++)logits[i]=(float)i;
    SampleRng r={11},before=r;
    CHECK("topk_over_capacity_rejected",sample_top_k_temperature(logits,V,257,1,&r)==-1&&r.state==before.state);
}

static int test_invalid_case(const char *which) {
    float good[4]={0,1,2,3}, nanv[4]={0,NAN,2,3}, pinf[4]={0,INFINITY,2,3};
    SampleRng r={7}, before=r;
    int ok=0;
    if(!strcmp(which,"zero-vocab"))ok=sample_top_k_temperature(good,0,2,1,&r)==-1&&r.state==before.state;
    else if(!strcmp(which,"negative-vocab"))ok=sample_top_k_temperature(good,-1,2,1,&r)==-1&&r.state==before.state;
    else if(!strcmp(which,"nan-logit"))ok=sample_top_k_temperature(nanv,4,2,1,&r)==-1&&r.state==before.state;
    else if(!strcmp(which,"inf-logit"))ok=sample_top_k_temperature(pinf,4,2,1,&r)==-1&&r.state==before.state;
    else if(!strcmp(which,"nan-temperature"))ok=sample_top_k_temperature(good,4,2,NAN,&r)==-1&&r.state==before.state;
    else if(!strcmp(which,"inf-temperature"))ok=sample_top_k_temperature(good,4,2,INFINITY,&r)==-1&&r.state==before.state;
    else if(!strcmp(which,"null-logits"))ok=sample_top_k_temperature(NULL,4,2,1,&r)==-1&&r.state==before.state;
    else if(!strcmp(which,"null-rng"))ok=sample_top_k_temperature(good,4,2,1,NULL)==-1;
    else return 2;
    printf("sampling_contract invalid_%-25s %s\n",which,ok?"PASS":"FAIL"); return ok?0:1;
}

static void test_replay(void) {
    float logits[9]={-.2f,.1f,1.2f,.7f,-.3f,2.0f,.4f,.8f,-.9f};
    SampleRng a={UINT32_C(0x12345678)}, b=a; int xa[64],xb[64];
    for(int i=0;i<64;i++) xa[i]=sample_top_k_temperature(logits,9,5,.7f,&a);
    for(int i=0;i<64;i++) xb[i]=sample_top_k_temperature(logits,9,5,.7f,&b);
    CHECK("deterministic_token_replay", !memcmp(xa,xb,sizeof xa));
    CHECK("deterministic_rng_state_replay", a.state==b.state);
}

static int hostile_large_vocab(void) {
    enum { V = 65537 }; float *x=calloc(V,sizeof *x); if(!x)return 2;
    SampleRng r={9}; int got=sample_top_k_temperature(x,V,8,1,&r); free(x);
    CHECK("vocab_over_limit_rejected", got==-1); return failures?1:0;
}

int main(int argc,char **argv) {
    setvbuf(stdout,NULL,_IONBF,0);
    if(argc==2&&!strcmp(argv[1],"--hostile-large-vocab"))return hostile_large_vocab();
    if(argc==3&&!strcmp(argv[1],"--invalid"))return test_invalid_case(argv[2]);
    test_zero_seed(); test_full_vocab(); test_topk_capacity(); test_replay();
    printf("sampling_contract verdict=%s failures=%d\n",failures?"FAIL":"PASS",failures);
    return failures?1:0;
}
