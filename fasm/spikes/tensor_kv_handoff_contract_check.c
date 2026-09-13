#include "tensor_gpt2_handoff.h"
#include <assert.h>
#include <float.h>

static Gpt2Weights weights;
static HandoffSession a,b;
static HandoffBridge identity;

int main(void) {
    assert(handoff_init(&a,&weights,0,gpt2_matmul_bias)<0);
    assert(handoff_init(&a,&weights,13,gpt2_matmul_bias)<0);
    assert(!handoff_init(&a,&weights,12,gpt2_matmul_bias));
    assert(!handoff_init(&b,&weights,6,gpt2_matmul_bias));
    assert(handoff_import(&b,&a,&identity)<0); /* empty source */
    assert(handoff_token(&a,-1,NULL)<0);
    assert(handoff_token(&a,GPT2_VOCAB,NULL)<0);
    for(int l=0;l<12;l++) {
        a.layer[l].graph[1].tensor.aux_count=1;
        for(int j=0;j<M;j++) { a.layer[l].k[j]=1; a.layer[l].v[j]=2; }
    }
    assert(handoff_import(&a,&a,&identity)<0);
    a.layer[1].k[0]=NAN;
    assert(handoff_import(&b,&a,&identity)<0);
    assert(b.layer[0].graph[1].tensor.aux_count==0);
    a.layer[1].k[0]=1;
    a.layer[4].graph[1].tensor.aux_count=2;
    assert(handoff_import(&b,&a,&identity)<0);
    a.layer[4].graph[1].tensor.aux_count=1;
    assert(!handoff_import(&b,&a,&identity));
    for(int l=0;l<6;l++) {
        assert(b.layer[l].graph[1].tensor.aux_count==1);
        for(int j=0;j<M;j++) assert(b.layer[l].k[j]==1 && b.layer[l].v[j]==2);
    }
    b.layer[0].k[0]=17;
    assert(a.layer[1].k[0]==1); /* import owns a copy */
    assert(handoff_import(&b,&a,&identity)<0);
    assert(b.layer[0].k[0]==17); /* rejection does not overwrite */
    assert(!handoff_init(&b,&weights,6,gpt2_matmul_bias));
    for(int l=0;l<12;l++) a.layer[l].graph[1].tensor.aux_count=MAXCACHE;
    assert(handoff_import(&b,&a,&identity)<0);
    assert(handoff_token(&a,0,NULL)<0);
    for(int l=0;l<12;l++) a.layer[l].graph[1].tensor.aux_count=1;
    identity.key[0].a[0]=FLT_MAX; identity.key[0].b[0]=FLT_MAX;
    assert(handoff_import(&b,&a,&identity)<0); /* overflow, no committed position */
    for(int l=0;l<6;l++) assert(b.layer[l].graph[1].tensor.aux_count==0 && b.layer[l].k[0]==0);
    puts("handoff contract passed: geometry, token bounds, empty/full/mismatched cache, finite checks, ownership, occupied import, overflow rollback");
    return 0;
}
