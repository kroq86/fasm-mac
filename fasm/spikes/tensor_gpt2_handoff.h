#ifndef TENSOR_GPT2_HANDOFF_H
#define TENSOR_GPT2_HANDOFF_H
/* Bounded inference API, GPT-2 geometry only. Caller owns weights/session.
 * Single-threaded: the existing scalar attention helper has static scratch.
 * Cache rows: [position][head * 64 + dimension], FP32. All layers have the
 * same Tensor.aux_count. Import is allowed only into an empty session and
 * copies source data: source and receiver never share mutable cache buffers.
 * Return 0 on success, -1 on malformed input, capacity or executor failure.
 * This is an integration spike, not a general model-switching ABI. */
#include "tensor_semantic_compiler.h"
#include "tensor_gpt2_forward.h"

#if M != 768 || H != 12 || D != 64 || QW != 2304 || MAXCACHE != 64
#error "handoff requires GPT-2 geometry and capacity=64"
#endif

typedef void (*HandoffMatmul)(const float *, int, int, const float *, const float *, int, float *);
typedef struct {
    Node graph[3]; ExecStep steps[8]; Context contexts[8]; uint32_t count;
    float qkv[QW], qkv_grad[QW], k[MAXCACHE*M], kg[MAXCACHE*M], v[MAXCACHE*M];
    float out[M], grad[M];
} HandoffLayer;
typedef struct {
    const Gpt2Weights *weights; int layers; HandoffMatmul matmul;
    HandoffLayer layer[GPT2_NLAYER];
} HandoffSession;

static int handoff_init(HandoffSession *s, const Gpt2Weights *w, int layers, HandoffMatmul mm) {
    if (!s || !w || !mm || layers < 1 || layers > GPT2_NLAYER) return -1;
    memset(s, 0, sizeof *s); s->weights=w; s->layers=layers; s->matmul=mm;
    for (int i=0;i<layers;i++) {
        HandoffLayer *l=&s->layer[i];
        l->graph[0]=(Node){LEAF,NONE,NONE,INPUT,{l->qkv,l->qkv_grad,NULL,1,QW,0}};
        l->graph[1]=(Node){LEAF,NONE,NONE,INPUT,{l->k,l->kg,l->v,MAXCACHE,M,0}};
        l->graph[2]=(Node){CAUSAL_ATTENTION_CACHED,0,1,TEMP,{l->out,l->grad,NULL,1,M,0}};
        if (compile(l->graph,3,.01f,l->steps,l->contexts,8,&l->count)) return -1;
    }
    return 0;
}

/* Correct prefill: each prefix token is evaluated exactly once. This bounded
 * path uses the same canonical cached-attention op for prefill and decode.
 * It favors reuse/verification, not optimal batched prefill throughput. */
static int handoff_token(HandoffSession *s, int token, float *logits) {
    uint32_t pos=s->layer[0].graph[1].tensor.aux_count;
    if (token<0 || token>=GPT2_VOCAB || pos>=MAXCACHE) return -1;
    for(int i=1;i<s->layers;i++) if(s->layer[i].graph[1].tensor.aux_count!=pos) return -1;
    float hidden[M], ln[M], proj[M], fc[GPT2_F], act[GPT2_F];
    for(int j=0;j<M;j++) hidden[j]=s->weights->wte[token*M+j]+s->weights->wpe[pos*M+j];
    for(int i=0;i<s->layers;i++) {
        HandoffLayer *l=&s->layer[i]; const Gpt2BlockWeights *w=&s->weights->blk[i];
        gpt2_layernorm_raw(hidden,1,M,ln);
        s->matmul(ln,1,M,w->attn_w,w->attn_b,QW,l->qkv);
        if(tensor_transformer_steps_execute(l->steps,l->count)) return -1;
        s->matmul(l->out,1,M,w->projw,w->projb,M,proj);
        for(int j=0;j<M;j++) hidden[j]+=proj[j];
        gpt2_layernorm_raw(hidden,1,M,ln);
        s->matmul(ln,1,M,w->fc_w,w->fc_b,GPT2_F,fc);
        gpt2_gelu(fc,GPT2_F,act);
        s->matmul(act,1,GPT2_F,w->fcproj_w,w->fcproj_b,M,proj);
        for(int j=0;j<M;j++) hidden[j]+=proj[j];
    }
    if(logits) {
        gpt2_layernorm_raw(hidden,1,M,ln);
        s->matmul(ln,1,M,s->weights->lnf_w_folded,s->weights->lnf_b_folded,GPT2_VOCAB,logits);
    }
    return 0;
}

typedef struct { float a[768*8], b[8*768]; } HandoffMap;
typedef struct { HandoffMap key[6], value[6]; } HandoffBridge;

static int handoff_load_bridge(HandoffBridge *b, const char *path) {
    SafetensorsFile f; if(st_open(&f,path)) return -1;
    uint64_t a[2]={768,8}, z[2]={8,768}; int bad=0; char name[40];
    for(int l=0;l<6;l++) for(int stream=0;stream<2;stream++) {
        HandoffMap *m=stream?&b->value[l]:&b->key[l];
        snprintf(name,sizeof name,"A_%c.%d",stream?'v':'k',l); bad|=st_read_f32(&f,name,a,2,m->a);
        snprintf(name,sizeof name,"B_%c.%d",stream?'v':'k',l); bad|=st_read_f32(&f,name,z,2,m->b);
        for(int j=0;j<768*8;j++) if(!isfinite(m->a[j]) || !isfinite(m->b[j])) bad=1;
    }
    st_close(&f); return bad?-1:0;
}

static int handoff_import(HandoffSession *dst, const HandoffSession *src, const HandoffBridge *bridge) {
    if(dst==src || dst->layers!=6 || src->layers!=12) return -1;
    uint32_t n=src->layer[0].graph[1].tensor.aux_count;
    if(!n || n>=MAXCACHE) return -1;
    for(int l=0;l<12;l++) if(src->layer[l].graph[1].tensor.aux_count!=n) return -1;
    for(int l=0;l<6;l++) if(dst->layer[l].graph[1].tensor.aux_count) return -1;
    /* Validate the complete source before mutating receiver state. */
    for(int l=0;l<6;l++) for(uint32_t j=0;j<n*M;j++)
        if(!isfinite(src->layer[2*l+1].k[j]) || !isfinite(src->layer[2*l+1].v[j])) return -1;
    float tmp[MAXCACHE*8], correction[MAXCACHE*M], zero[M]={0};
    for(int l=0;l<6;l++) for(int stream=0;stream<2;stream++) {
        const float *x=stream?src->layer[2*l+1].v:src->layer[2*l+1].k;
        float *y=stream?dst->layer[l].v:dst->layer[l].k;
        const HandoffMap *m=stream?&bridge->value[l]:&bridge->key[l];
        dst->matmul(x,(int)n,M,m->a,zero,8,tmp);
        dst->matmul(tmp,(int)n,8,m->b,zero,M,correction);
        for(uint32_t j=0;j<n*M;j++) y[j]=x[j]+correction[j];
    }
    for(int l=0;l<6;l++) for(uint32_t j=0;j<n*M;j++)
        if(!isfinite(dst->layer[l].k[j]) || !isfinite(dst->layer[l].v[j])) {
            /* Session is empty on failure; discard candidate buffers. */
            for(int k=0;k<6;k++) { memset(dst->layer[k].k,0,sizeof dst->layer[k].k); memset(dst->layer[k].v,0,sizeof dst->layer[k].v); }
            return -1;
        }
    for(int l=0;l<6;l++) dst->layer[l].graph[1].tensor.aux_count=n;
    return 0;
}
#endif
