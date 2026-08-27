/* Real GunPoint sequence classifier through the canonical semantic compiler.
 * The model is only a Node graph: compile() derives forward, zero-grad,
 * backward, and optimizer actions. The x86_64 assembly executor runs them. */
#define T 3
#define M 4
#define H 2
#define D 2
#define F 6
#define QW (3 * M)
#include "tensor_semantic_compiler.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct { char magic[8]; uint32_t rows, tokens, features, reserved; } Header;
typedef struct { uint32_t test, id; float x[T * M], y; } Record;
static uint32_t rng = 0x47554e31u;
static uint32_t random_u32(void) { rng = rng * 1664525u + 1013904223u; return rng; }
static float initv(int i, int s) { return (float)(((i * 37 + s * 17) % 29) - 14) / 41.0f; }
static int exact(void *p, size_t n, FILE *f) { return fread(p, 1, n, f) == n ? 0 : -1; }

enum {
    N_X, N_WQ, N_WO, N_W1, N_W2, N_HEAD, N_BIAS, N_TARGET,
    N_QKV, N_ATTN, N_MERGED, N_PROJ, N_SUM1, N_LN1, N_Z1, N_ACT,
    N_FF, N_SUM2, N_LN2, N_MEAN, N_LOGIT, N_AFFINE, N_PRED, N_LOSS, N_COUNT
};

typedef struct {
    float x[T*M], wq[M*QW], wo[M*M], w1[M*F], w2[F*M], head[M], bias[1], target[1];
    float qkv[T*QW], attn[H*T*D], merged[T*M], proj[T*M], sum1[T*M], ln1[T*M];
    float z1[T*F], act[T*F], ff[T*M], sum2[T*M], ln2[T*M], mean[M], logit[1], affine[1], pred[1], loss[1];
    float gx[T*M], gwq[M*QW], gwo[M*M], gw1[M*F], gw2[F*M], ghead[M], gbias[1], gtarget[1];
    float gqkv[T*QW], gattn[H*T*D], gmerged[T*M], gproj[T*M], gsum1[T*M], gln1[T*M];
    float gz1[T*F], gact[T*F], gff[T*M], gsum2[T*M], gln2[T*M], gmean[M], glogit[1], gaffine[1], gpred[1], gloss[1];
    float attn_aux[H*T*T], ln1_aux[2*T], ln2_aux[2*T];
    ParameterOptimizationMetadata head_optimization, bias_optimization;
    Node graph[N_COUNT]; ExecStep steps[96]; Context ctx[96]; uint32_t step_count, forward_count;
} Model;

static int model_init(Model *m) {
    memset(m, 0, sizeof *m);
    for (int i=0;i<M*QW;i++) m->wq[i]=initv(i,2)*.4f;
    for (int i=0;i<M*M;i++) m->wo[i]=initv(i,3)*.4f;
    for (int i=0;i<M*F;i++) m->w1[i]=initv(i,4)*.5f;
    for (int i=0;i<F*M;i++) m->w2[i]=initv(i,5)*.5f;
    for (int i=0;i<M;i++) m->head[i]=initv(i,19)*.2f;
    m->head_optimization.lr_multiplier=2.0f;
    m->bias_optimization.lr_multiplier=2.0f;
    Node g[N_COUNT] = {
        [N_X]={LEAF,NONE,NONE,INPUT,{m->x,m->gx,NULL,T,M,0}},
        [N_WQ]={LEAF,NONE,NONE,PARAM,{m->wq,m->gwq,NULL,M,QW,0}},
        [N_WO]={LEAF,NONE,NONE,PARAM,{m->wo,m->gwo,NULL,M,M,0}},
        [N_W1]={LEAF,NONE,NONE,PARAM,{m->w1,m->gw1,NULL,M,F,0}},
        [N_W2]={LEAF,NONE,NONE,PARAM,{m->w2,m->gw2,NULL,F,M,0}},
        [N_HEAD]={LEAF,NONE,NONE,PARAM,{m->head,m->ghead,(float *)&m->head_optimization,M,1,1}},
        [N_BIAS]={LEAF,NONE,NONE,PARAM,{m->bias,m->gbias,(float *)&m->bias_optimization,1,1,1}},
        [N_TARGET]={LEAF,NONE,NONE,CONSTANT,{m->target,m->gtarget,NULL,1,1,0}},
        [N_QKV]={MATMUL,N_X,N_WQ,TEMP,{m->qkv,m->gqkv,NULL,T,QW,0}},
        [N_ATTN]={ATTENTION,N_QKV,NONE,TEMP,{m->attn,m->gattn,m->attn_aux,1,H*T*D,H*T*T}},
        [N_MERGED]={CONTIGUOUS,N_ATTN,NONE,TEMP,{m->merged,m->gmerged,NULL,T,M,0}},
        [N_PROJ]={MATMUL,N_MERGED,N_WO,TEMP,{m->proj,m->gproj,NULL,T,M,0}},
        [N_SUM1]={RESIDUAL,N_X,N_PROJ,TEMP,{m->sum1,m->gsum1,NULL,T,M,0}},
        [N_LN1]={LAYERNORM,N_SUM1,NONE,TEMP,{m->ln1,m->gln1,m->ln1_aux,T,M,2*T}},
        [N_Z1]={MATMUL,N_LN1,N_W1,TEMP,{m->z1,m->gz1,NULL,T,F,0}},
        [N_ACT]={RELU,N_Z1,NONE,TEMP,{m->act,m->gact,NULL,T,F,0}},
        [N_FF]={MATMUL,N_ACT,N_W2,TEMP,{m->ff,m->gff,NULL,T,M,0}},
        [N_SUM2]={RESIDUAL,N_LN1,N_FF,TEMP,{m->sum2,m->gsum2,NULL,T,M,0}},
        [N_LN2]={LAYERNORM,N_SUM2,NONE,TEMP,{m->ln2,m->gln2,m->ln2_aux,T,M,2*T}},
        [N_MEAN]={REDUCE_MEAN_ROWS,N_LN2,NONE,TEMP,{m->mean,m->gmean,NULL,1,M,0}},
        [N_LOGIT]={MATMUL,N_MEAN,N_HEAD,TEMP,{m->logit,m->glogit,NULL,1,1,0}},
        [N_AFFINE]={BIAS_ADD,N_LOGIT,N_BIAS,TEMP,{m->affine,m->gaffine,NULL,1,1,0}},
        [N_PRED]={SIGMOID,N_AFFINE,NONE,TEMP,{m->pred,m->gpred,NULL,1,1,0}},
        [N_LOSS]={BINARY_CROSS_ENTROPY,N_PRED,N_TARGET,TEMP,{m->loss,m->gloss,NULL,1,1,0}},
    };
    memcpy(m->graph,g,sizeof g);
    for (unsigned i=0;i<N_COUNT;i++) if (g[i].op!=LEAF) m->forward_count++;
    return compile(m->graph,N_COUNT,.01f,m->steps,m->ctx,96,&m->step_count);
}

static void load_input(Model *m, const Record *r, int shuffled) {
    int order[T]={0,1,2};
    if (shuffled) { uint32_t z=r->id*747796405u+2891336453u; for(int i=T-1;i;i--){z=z*1664525u+1013904223u;int j=z%(i+1),q=order[i];order[i]=order[j];order[j]=q;} }
    for(int t=0;t<T;t++) for(int j=0;j<M;j++) m->x[t*M+j]=r->x[order[t]*M+j];
    m->target[0]=r->y;
}
static float evaluate(Model *m,const Record *r,const uint32_t *ids,uint32_t n,int shuffled){uint32_t ok=0;for(uint32_t i=0;i<n;i++){load_input(m,&r[ids[i]],shuffled);if(tensor_transformer_steps_execute(m->steps,m->forward_count))return NAN;ok+=(m->pred[0]>=.5f)==(r[ids[i]].y>=.5f);}return(float)ok/n;}

int main(int argc,char **argv){
    if(argc!=3||(strcmp(argv[2],"ordered")&&strcmp(argv[2],"shuffled"))){fprintf(stderr,"usage: %s sequence.bin ordered|shuffled\n",argv[0]);return 2;}
    int shuffled=!strcmp(argv[2],"shuffled"); FILE*f=fopen(argv[1],"rb"); if(!f)return 2;
    Header h; if(exact(&h,sizeof h,f)||memcmp(h.magic,"GUNSEQ1",7)||h.tokens!=T||h.features!=M)return 2;
    Record*r=calloc(h.rows,sizeof*r); if(!r)return 2; for(uint32_t i=0;i<h.rows;i++)if(exact(&r[i].test,4,f)||exact(&r[i].id,4,f)||exact(r[i].x,sizeof r[i].x,f)||exact(&r[i].y,4,f))return 2; fclose(f);
    uint32_t*train=malloc(h.rows*4),*test=malloc(h.rows*4),nt=0,nv=0; if(!train||!test)return 2;
    for(uint32_t i=0;i<h.rows;i++)(r[i].test?test:train)[r[i].test?nv++:nt++]=i;
    Model*m=calloc(1,sizeof* m); if(!m||model_init(m))return 3; float initial=evaluate(m,r,test,nv,shuffled);
    enum{EPOCHS=2500}; for(unsigned epoch=0;epoch<EPOCHS;epoch++){for(uint32_t i=nt-1;i;i--){uint32_t j=random_u32()%(i+1),x=train[i];train[i]=train[j];train[j]=x;}for(uint32_t q=0;q<nt;q++){load_input(m,&r[train[q]],shuffled);if(tensor_transformer_steps_execute(m->steps,m->step_count))return 4;}}
    float final=evaluate(m,r,test,nv,shuffled);
    printf("gunpoint semantic transformer: mode=%s nodes=%u steps=%u forward=%u samples=%u train=%u test=%u epochs=%u schedule=compile-derived executor=x86_64-assembly test_acc=%.3f->%.3f\n",argv[2],(unsigned)N_COUNT,m->step_count,m->forward_count,h.rows,nt,nv,(unsigned)EPOCHS,initial,final);
    free(m);free(r);free(train);free(test);return isfinite(final)?0:1;
}
