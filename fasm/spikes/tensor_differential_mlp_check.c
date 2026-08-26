/* Differential oracle: canonical semantic compiler vs the independent,
 * monolithic MLP implementation retained in tensor_mlp_executor_spike.h.
 * This file is deliberately a consumer only: do not fix compiler failures here. */
#include "tensor_semantic_compiler.h"
#include <math.h>
#include <stdio.h>
#include <string.h>

enum { IN = 2, HID = 4, OUT = 1 };
enum { X, W1, B1, W2, B2, TARGET, MM1, ADD1, ACT1, MM2, ADD2, LOSS, NODE_COUNT };

typedef struct {
    float x[IN], w1[IN*HID], b1[HID], z1[HID], h1[HID], w2[HID], b2, out;
    float dx[IN], dw1[IN*HID], db1[HID], dw2[HID], db2;
} Oracle;

static void init(float *w1, float *b1, float *w2, float *b2) {
    const float iw1[] = {.5f,-.7f,.3f,.8f,-.4f,.6f,.9f,-.2f};
    const float ib1[] = {.1f,.1f,-.1f,0};
    const float iw2[] = {.7f,-.5f,.6f,-.8f};
    memcpy(w1, iw1, sizeof iw1); memcpy(b1, ib1, sizeof ib1);
    memcpy(w2, iw2, sizeof iw2); *b2 = 0;
}

static void oracle_forward(Oracle *o) {
    for (int j=0;j<HID;j++) { float s=o->b1[j]; for(int i=0;i<IN;i++) s+=o->x[i]*o->w1[i*HID+j]; o->z1[j]=s; o->h1[j]=s>0?s:0; }
    float s=o->b2; for(int j=0;j<HID;j++) s+=o->h1[j]*o->w2[j]; o->out=s;
}
static void oracle_backward(Oracle *o, float seed) {
    memset(o->dx,0,sizeof o->dx); memset(o->dw1,0,sizeof o->dw1); memset(o->db1,0,sizeof o->db1); memset(o->dw2,0,sizeof o->dw2); o->db2=seed;
    float dh[HID], dz[HID];
    for(int j=0;j<HID;j++){o->dw2[j]=o->h1[j]*seed;dh[j]=o->w2[j]*seed;dz[j]=o->z1[j]>0?dh[j]:0;o->db1[j]=dz[j];}
    for(int i=0;i<IN;i++)for(int j=0;j<HID;j++){o->dw1[i*HID+j]=o->x[i]*dz[j];o->dx[i]+=o->w1[i*HID+j]*dz[j];}
}
static int closev(const char *name,const float *a,const float *b,int n,float tol) {
    for(int i=0;i<n;i++) if(fabsf(a[i]-b[i])>tol*fmaxf(1,fmaxf(fabsf(a[i]),fabsf(b[i])))) { fprintf(stderr,"DIFF mlp %s[%d]: canonical=%.9g oracle=%.9g\n",name,i,a[i],b[i]); return 1; }
    return 0;
}

int main(void) {
    float x[IN]={0,1},w1[IN*HID],b1[HID],w2[HID],b2,target[1]={1}; init(w1,b1,w2,&b2);
    float mm1[HID],add1[HID],act1[HID],mm2[1],add2[1],loss[1];
    float gx[IN]={0},gw1[IN*HID],gb1[HID],gw2[HID],gb2[1],gt[1]={0},gmm1[HID],gadd1[HID],gact1[HID],gmm2[1],gadd2[1],gloss[1];
    Node g[NODE_COUNT]={{LEAF,NONE,NONE,INPUT|RETAIN_GRAD,{x,gx,NULL,1,IN,0}},{LEAF,NONE,NONE,PARAM,{w1,gw1,NULL,IN,HID,0}},{LEAF,NONE,NONE,PARAM,{b1,gb1,NULL,1,HID,0}},{LEAF,NONE,NONE,PARAM,{w2,gw2,NULL,HID,OUT,0}},{LEAF,NONE,NONE,PARAM,{&b2,gb2,NULL,1,OUT,0}},{LEAF,NONE,NONE,CONSTANT,{target,gt,NULL,1,OUT,0}},{MATMUL,X,W1,TEMP,{mm1,gmm1,NULL,1,HID,0}},{BIAS_ADD,MM1,B1,TEMP,{add1,gadd1,NULL,1,HID,0}},{RELU,ADD1,NONE,TEMP,{act1,gact1,NULL,1,HID,0}},{MATMUL,ACT1,W2,TEMP,{mm2,gmm2,NULL,1,OUT,0}},{BIAS_ADD,MM2,B2,TEMP,{add2,gadd2,NULL,1,OUT,0}},{MSE,ADD2,TARGET,TEMP,{loss,gloss,NULL,1,1,0}}};
    ExecStep steps[32]; Context ctx[32]; uint32_t count=0;
    if(compile(g,NODE_COUNT,.1f,steps,ctx,32,&count)||count!=26){fprintf(stderr,"DIFF mlp compile metadata: steps=%u want=26\n",count);return 2;}
    if(tensor_transformer_steps_execute(steps,22)) return 2;
    Oracle o={0}; memcpy(o.x,x,sizeof x); memcpy(o.w1,w1,sizeof w1); memcpy(o.b1,b1,sizeof b1); memcpy(o.w2,w2,sizeof w2); o.b2=b2;
    oracle_forward(&o); float seed=2*(o.out-target[0]); oracle_backward(&o,seed);
    int mismatches=0;
    mismatches+=closev("forward",add2,&o.out,1,1e-6f); mismatches+=closev("dx",gx,o.dx,IN,2e-6f);
    mismatches+=closev("dw1",gw1,o.dw1,IN*HID,2e-6f); mismatches+=closev("db1",gb1,o.db1,HID,2e-6f);
    mismatches+=closev("dw2",gw2,o.dw2,HID,2e-6f); mismatches+=closev("db2",gb2,&o.db2,1,2e-6f);
    for(int i=0;i<IN*HID;i++)o.w1[i]-=.1f*o.dw1[i]; for(int i=0;i<HID;i++){o.b1[i]-=.1f*o.db1[i];o.w2[i]-=.1f*o.dw2[i];} o.b2-=.1f*o.db2;
    if(tensor_transformer_steps_execute(steps+22,4)) return 2;
    mismatches+=closev("updated_w1",w1,o.w1,IN*HID,2e-6f); mismatches+=closev("updated_b1",b1,o.b1,HID,2e-6f);
    mismatches+=closev("updated_w2",w2,o.w2,HID,2e-6f); mismatches+=closev("updated_b2",&b2,&o.b2,1,2e-6f);
    if(mismatches){fprintf(stderr,"DIFF mlp failed: mismatched_fields=%d minimal_case=x=[0,1] target=1 steps=%u\n",mismatches,count);return 3;}
    printf("differential XOR passed: forward=equivalent gradients=equivalent update=equivalent steps=%u\n",count);
    return 0;
}
