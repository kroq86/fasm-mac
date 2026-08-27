/* Canonical compiler vs retained monolithic Transformer oracle.
 *
 * T/M/H/D/F/QW are deliberately NOT renamed before including the oracle
 * header: tensor_transformer_reference_spike.h now defines them itself
 * via #ifndef-guarded defaults (T=3,M=4,H=2,D=2,F=6,QW=3*M) — the same
 * values this file sets explicitly for the canonical side below. Renaming
 * them to OT/OM/OH/OD/OF/OQW (as this file used to) broke that: the
 * header's own #ifndef T sees T already defined (as the literal token
 * "OT", not a number) and skips setting a value at all, so every use of T
 * inside the header expands to the permanently-undefined identifier OT.
 * No rename is needed any more — both headers agree on the same names and
 * the same intended values, so a plain #undef between the two #include
 * blocks (already present below) is sufficient to reuse them safely. */
#define Block OracleBlock
#define initv oracle_initv
#define mm oracle_mm
#define mmback oracle_mmback
#define ln oracle_ln
#define lnback oracle_lnback
#define forward oracle_forward
#define backward oracle_backward
#define loss oracle_loss
#define check oracle_check
#define TRANSFORMER_REFERENCE_NO_MAIN
#include "tensor_transformer_reference_spike.h"
#undef T
#undef M
#undef H
#undef D
#undef F
#undef QW
#undef Block
#undef initv
#undef mm
#undef mmback
#undef ln
#undef lnback
#undef forward
#undef backward
#undef loss
#undef check

#define T 3
#define M 4
#define H 2
#define D 2
#define F 6
#define QW (3*M)
#include "tensor_semantic_compiler.h"
#include <stdio.h>

enum { X,WQ,WO,W1,W2,TARGET,QKV,ATTN,MERGED,PROJ,SUM1,LN1,Z1,ACT,FF,SUM2,LN2,LOSS,NODES };
static int closev(const char *name,const float *a,const float *b,int n,float tol){for(int i=0;i<n;i++)if(fabsf(a[i]-b[i])>tol*fmaxf(1,fmaxf(fabsf(a[i]),fabsf(b[i])))){fprintf(stderr,"DIFF transformer %s[%d]: canonical=%.9g oracle=%.9g\n",name,i,a[i],b[i]);return 1;}return 0;}

int main(void){
 float x[T*M],wq[M*QW],wo[M*M],w1[M*F],w2[F*M],target[T*M];
 float qkv[T*QW],attn[H*T*D],merged[T*M],proj[T*M],sum1[T*M],ln1[T*M],z1[T*F],act[T*F],ff[T*M],sum2[T*M],ln2[T*M],lossv[1];
 float gx[T*M]={0},gwq[M*QW],gwo[M*M],gw1[M*F],gw2[F*M],gt[T*M]={0},gqkv[T*QW],gattn[H*T*D],gmerged[T*M],gproj[T*M],gsum1[T*M],gln1[T*M],gz1[T*F],gact[T*F],gff[T*M],gsum2[T*M],gln2[T*M],gloss[1];
 float aaux[H*T*T],l1aux[2*T],l2aux[2*T]; OracleBlock o={0};
 for(int i=0;i<T*M;i++)x[i]=o.x[i]=oracle_initv(i,1);
 for(int i=0;i<M*QW;i++)wq[i]=o.wq[i]=oracle_initv(i,2)*.4f;
 for(int i=0;i<M*M;i++)wo[i]=o.wo[i]=oracle_initv(i,3)*.4f;
 for(int i=0;i<M*F;i++)w1[i]=o.w1[i]=oracle_initv(i,4)*.5f;
 for(int i=0;i<F*M;i++)w2[i]=o.w2[i]=oracle_initv(i,5)*.5f;
 for(int i=0;i<T;i++){float m=0,v=0;for(int j=0;j<M;j++){target[i*M+j]=sinf((float)((i+1)*(j+2))*.7f)+cosf((float)(i-j)*.4f);m+=target[i*M+j];}m/=M;for(int j=0;j<M;j++){float q=target[i*M+j]-m;v+=q*q;}float inv=1/sqrtf(v/M+1e-5f);for(int j=0;j<M;j++)target[i*M+j]=(target[i*M+j]-m)*inv;}
 Node g[NODES]={{LEAF,NONE,NONE,INPUT|RETAIN_GRAD,{x,gx,NULL,T,M,0}},{LEAF,NONE,NONE,PARAM,{wq,gwq,NULL,M,QW,0}},{LEAF,NONE,NONE,PARAM,{wo,gwo,NULL,M,M,0}},{LEAF,NONE,NONE,PARAM,{w1,gw1,NULL,M,F,0}},{LEAF,NONE,NONE,PARAM,{w2,gw2,NULL,F,M,0}},{LEAF,NONE,NONE,CONSTANT,{target,gt,NULL,T,M,0}},{MATMUL,X,WQ,TEMP,{qkv,gqkv,NULL,T,QW,0}},{ATTENTION,QKV,NONE,TEMP,{attn,gattn,aaux,1,H*T*D,H*T*T}},{CONTIGUOUS,ATTN,NONE,TEMP,{merged,gmerged,NULL,T,M,0}},{MATMUL,MERGED,WO,TEMP,{proj,gproj,NULL,T,M,0}},{RESIDUAL,X,PROJ,TEMP,{sum1,gsum1,NULL,T,M,0}},{LAYERNORM,SUM1,NONE,TEMP,{ln1,gln1,l1aux,T,M,2*T}},{MATMUL,LN1,W1,TEMP,{z1,gz1,NULL,T,F,0}},{RELU,Z1,NONE,TEMP,{act,gact,NULL,T,F,0}},{MATMUL,ACT,W2,TEMP,{ff,gff,NULL,T,M,0}},{RESIDUAL,LN1,FF,TEMP,{sum2,gsum2,NULL,T,M,0}},{LAYERNORM,SUM2,NONE,TEMP,{ln2,gln2,l2aux,T,M,2*T}},{MSE,LN2,TARGET,TEMP,{lossv,gloss,NULL,1,1,0}}};
 ExecStep steps[64];Context ctx[64];uint32_t count=0;if(compile(g,NODES,.02f,steps,ctx,64,&count)||count!=44){fprintf(stderr,"DIFF transformer compile metadata: steps=%u want=44\n",count);return 2;}
 if(tensor_transformer_steps_execute(steps,40))return 2;
 oracle_forward(&o);float seed[T*M];float oloss=0;for(int i=0;i<T*M;i++){float e=o.out[i]-target[i];oloss+=e*e;seed[i]=2*e/(T*M);}oloss/=T*M;oracle_backward(&o,seed);
 int bad=0;bad+=closev("forward",ln2,o.out,T*M,3e-5f);bad+=closev("initial_loss",lossv,&oloss,1,3e-5f);bad+=closev("dx",gx,o.dx,T*M,5e-5f);bad+=closev("dwq",gwq,o.dwq,M*QW,5e-5f);bad+=closev("dwo",gwo,o.dwo,M*M,5e-5f);bad+=closev("dw1",gw1,o.dw1,M*F,5e-5f);bad+=closev("dw2",gw2,o.dw2,F*M,5e-5f);
 for(int i=0;i<M*QW;i++)o.wq[i]-=.02f*o.dwq[i];for(int i=0;i<M*M;i++)o.wo[i]-=.02f*o.dwo[i];for(int i=0;i<M*F;i++)o.w1[i]-=.02f*o.dw1[i];for(int i=0;i<F*M;i++)o.w2[i]-=.02f*o.dw2[i];
 if(tensor_transformer_steps_execute(steps+40,4))return 2;
 bad+=closev("updated_wq",wq,o.wq,M*QW,5e-5f);bad+=closev("updated_wo",wo,o.wo,M*M,5e-5f);bad+=closev("updated_w1",w1,o.w1,M*F,5e-5f);bad+=closev("updated_w2",w2,o.w2,F*M,5e-5f);
 if(bad){fprintf(stderr,"DIFF transformer failed: mismatched_fields=%d shape=T3/M4/H2/D2/F6\n",bad);return 3;}
 printf("differential Transformer passed: forward=equivalent initial_loss=%.6f dx=equivalent parameter_gradients=equivalent update=equivalent steps=%u\n",lossv[0],count);return 0;
}
