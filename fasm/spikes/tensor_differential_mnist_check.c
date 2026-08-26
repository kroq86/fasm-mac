/* MNIST snapshot differential: canonical schedule vs the retained scalar
 * graph-engine math. Consumer-only: compiler and oracle contracts are not
 * changed here. Requires extracted IDX files in MNIST_DIR. */
#include "tensor_semantic_compiler.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum { IN=784,HID=32,OUT=10 };
enum { X,W1,B1,W2,B2,TARGET,MM1,ADD1,ACT1,MM2,ADD2,LOSS,NODES };
typedef struct { float w1[IN*HID],b1[HID],w2[HID*OUT],b2[OUT]; } Params;
typedef struct { float x[IN],z[HID],h[HID],out[OUT],dw1[IN*HID],db1[HID],dw2[HID*OUT],db2[OUT]; } Oracle;

static uint32_t be32(const unsigned char*p){return((uint32_t)p[0]<<24)|((uint32_t)p[1]<<16)|((uint32_t)p[2]<<8)|p[3];}
static unsigned char *read_idx(const char*path,uint32_t magic,uint32_t*n,uint32_t*rows,uint32_t*cols){
 FILE*f=fopen(path,"rb");if(!f)return NULL;unsigned char h[16];size_t hs=magic==2051?16:8;if(fread(h,1,hs,f)!=hs||be32(h)!=magic){fclose(f);return NULL;}*n=be32(h+4);if(rows)*rows=be32(h+8);if(cols)*cols=be32(h+12);size_t bytes=(size_t)*n*(magic==2051?(size_t)*rows**cols:1);unsigned char*p=malloc(bytes);if(!p||fread(p,1,bytes,f)!=bytes){free(p);p=NULL;}fclose(f);return p;
}
static float initv(int i,int s){return(float)(((i*37+s*17)%29)-14)/41.0f;}
static void init_params(Params*p){memset(p,0,sizeof*p);for(int i=0;i<IN*HID;i++)p->w1[i]=initv(i,2)*.05f;for(int i=0;i<HID*OUT;i++)p->w2[i]=initv(i,5)*.1f;}
static float oracle_forward(Oracle*o,const Params*p,int label){
 for(int j=0;j<HID;j++){float s=p->b1[j];for(int i=0;i<IN;i++)s+=o->x[i]*p->w1[i*HID+j];o->z[j]=s;o->h[j]=s>0?s:0;}
 float loss=0;for(int k=0;k<OUT;k++){float s=p->b2[k];for(int j=0;j<HID;j++)s+=o->h[j]*p->w2[j*OUT+k];o->out[k]=s;float e=s-(k==label);loss+=e*e;}return loss/OUT;
}
static void oracle_backward(Oracle*o,const Params*p,int label){
 memset(o->dw1,0,sizeof o->dw1);memset(o->db1,0,sizeof o->db1);memset(o->dw2,0,sizeof o->dw2);memset(o->db2,0,sizeof o->db2);float dh[HID]={0};
 for(int k=0;k<OUT;k++){float g=2*(o->out[k]-(k==label))/OUT;o->db2[k]=g;for(int j=0;j<HID;j++){o->dw2[j*OUT+k]=o->h[j]*g;dh[j]+=p->w2[j*OUT+k]*g;}}
 for(int j=0;j<HID;j++){float g=o->z[j]>0?dh[j]:0;o->db1[j]=g;for(int i=0;i<IN;i++)o->dw1[i*HID+j]=o->x[i]*g;}
}
static void oracle_update(Params*p,const Oracle*o,float lr){for(int i=0;i<IN*HID;i++)p->w1[i]-=lr*o->dw1[i];for(int i=0;i<HID;i++)p->b1[i]-=lr*o->db1[i];for(int i=0;i<HID*OUT;i++)p->w2[i]-=lr*o->dw2[i];for(int i=0;i<OUT;i++)p->b2[i]-=lr*o->db2[i];}
static int closev(const char*n,const float*a,const float*b,size_t count,float tol){for(size_t i=0;i<count;i++)if(fabsf(a[i]-b[i])>tol*fmaxf(1,fmaxf(fabsf(a[i]),fabsf(b[i])))){fprintf(stderr,"DIFF mnist %s[%zu]: canonical=%.9g oracle=%.9g\n",n,i,a[i],b[i]);return 1;}return 0;}
static void load_sample(float*x,const unsigned char*images,unsigned s){for(int i=0;i<IN;i++)x[i]=images[(size_t)s*IN+i]/255.0f;}

int main(void){
 const char*dir=getenv("MNIST_DIR");if(!dir){fprintf(stderr,"MNIST snapshot skipped: set MNIST_DIR\n");return 77;}char path[1024];uint32_t tn,tln,te,ten,r,c;
 snprintf(path,sizeof path,"%s/train-images-idx3-ubyte",dir);unsigned char*ti=read_idx(path,2051,&tn,&r,&c);snprintf(path,sizeof path,"%s/train-labels-idx1-ubyte",dir);unsigned char*tl=read_idx(path,2049,&tln,NULL,NULL);snprintf(path,sizeof path,"%s/t10k-images-idx3-ubyte",dir);unsigned char*ei=read_idx(path,2051,&te,&r,&c);snprintf(path,sizeof path,"%s/t10k-labels-idx1-ubyte",dir);unsigned char*el=read_idx(path,2049,&ten,NULL,NULL);if(!ti||!tl||!ei||!el||tn!=tln||te!=ten){fprintf(stderr,"MNIST snapshot: IDX load failed\n");return 2;}
 Params cp,op;init_params(&cp);op=cp;Oracle o={0};float x[IN],target[OUT],mm1[HID],add1[HID],act1[HID],mm2[OUT],add2[OUT],loss[1],gx[IN]={0},gw1[IN*HID],gb1[HID],gw2[HID*OUT],gb2[OUT],gt[OUT]={0},gmm1[HID],gadd1[HID],gact1[HID],gmm2[OUT],gadd2[OUT],gloss[1];
 Node g[NODES]={{LEAF,NONE,NONE,INPUT,{x,gx,NULL,1,IN,0}},{LEAF,NONE,NONE,PARAM,{cp.w1,gw1,NULL,IN,HID,0}},{LEAF,NONE,NONE,PARAM,{cp.b1,gb1,NULL,1,HID,0}},{LEAF,NONE,NONE,PARAM,{cp.w2,gw2,NULL,HID,OUT,0}},{LEAF,NONE,NONE,PARAM,{cp.b2,gb2,NULL,1,OUT,0}},{LEAF,NONE,NONE,CONSTANT,{target,gt,NULL,1,OUT,0}},{MATMUL,X,W1,TEMP,{mm1,gmm1,NULL,1,HID,0}},{BIAS_ADD,MM1,B1,TEMP,{add1,gadd1,NULL,1,HID,0}},{RELU,ADD1,NONE,TEMP,{act1,gact1,NULL,1,HID,0}},{MATMUL,ACT1,W2,TEMP,{mm2,gmm2,NULL,1,OUT,0}},{BIAS_ADD,MM2,B2,TEMP,{add2,gadd2,NULL,1,OUT,0}},{MSE,ADD2,TARGET,TEMP,{loss,gloss,NULL,1,1,0}}};ExecStep steps[32];Context ctx[32];uint32_t count=0;if(compile(g,NODES,.05f,steps,ctx,32,&count)||count!=25)return 2;
 float checkpoints_c[3],checkpoints_o[3],initial_c,initial_o;int bad=0;unsigned train=tn<3000?tn:3000;
 for(unsigned step=0;step<8*train;step++){unsigned s=step%train;load_sample(x,ti,s);memcpy(o.x,x,sizeof x);for(int k=0;k<OUT;k++)target[k]=tl[s]==k;initial_o=oracle_forward(&o,&op,tl[s]);if(tensor_transformer_steps_execute(steps,6))return 2;initial_c=loss[0];if(step==0){bad+=closev("initial_loss",&initial_c,&initial_o,1,2e-5f);if(tensor_transformer_steps_execute(steps+6,15))return 2;oracle_backward(&o,&op,tl[s]);bad+=closev("gw1",gw1,o.dw1,IN*HID,3e-5f);bad+=closev("gb1",gb1,o.db1,HID,3e-5f);bad+=closev("gw2",gw2,o.dw2,HID*OUT,3e-5f);bad+=closev("gb2",gb2,o.db2,OUT,3e-5f);if(tensor_transformer_steps_execute(steps+21,4))return 2;oracle_update(&op,&o,.05f);bad+=closev("updated_w1",cp.w1,op.w1,IN*HID,3e-5f);bad+=closev("updated_b1",cp.b1,op.b1,HID,3e-5f);bad+=closev("updated_w2",cp.w2,op.w2,HID*OUT,3e-5f);bad+=closev("updated_b2",cp.b2,op.b2,OUT,3e-5f);}else{if(tensor_transformer_steps_execute(steps+6,19))return 2;oracle_backward(&o,&op,tl[s]);oracle_update(&op,&o,.05f);}if(step==0||step==9||step==99){int q=step==0?0:step==9?1:2;load_sample(x,ti,s);memcpy(o.x,x,sizeof x);checkpoints_o[q]=oracle_forward(&o,&op,tl[s]);if(tensor_transformer_steps_execute(steps,6))return 2;checkpoints_c[q]=loss[0];}}
 bad+=closev("checkpoint_loss",checkpoints_c,checkpoints_o,3,4e-5f);unsigned ca=0,oa=0,test=te<1000?te:1000;for(unsigned s=0;s<test;s++){load_sample(x,ei,s);memcpy(o.x,x,sizeof x);oracle_forward(&o,&op,el[s]);FWD[MATMUL](&g[MM1],&g[X],&g[W1]);FWD[BIAS_ADD](&g[ADD1],&g[MM1],&g[B1]);FWD[RELU](&g[ACT1],&g[ADD1],NULL);FWD[MATMUL](&g[MM2],&g[ACT1],&g[W2]);FWD[BIAS_ADD](&g[ADD2],&g[MM2],&g[B2]);int pc=0,po=0;for(int k=1;k<OUT;k++){if(add2[k]>add2[pc])pc=k;if(o.out[k]>o.out[po])po=k;}ca+=pc==el[s];oa+=po==el[s];}if(ca!=oa)bad++,fprintf(stderr,"DIFF mnist accuracy canonical=%u oracle=%u\n",ca,oa);free(ti);free(tl);free(ei);free(el);if(bad)return 3;printf("MNIST differential passed: initial_loss=equivalent grads=equivalent first_update=equivalent checkpoints(1,10,100)=equivalent final_accuracy=%.1f%% vs %.1f%%\n",100.0*ca/test,100.0*oa/test);return 0;
}
