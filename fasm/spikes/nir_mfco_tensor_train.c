#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct { float *data,*grad; uint64_t rows,cols; } Tensor;
typedef struct { uint32_t op,lhs,rhs,flags; Tensor *tensor; uint32_t slot,gen; } Node;
typedef struct { void *forward,*backward; Tensor *lhs,*rhs,*out; } Step;
typedef struct { Tensor *a,*w,*bias,*matmul_out,*bias_out,*out; uint64_t action_flags; } FusionContext;
enum { LEAF, MATMUL, RELU, MSE, BIAS };
enum { PARAM=1, CONSTANT=2, INPUT=4, TEMP=8 };
#define NONE UINT32_MAX

extern int tensor_plan_compile_fused_f32(const Node*,uint64_t,Step*,uint64_t,FusionContext*,uint64_t,uint64_t*,uint64_t*);
extern int tensor_plan_forward_f32(const Step*,uint64_t);
extern int tensor_plan_backward_f32(const Step*,uint64_t);
extern int tensor_parameters_zero_grad_f32(Node*,uint64_t);
extern int tensor_parameters_sgd_f32(Node*,uint64_t,float);
extern int tensor_matmul_forward_f32(const Tensor*,const Tensor*,Tensor*);
extern int tensor_bias_add_forward_f32(const Tensor*,const Tensor*,Tensor*);
extern int tensor_relu_forward_f32(const Tensor*,Tensor*);

typedef struct { char magic[8]; uint32_t rows,features,reserved; } Header;

static void *zalloc(size_t count){ void *p=calloc(count,sizeof(float)); if(!p){perror("calloc");exit(2);} return p; }
static int read_exact(void *p,size_t size,FILE *f){return fread(p,1,size,f)==size?0:-1;}
static uint32_t rng=0x4d46434fu;
static float random_weight(void){rng=rng*1664525u+1013904223u;return ((int)(rng>>9)%2049-1024)/1024.0f;}

int main(int argc,char **argv){
 if(argc!=2){fprintf(stderr,"usage: %s fixture.bin\n",argv[0]);return 2;}
 FILE *f=fopen(argv[1],"rb");if(!f){perror(argv[1]);return 2;}
 Header h;if(read_exact(&h,sizeof h,f)||memcmp(h.magic,"NIRMFCO1",8)||h.rows!=880||h.features!=60){fprintf(stderr,"bad NIR-MFCO fixture\n");return 2;}
 const uint32_t train_rows=748,test_rows=132,features=h.features,hidden=16;
 float *train_x=zalloc((size_t)train_rows*features),*train_y=zalloc(train_rows);
 float *test_x=zalloc((size_t)test_rows*features),*test_y=zalloc(test_rows);
 uint32_t ti=0,vi=0;
 for(uint32_t row=0;row<h.rows;row++){
  uint32_t test;float values[60],label;
  if(read_exact(&test,4,f)||read_exact(values,sizeof values,f)||read_exact(&label,4,f)){fprintf(stderr,"truncated fixture\n");return 2;}
  uint32_t at=test?vi++:ti++;
  float *dst=test?test_x+(size_t)at*features:train_x+(size_t)at*features;
  memcpy(dst,values,sizeof values);(test?test_y:train_y)[at]=label;
 }
 fclose(f);if(ti!=train_rows||vi!=test_rows){fprintf(stderr,"bad split %u/%u\n",ti,vi);return 2;}

 float *w1=zalloc(features*hidden),*w1g=zalloc(features*hidden),*b1=zalloc(hidden),*b1g=zalloc(hidden);
 float *w2=zalloc(hidden),*w2g=zalloc(hidden),*b2=zalloc(1),*b2g=zalloc(1);
 float *z1=zalloc(train_rows*hidden),*z1g=zalloc(train_rows*hidden),*z1b=zalloc(train_rows*hidden),*z1bg=zalloc(train_rows*hidden);
 float *act=zalloc(train_rows*hidden),*actg=zalloc(train_rows*hidden),*z2=zalloc(train_rows),*z2g=zalloc(train_rows),*pred=zalloc(train_rows),*predg=zalloc(train_rows);
 float loss=0,lossg=1;
 for(uint32_t i=0;i<features*hidden;i++)w1[i]=random_weight()*.08f;
 for(uint32_t i=0;i<hidden;i++)w2[i]=random_weight()*.08f;
 Tensor t[12]={{train_x,0,train_rows,features},{w1,w1g,features,hidden},{b1,b1g,1,hidden},{w2,w2g,hidden,1},{b2,b2g,1,1},
  {train_y,0,train_rows,1},{z1,z1g,train_rows,hidden},{z1b,z1bg,train_rows,hidden},{act,actg,train_rows,hidden},
  {z2,z2g,train_rows,1},{pred,predg,train_rows,1},{&loss,&lossg,1,1}};
 Node n[12]={{LEAF,NONE,NONE,INPUT,&t[0],0,1},{LEAF,NONE,NONE,PARAM,&t[1],0,1},{LEAF,NONE,NONE,PARAM,&t[2],0,1},
  {LEAF,NONE,NONE,PARAM,&t[3],0,1},{LEAF,NONE,NONE,PARAM,&t[4],0,1},{LEAF,NONE,NONE,CONSTANT,&t[5],0,1},
  {MATMUL,0,1,TEMP,&t[6],1,1},{BIAS,6,2,TEMP,&t[7],1,1},{RELU,7,NONE,TEMP,&t[8],1,1},
  {MATMUL,8,3,TEMP,&t[9],1,1},{BIAS,9,4,TEMP,&t[10],1,1},{MSE,10,5,TEMP,&t[11],1,1}};
 Step plan[3];FusionContext contexts[2];uint64_t pc=0,cc=0;
 if(tensor_plan_compile_fused_f32(n,12,plan,3,contexts,2,&pc,&cc)||pc!=3||cc!=2)return 3;
 for(uint32_t epoch=0;epoch<5000;epoch++){
  if(tensor_parameters_zero_grad_f32(n,12)||tensor_plan_forward_f32(plan,pc)||tensor_plan_backward_f32(plan,pc)||tensor_parameters_sgd_f32(n,12,.8f))return 4;
 }

 float *tz1=zalloc(test_rows*hidden),*tact=zalloc(test_rows*hidden),*tout=zalloc(test_rows),*tpred=zalloc(test_rows);
 Tensor tx={test_x,0,test_rows,features},tw1={w1,0,features,hidden},tb1={b1,0,1,hidden},a={tz1,0,test_rows,hidden},actt={tact,0,test_rows,hidden};
 Tensor tw2={w2,0,hidden,1},tb2={b2,0,1,1},o={tout,0,test_rows,1},p={tpred,0,test_rows,1};
 if(tensor_matmul_forward_f32(&tx,&tw1,&a)||tensor_bias_add_forward_f32(&a,&tb1,&actt)||tensor_relu_forward_f32(&actt,&actt)||
    tensor_matmul_forward_f32(&actt,&tw2,&o)||tensor_bias_add_forward_f32(&o,&tb2,&p))return 5;
 double ae=0,se=0,mean=0,tot=0;for(uint32_t i=0;i<test_rows;i++){double e=tpred[i]-test_y[i];ae+=fabs(e);se+=e*e;mean+=test_y[i];}mean/=test_rows;
 for(uint32_t i=0;i<test_rows;i++){double d=test_y[i]-mean;tot+=d*d;}
 double mae=ae/test_rows,rmse=sqrt(se/test_rows),r2=1-se/tot;
 printf("nir-mfco tensor architecture=60x16x1 samples=880 train=748 test=132 plan_steps=%llu epochs=5000 loss=%.6f mae_pp=%.3f rmse_pp=%.3f r2=%.5f\n",
  (unsigned long long)pc,loss,mae*100,rmse*100,r2);
 return isfinite(mae)&&mae<.03?0:1;
}
