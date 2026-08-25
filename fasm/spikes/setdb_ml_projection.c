#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
typedef struct{float*data,*grad;uint64_t rows,cols;}Tensor;
typedef struct{uint32_t op,lhs,rhs,flags;Tensor*tensor;uint32_t slot,gen;}Node;
typedef struct{void*forward,*backward;Tensor*lhs,*rhs,*out;}Step;
typedef struct{Tensor*a,*w,*bias,*matmul_out,*bias_out,*out;uint64_t action_flags;}FusionContext;
enum{LEAF,MATMUL,RELU,MSE,BIAS};enum{PARAM=1,CONSTANT=2,INPUT=4,TEMP=8};
#define NONE UINT32_MAX
extern int tensor_plan_compile_fused_f32(const Node*,uint64_t,Step*,uint64_t,FusionContext*,uint64_t,uint64_t*,uint64_t*);
extern int tensor_plan_forward_f32(const Step*,uint64_t);extern int tensor_plan_backward_f32(const Step*,uint64_t);
extern int tensor_parameters_zero_grad_f32(Node*,uint64_t);extern int tensor_parameters_sgd_f32(Node*,uint64_t,float);
static const char*features[]={"check","release","docs","tests"};
int main(int argc,char**argv){
 if(argc!=2){fprintf(stderr,"usage: setdb-ml-projection FACTS\n");return 2;}FILE*f=fopen(argv[1],"r");if(!f)return 2;
 char names[8][32]={{0}},line[256],a[32],b[32],c[32];int count=0,labeled=0;float x[32]={0},target[8]={0};
 while(fgets(line,sizeof line,f)){
  if(sscanf(line,"SADD entities %31s",a)==1){if(count<8)strcpy(names[count++],a);continue;}
  if(sscanf(line,"RADD feature %31s %31s",a,b)==2){int e=-1,q=-1;for(int i=0;i<count;i++)if(!strcmp(names[i],a))e=i;for(int i=0;i<4;i++)if(!strcmp(features[i],b))q=i;if(e>=0&&q>=0)x[e*4+q]=1;continue;}
  if(sscanf(line,"RADD label %31s %31s",a,b)==2){int e=-1;for(int i=0;i<count;i++)if(!strcmp(names[i],a))e=i;if(e>=0){target[e]=!strcmp(b,"risky");if(e+1>labeled)labeled=e+1;}continue;}
 }
 fclose(f);if(count!=8||labeled!=6)return 2;
 float xg[32]={0},w1[24],w1g[24]={0},b1[6]={0},b1g[6]={0},w2[6],w2g[6]={0},b2=0,b2g=0;
 float z1[48]={0},z1g[48]={0},z1b[48]={0},z1bg[48]={0},h[48]={0},hg[48]={0},z2[8]={0},z2g[8]={0},pred[8]={0},predg[8]={0},loss=0,lossg=1;
 for(int i=0;i<24;i++)w1[i]=((i*17%19)-9)*.025f;for(int i=0;i<6;i++)w2[i]=((i*11%13)-6)*.03f;
 Tensor t[12]={{x,xg,6,4},{w1,w1g,4,6},{b1,b1g,1,6},{w2,w2g,6,1},{&b2,&b2g,1,1},{target,0,6,1},{z1,z1g,6,6},{z1b,z1bg,6,6},{h,hg,6,6},{z2,z2g,6,1},{pred,predg,6,1},{&loss,&lossg,1,1}};
 Node n[12]={{LEAF,NONE,NONE,INPUT,&t[0],0,1},{LEAF,NONE,NONE,PARAM,&t[1],0,1},{LEAF,NONE,NONE,PARAM,&t[2],0,1},{LEAF,NONE,NONE,PARAM,&t[3],0,1},{LEAF,NONE,NONE,PARAM,&t[4],0,1},{LEAF,NONE,NONE,CONSTANT,&t[5],0,1},{MATMUL,0,1,TEMP,&t[6],1,1},{BIAS,6,2,TEMP,&t[7],1,1},{RELU,7,NONE,TEMP,&t[8],1,1},{MATMUL,8,3,TEMP,&t[9],1,1},{BIAS,9,4,TEMP,&t[10],1,1},{MSE,10,5,TEMP,&t[11],1,1}};
 Step p[3];FusionContext fc[2];uint64_t pc=0,cc=0;if(tensor_plan_compile_fused_f32(n,12,p,3,fc,2,&pc,&cc))return 3;
 for(int e=0;e<10000;e++){if(tensor_parameters_zero_grad_f32(n,12)||tensor_plan_forward_f32(p,pc)||tensor_plan_backward_f32(p,pc)||tensor_parameters_sgd_f32(n,12,.25f))return 3;}
 t[0].rows=t[6].rows=t[7].rows=t[8].rows=t[9].rows=t[10].rows=8;
 if(tensor_plan_forward_f32(p,2))return 3;
 for(int i=6;i<8;i++)printf("RADD ml/class/v1 %s %s\n",names[i],pred[i]>=.5f?"risky":"safe");
 return 0;
}
