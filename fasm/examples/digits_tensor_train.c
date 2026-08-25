#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

typedef struct { float *data,*grad; uint64_t rows,cols; } Tensor;
typedef struct { uint32_t op,lhs,rhs,flags; Tensor *tensor; uint32_t slot,gen; } Node;
typedef struct { void *forward,*backward; Tensor *lhs,*rhs,*out; } Step;
typedef struct { Tensor *a,*w,*bias,*matmul_out,*bias_out,*out; uint64_t action_flags; } FusionContext;
enum { LEAF, MATMUL, RELU, MSE, BIAS };
enum { PARAM=1, CONSTANT=2, INPUT=4, TEMP=8 };
#define NONE UINT32_MAX

extern int tensor_plan_compile_fused_f32(const Node*,uint64_t,Step*,uint64_t,FusionContext*,uint64_t,uint64_t*,uint64_t*);
extern int tensor_plan_compile_f32(const Node*,uint64_t,Step*,uint64_t,uint64_t*);
extern int tensor_plan_forward_f32(const Step*,uint64_t);
extern int tensor_plan_backward_f32(const Step*,uint64_t);
extern int tensor_parameters_zero_grad_f32(Node*,uint64_t);
extern int tensor_parameters_sgd_f32(Node*,uint64_t,float);
static int same(const float*a,const float*b,size_t n){for(size_t i=0;i<n;i++)if(fabsf(a[i]-b[i])>1e-6f)return 0;return 1;}

static const float digits[70]={
 1,1,1,1,1,1,0, 0,1,1,0,0,0,0, 1,1,0,1,1,0,1, 1,1,1,1,0,0,1,
 0,1,1,0,0,1,1, 1,0,1,1,0,1,1, 1,0,1,1,1,1,1, 1,1,1,0,0,0,0,
 1,1,1,1,1,1,1, 1,1,1,1,0,1,1
};

int main(void){
 float x[70],xg[70]={0},target[100]={0};memcpy(x,digits,sizeof x);
 float w1[112],w1g[112]={0},b1[16]={0},b1g[16]={0};
 float w2[160],w2g[160]={0},b2[10]={0},b2g[10]={0};
 float z1[160]={0},z1g[160]={0},z1b[160]={0},z1bg[160]={0},h[160]={0},hg[160]={0};
 float z2[100]={0},z2g[100]={0},pred[100]={0},predg[100]={0},loss=0,lossg=1;
 for(int i=0;i<10;i++)target[i*10+i]=1;
 for(int i=0;i<112;i++)w1[i]=((i*37%29)-14)*0.012f;
 for(int i=0;i<160;i++)w2[i]=((i*53%31)-15)*0.010f;
 Tensor t[12]={{x,xg,10,7},{w1,w1g,7,16},{b1,b1g,1,16},{w2,w2g,16,10},
  {b2,b2g,1,10},{target,0,10,10},{z1,z1g,10,16},{z1b,z1bg,10,16},
  {h,hg,10,16},{z2,z2g,10,10},{pred,predg,10,10},{&loss,&lossg,1,1}};
 Node n[12]={{LEAF,NONE,NONE,INPUT,&t[0],0,1},{LEAF,NONE,NONE,PARAM,&t[1],0,1},
  {LEAF,NONE,NONE,PARAM,&t[2],0,1},{LEAF,NONE,NONE,PARAM,&t[3],0,1},
  {LEAF,NONE,NONE,PARAM,&t[4],0,1},{LEAF,NONE,NONE,CONSTANT,&t[5],0,1},
  {MATMUL,0,1,TEMP,&t[6],1,1},{BIAS,6,2,TEMP,&t[7],1,1},{RELU,7,NONE,TEMP,&t[8],1,1},
  {MATMUL,8,3,TEMP,&t[9],1,1},{BIAS,9,4,TEMP,&t[10],1,1},{MSE,10,5,TEMP,&t[11],1,1}};
 Step plan[3],unfused[6];FusionContext contexts[2];uint64_t pc=0,cc=0,uc=0;
 if(tensor_plan_compile_fused_f32(n,12,plan,3,contexts,2,&pc,&cc)||pc!=3||cc!=2)return 2;
 if(tensor_plan_compile_f32(n,12,unfused,6,&uc)||uc!=6)return 2;
 float ref_pred[100],ref_w1g[112],ref_b1g[16],ref_w2g[160],ref_b2g[10];
 if(tensor_plan_forward_f32(unfused,uc)||tensor_plan_backward_f32(unfused,uc))return 3;
 memcpy(ref_pred,pred,sizeof pred);memcpy(ref_w1g,w1g,sizeof w1g);memcpy(ref_b1g,b1g,sizeof b1g);
 memcpy(ref_w2g,w2g,sizeof w2g);memcpy(ref_b2g,b2g,sizeof b2g);
 memset(z1g,0,sizeof z1g);memset(z1bg,0,sizeof z1bg);memset(hg,0,sizeof hg);memset(z2g,0,sizeof z2g);memset(predg,0,sizeof predg);
 if(tensor_parameters_zero_grad_f32(n,12)||tensor_plan_forward_f32(plan,pc)||tensor_plan_backward_f32(plan,pc))return 3;
 if(!same(ref_pred,pred,100)||!same(ref_w1g,w1g,112)||!same(ref_b1g,b1g,16)||!same(ref_w2g,w2g,160)||!same(ref_b2g,b2g,10))return 3;
 for(int epoch=0;epoch<12000;epoch++){
  if(tensor_parameters_zero_grad_f32(n,12))return 3;
  if(tensor_plan_forward_f32(plan,pc)||tensor_plan_backward_f32(plan,pc))return 4;
  if(tensor_parameters_sgd_f32(n,12,.35f))return 5;
 }
 if(tensor_plan_forward_f32(plan,pc))return 6;
 int correct=0,guess[10];
 for(int i=0;i<10;i++){
  int best=0;for(int j=1;j<10;j++)if(pred[i*10+j]>pred[i*10+best])best=j;
  guess[i]=best;correct+=best==i;
 }
 printf("digits trained architecture=7x16x10 plan_steps=6->%llu reference=exact accuracy=%d/10 loss_milli=%d predictions=",
        (unsigned long long)pc,correct,(int)(loss*1000));
 for(int i=0;i<10;i++)printf("%s%d",i?",":"",guess[i]);
 putchar('\n');
 return correct==10&&loss<.01f?0:1;
}
