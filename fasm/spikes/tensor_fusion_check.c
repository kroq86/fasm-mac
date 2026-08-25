#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
typedef struct { float *data,*grad; uint64_t rows,cols; } Tensor;
typedef struct { void *forward,*backward; Tensor *lhs,*rhs,*out; } Step;
typedef struct { Tensor *a,*w,*bias,*matmul_out,*bias_out,*out; uint64_t action_flags; } FusionContext;
extern int tensor_matmul_forward_f32(Tensor*,Tensor*,Tensor*);
extern int tensor_matmul_backward_f32(Tensor*,Tensor*,Tensor*);
extern int tensor_bias_add_forward_f32(Tensor*,Tensor*,Tensor*);
extern int tensor_bias_add_backward_f32(Tensor*,Tensor*,Tensor*);
extern int tensor_relu_forward_f32(Tensor*,Tensor*);
extern int tensor_relu_backward_f32(Tensor*,Tensor*);
extern int tensor_matmul_bias_forward_f32(Tensor*,Tensor*,Tensor*,Tensor*,Tensor*,Tensor*);
extern int tensor_matmul_bias_backward_f32(Tensor*,Tensor*,Tensor*,Tensor*,Tensor*,Tensor*);
extern int tensor_matmul_bias_relu_forward_f32(Tensor*,Tensor*,Tensor*,Tensor*,Tensor*,Tensor*);
extern int tensor_matmul_bias_relu_backward_f32(Tensor*,Tensor*,Tensor*,Tensor*,Tensor*,Tensor*);
extern int tensor_plan_forward_f32(const Step*,uint64_t);
extern int tensor_plan_backward_f32(const Step*,uint64_t);
extern int tensor_plan_matmul_bias_relu_forward_f32(void*,void*,void*);
extern int tensor_plan_matmul_bias_relu_backward_f32(void*,void*,void*);
static int closev(const float*a,const float*b,int n){for(int i=0;i<n;i++)if(fabsf(a[i]-b[i])>1e-6f)return 0;return 1;}
int main(void){
 float ad[]={1,-2,3,4,0.5f,-1},wd[]={.2f,-.3f,.7f,.1f,-.4f,.8f},bd[]={.25f,-.5f};
 float ag0[6]={0},wg0[6]={0},bg0[2]={0},mg0[4]={0},m0[4]={0},zg0[4]={0},z0[4]={0},og0[]={1,-2,.5f,3},o0[4]={0};
 float ag1[6]={0},wg1[6]={0},bg1[2]={0},mg1[4]={0},m1[4]={0},zg1[4]={0},z1[4]={0},og1[]={1,-2,.5f,3},o1[4]={0};
 Tensor a0={ad,ag0,2,3},w0={wd,wg0,3,2},b0={bd,bg0,1,2},m_0={m0,mg0,2,2},z_0={z0,zg0,2,2},o_0={o0,og0,2,2};
 Tensor a1={ad,ag1,2,3},w1={wd,wg1,3,2},b1={bd,bg1,1,2},m_1={m1,mg1,2,2},z_1={z1,zg1,2,2},o_1={o1,og1,2,2};
 if(tensor_matmul_forward_f32(&a0,&w0,&m_0)||tensor_bias_add_forward_f32(&m_0,&b0,&o_0))return 1;
 if(tensor_bias_add_backward_f32(&m_0,&b0,&o_0)||tensor_matmul_backward_f32(&a0,&w0,&m_0))return 1;
 if(tensor_matmul_bias_forward_f32(&a1,&w1,&b1,&m_1,&z_1,&o_1)||tensor_matmul_bias_backward_f32(&a1,&w1,&b1,&m_1,&z_1,&o_1))return 1;
 if(!closev(o0,o1,4)||!closev(ag0,ag1,6)||!closev(wg0,wg1,6)||!closev(bg0,bg1,2))return 1;
 memset(ag0,0,sizeof ag0);memset(wg0,0,sizeof wg0);memset(bg0,0,sizeof bg0);memset(mg0,0,sizeof mg0);memset(zg0,0,sizeof zg0);
 memset(ag1,0,sizeof ag1);memset(wg1,0,sizeof wg1);memset(bg1,0,sizeof bg1);memset(mg1,0,sizeof mg1);memset(zg1,0,sizeof zg1);
 if(tensor_matmul_forward_f32(&a0,&w0,&m_0)||tensor_bias_add_forward_f32(&m_0,&b0,&z_0)||tensor_relu_forward_f32(&z_0,&o_0))return 1;
 if(tensor_relu_backward_f32(&z_0,&o_0)||tensor_bias_add_backward_f32(&m_0,&b0,&z_0)||tensor_matmul_backward_f32(&a0,&w0,&m_0))return 1;
 if(tensor_matmul_bias_relu_forward_f32(&a1,&w1,&b1,&m_1,&z_1,&o_1)||tensor_matmul_bias_relu_backward_f32(&a1,&w1,&b1,&m_1,&z_1,&o_1))return 1;
 if(!closev(o0,o1,4)||!closev(ag0,ag1,6)||!closev(wg0,wg1,6)||!closev(bg0,bg1,2))return 1;
 memset(ag1,0,sizeof ag1);memset(wg1,0,sizeof wg1);memset(bg1,0,sizeof bg1);memset(mg1,0,sizeof mg1);memset(zg1,0,sizeof zg1);memset(o1,0,sizeof o1);
 FusionContext context={&a1,&w1,&b1,&m_1,&z_1,&o_1};
 Step fused={(void*)tensor_plan_matmul_bias_relu_forward_f32,(void*)tensor_plan_matmul_bias_relu_backward_f32,(Tensor*)&context,0,0};
 if(tensor_plan_forward_f32(&fused,1)||tensor_plan_backward_f32(&fused,1))return 1;
 if(!closev(o0,o1,4)||!closev(ag0,ag1,6)||!closev(wg0,wg1,6)||!closev(bg0,bg1,2))return 1;
 puts("tensor fusion spike passed: semantic_ops=3 lowered_steps=1 PlanStep=40 context=56 scalar-reference");return 0;
}
