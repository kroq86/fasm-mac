#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
typedef struct{float*data,*grad;uint64_t rows,cols;}Tensor;
typedef struct{uint32_t op,lhs,rhs,flags;Tensor*tensor;uint32_t slot,gen;}Node;
typedef struct{void*forward,*backward;Tensor*lhs,*rhs,*out;}Step;
enum{LEAF,MATMUL,RELU,MSE,BIAS};enum{PARAM=1,CONST=2,INPUT=4,TEMP=8};
#define NONE UINT32_MAX
extern int tensor_plan_compile_f32(const Node*,uint64_t,Step*,uint64_t,uint64_t*);
extern int tensor_plan_forward_f32(const Step*,uint64_t);
extern int tensor_plan_backward_f32(const Step*,uint64_t);
int main(void){
 float xd[4]={0,0,1,1},xg[4]={0},wd[2]={.5f,-.25f},wg[2]={0};
 float bd[1]={.1f},bg[1]={0},md[2]={0},mg[2]={0},zd[2]={0},zg[2]={0};
 float td[2]={.1f,.4f},tg[2]={0},ld=0,lg=1;
 Tensor t[7]={{xd,xg,2,2},{wd,wg,2,1},{bd,bg,1,1},{td,tg,2,1},
              {md,mg,2,1},{zd,zg,2,1},{&ld,&lg,1,1}};
 Node n[7]={{LEAF,NONE,NONE,INPUT,&t[0],0,1},{LEAF,NONE,NONE,PARAM,&t[1],0,1},
 {LEAF,NONE,NONE,PARAM,&t[2],0,1},{LEAF,NONE,NONE,CONST,&t[3],0,1},
 {MATMUL,0,1,TEMP,&t[4],1,1},{BIAS,4,2,TEMP,&t[5],1,1},{MSE,5,3,TEMP,&t[6],1,1}};
 Step steps[3];uint64_t count=0;
 if(tensor_plan_compile_f32(n,7,steps,3,&count)||count!=3)return 1;
 if(tensor_plan_forward_f32(steps,count)||tensor_plan_backward_f32(steps,count))return 1;
 if(fabsf(ld-.00125f)>1e-6f||fabsf(wg[0]+.05f)>1e-6f||fabsf(wg[1]+.05f)>1e-6f)return 1;
 uint64_t untouched=99;if(tensor_plan_compile_f32(n,7,steps,2,&untouched)!=-3||untouched!=99)return 1;
 printf("tensor plan spike passed: graph_nodes=7 direct_steps=%llu loss=%g\n",count,ld);return 0;
}
