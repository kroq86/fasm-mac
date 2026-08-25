#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
typedef struct{float*data,*grad;uint64_t rows,cols;}Tensor;
extern int tensor_matmul_forward_f32(const Tensor*,const Tensor*,Tensor*);
extern int tensor_matmul_backward_f32(Tensor*,Tensor*,const Tensor*);
extern int tensor_matmul_backward_lhs_f32(Tensor*,Tensor*,const Tensor*);
extern int tensor_matmul_backward_rhs_f32(Tensor*,Tensor*,const Tensor*);

enum{GUARD=8};static const float CANARY=1234567.0f;
typedef struct{float*base,*p;size_t count;}Guarded;
static Guarded alloc_guard(size_t n){Guarded g={calloc(n+2*GUARD,sizeof(float)),0,n};if(!g.base)return g;g.p=g.base+GUARD;for(int i=0;i<GUARD;i++)g.base[i]=g.base[GUARD+n+i]=CANARY;return g;}
static int guard_ok(Guarded g){for(int i=0;i<GUARD;i++)if(g.base[i]!=CANARY||g.base[GUARD+g.count+i]!=CANARY)return 0;return 1;}
static uint32_t rng=0x91e10da5u;static float nextf(void){rng=rng*1664525u+1013904223u;return ((int)(rng>>9)%2049-1024)/1024.0f;}
static int closev(float a,float b){float d=fabsf(a-b),s=fmaxf(1.0f,fmaxf(fabsf(a),fabsf(b)));return d<=2e-5f*s;}
static int one(uint64_t m,uint64_t k,uint64_t n){
 size_t ac=m*k,bc=k*n,oc=m*n;Guarded ad=alloc_guard(ac),bd=alloc_guard(bc),od=alloc_guard(oc),og=alloc_guard(oc);
 Guarded ag=alloc_guard(ac),bg=alloc_guard(bc),lag=alloc_guard(ac),rbg=alloc_guard(bc),ref=alloc_guard(oc);
 if(!ad.base||!bd.base||!od.base||!og.base||!ag.base||!bg.base||!lag.base||!rbg.base||!ref.base)return 1;
 for(size_t i=0;i<ac;i++)ad.p[i]=nextf();for(size_t i=0;i<bc;i++)bd.p[i]=nextf();for(size_t i=0;i<oc;i++)og.p[i]=nextf();
 for(uint64_t i=0;i<m;i++)for(uint64_t j=0;j<n;j++)for(uint64_t q=0;q<k;q++)ref.p[i*n+j]+=ad.p[i*k+q]*bd.p[q*n+j];
 Tensor a={ad.p,ag.p,m,k},b={bd.p,bg.p,k,n},o={od.p,og.p,m,n};
 if(tensor_matmul_forward_f32(&a,&b,&o)||tensor_matmul_backward_f32(&a,&b,&o))return 1;
 for(size_t i=0;i<oc;i++)if(!closev(od.p[i],ref.p[i]))return 1;
 Tensor al={ad.p,lag.p,m,k},bn={bd.p,0,k,n};if(tensor_matmul_backward_lhs_f32(&al,&bn,&o))return 1;
 Tensor an={ad.p,0,m,k},br={bd.p,rbg.p,k,n};if(tensor_matmul_backward_rhs_f32(&an,&br,&o))return 1;
 for(size_t i=0;i<ac;i++)if(!closev(ag.p[i],lag.p[i]))return 1;
 for(size_t i=0;i<bc;i++)if(!closev(bg.p[i],rbg.p[i]))return 1;
 Guarded all[]={ad,bd,od,og,ag,bg,lag,rbg,ref};for(size_t i=0;i<sizeof(all)/sizeof(all[0]);i++)if(!guard_ok(all[i]))return 1;
 for(size_t i=0;i<sizeof(all)/sizeof(all[0]);i++)free(all[i].base);return 0;
}
int main(void){
 static const uint8_t dims[]={1,2,3,7,8,9,15,16,17,31,32,33};unsigned cases=0;
 for(unsigned i=0;i<144;i++){uint64_t m=dims[(i*5+1)%12],k=dims[(i*7+3)%12],n=dims[(i*11+4)%12];if(one(m,k,n)){fprintf(stderr,"volume failure case=%u shape=%llux%llux%llu\n",i,m,k,n);return 1;}cases++;}
 printf("tensor volume stress passed: cases=%u max_dim=33 guards=8 forward=reference backward=lhs/rhs/both\n",cases);return 0;
}
