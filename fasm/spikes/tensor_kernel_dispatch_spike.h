#ifndef TENSOR_KERNEL_DISPATCH_SPIKE_H
#define TENSOR_KERNEL_DISPATCH_SPIKE_H
/* Experimental scalar/NEON/AVX2 dispatch contract; not a stable core ABI. */
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#if defined(__aarch64__)
#include <arm_neon.h>
#elif defined(__x86_64__)
#include <immintrin.h>
#endif
#endif
typedef void(*Matmul)(const float*,const float*,float*,uint64_t,uint64_t,uint64_t);typedef void(*Binary)(const float*,const float*,float*,uint64_t);typedef void(*Unary)(const float*,float*,uint64_t);typedef void(*Bias)(const float*,const float*,float*,uint64_t,uint64_t);typedef void(*Sgd)(float*,const float*,float,uint64_t);
typedef struct{const char*backend;Matmul matmul;Binary residual;Unary relu,zero;Bias bias;Sgd sgd;}Dispatch;
static void scalar_matmul(const float*a,const float*b,float*o,uint64_t m,uint64_t k,uint64_t n){memset(o,0,m*n*4);for(uint64_t i=0;i<m;i++)for(uint64_t p=0;p<k;p++){float x=a[i*k+p];for(uint64_t j=0;j<n;j++)o[i*n+j]+=x*b[p*n+j];}}
static void scalar_residual(const float*a,const float*b,float*o,uint64_t n){for(uint64_t i=0;i<n;i++)o[i]=a[i]+b[i];}static void scalar_relu(const float*a,float*o,uint64_t n){for(uint64_t i=0;i<n;i++)o[i]=a[i]>0?a[i]:0;}static void scalar_zero(const float*a,float*o,uint64_t n){(void)a;for(uint64_t i=0;i<n;i++)o[i]=0;}static void scalar_bias(const float*a,const float*b,float*o,uint64_t rows,uint64_t cols){for(uint64_t i=0;i<rows;i++)for(uint64_t j=0;j<cols;j++)o[i*cols+j]=a[i*cols+j]+b[j];}static void scalar_sgd(float*p,const float*g,float lr,uint64_t n){for(uint64_t i=0;i<n;i++)p[i]-=lr*g[i];}
#if defined(__aarch64__)
static void native_matmul(const float*a,const float*b,float*o,uint64_t m,uint64_t k,uint64_t n){memset(o,0,m*n*4);for(uint64_t i=0;i<m;i++)for(uint64_t p=0;p<k;p++){float32x4_t x=vdupq_n_f32(a[i*k+p]);uint64_t j=0;for(;j+4<=n;j+=4)vst1q_f32(o+i*n+j,vfmaq_f32(vld1q_f32(o+i*n+j),x,vld1q_f32(b+p*n+j)));for(;j<n;j++)o[i*n+j]+=a[i*k+p]*b[p*n+j];}}
static void native_residual(const float*a,const float*b,float*o,uint64_t n){uint64_t i=0;for(;i+4<=n;i+=4)vst1q_f32(o+i,vaddq_f32(vld1q_f32(a+i),vld1q_f32(b+i)));for(;i<n;i++)o[i]=a[i]+b[i];}static void native_relu(const float*a,float*o,uint64_t n){uint64_t i=0;float32x4_t z=vdupq_n_f32(0);for(;i+4<=n;i+=4)vst1q_f32(o+i,vmaxq_f32(vld1q_f32(a+i),z));for(;i<n;i++)o[i]=a[i]>0?a[i]:0;}static void native_zero(const float*a,float*o,uint64_t n){(void)a;uint64_t i=0;float32x4_t z=vdupq_n_f32(0);for(;i+4<=n;i+=4)vst1q_f32(o+i,z);for(;i<n;i++)o[i]=0;}static void native_bias(const float*a,const float*b,float*o,uint64_t r,uint64_t c){for(uint64_t y=0;y<r;y++){uint64_t i=0;for(;i+4<=c;i+=4)vst1q_f32(o+y*c+i,vaddq_f32(vld1q_f32(a+y*c+i),vld1q_f32(b+i)));for(;i<c;i++)o[y*c+i]=a[y*c+i]+b[i];}}static void native_sgd(float*p,const float*g,float lr,uint64_t n){uint64_t i=0;float32x4_t l=vdupq_n_f32(lr);for(;i+4<=n;i+=4)vst1q_f32(p+i,vfmsq_f32(vld1q_f32(p+i),vld1q_f32(g+i),l));for(;i<n;i++)p[i]-=lr*g[i];}
#define NATIVE_NAME "neon"
static int native_available(void){return 1;}
#elif defined(__x86_64__)
__attribute__((target("avx2,fma")))static void native_matmul(const float*a,const float*b,float*o,uint64_t m,uint64_t k,uint64_t n){memset(o,0,m*n*4);for(uint64_t i=0;i<m;i++)for(uint64_t p=0;p<k;p++){__m256 x=_mm256_set1_ps(a[i*k+p]);uint64_t j=0;for(;j+8<=n;j+=8)_mm256_storeu_ps(o+i*n+j,_mm256_fmadd_ps(x,_mm256_loadu_ps(b+p*n+j),_mm256_loadu_ps(o+i*n+j)));for(;j<n;j++)o[i*n+j]+=a[i*k+p]*b[p*n+j];}}
__attribute__((target("avx2")))static void native_residual(const float*a,const float*b,float*o,uint64_t n){uint64_t i=0;for(;i+8<=n;i+=8)_mm256_storeu_ps(o+i,_mm256_add_ps(_mm256_loadu_ps(a+i),_mm256_loadu_ps(b+i)));for(;i<n;i++)o[i]=a[i]+b[i];}__attribute__((target("avx2")))static void native_relu(const float*a,float*o,uint64_t n){uint64_t i=0;__m256 z=_mm256_setzero_ps();for(;i+8<=n;i+=8)_mm256_storeu_ps(o+i,_mm256_max_ps(_mm256_loadu_ps(a+i),z));for(;i<n;i++)o[i]=a[i]>0?a[i]:0;}__attribute__((target("avx2")))static void native_zero(const float*a,float*o,uint64_t n){(void)a;uint64_t i=0;__m256 z=_mm256_setzero_ps();for(;i+8<=n;i+=8)_mm256_storeu_ps(o+i,z);for(;i<n;i++)o[i]=0;}__attribute__((target("avx2")))static void native_bias(const float*a,const float*b,float*o,uint64_t r,uint64_t c){for(uint64_t y=0;y<r;y++){uint64_t i=0;for(;i+8<=c;i+=8)_mm256_storeu_ps(o+y*c+i,_mm256_add_ps(_mm256_loadu_ps(a+y*c+i),_mm256_loadu_ps(b+i)));for(;i<c;i++)o[y*c+i]=a[y*c+i]+b[i];}}__attribute__((target("avx2,fma")))static void native_sgd(float*p,const float*g,float lr,uint64_t n){uint64_t i=0;__m256 l=_mm256_set1_ps(lr);for(;i+8<=n;i+=8)_mm256_storeu_ps(p+i,_mm256_fnmadd_ps(l,_mm256_loadu_ps(g+i),_mm256_loadu_ps(p+i)));for(;i<n;i++)p[i]-=lr*g[i];}
#define NATIVE_NAME "avx2"
static int native_available(void){return __builtin_cpu_supports("avx2")&&__builtin_cpu_supports("fma");}
#else
#define NATIVE_NAME "none"
static int native_available(void){return 0;}
#endif
static Dispatch scalar_dispatch(void){return(Dispatch){"scalar",scalar_matmul,scalar_residual,scalar_relu,scalar_zero,scalar_bias,scalar_sgd};}static Dispatch auto_dispatch(void){Dispatch d=scalar_dispatch();if(native_available())d=(Dispatch){NATIVE_NAME,native_matmul,native_residual,native_relu,native_zero,native_bias,native_sgd};return d;}
static float v(uint64_t i,uint64_t s){return(float)((int)((i*41+s*13)%37)-18)/29.0f;}static int eq(const float*a,const float*b,uint64_t n){for(uint64_t i=0;i<n;i++)if(fabsf(a[i]-b[i])>3e-5f*fmaxf(1,fmaxf(fabsf(a[i]),fabsf(b[i]))))return 0;return 1;}
static int check(Dispatch d){uint64_t tails[]={1,7,8,9,15,16,17};Dispatch s=scalar_dispatch();for(unsigned z=0;z<7;z++){uint64_t n=tails[z],rows=3,total=rows*n;float*a=malloc(total*4),*b=malloc(total*4),*bm=malloc(n*n*4),*x=malloc(total*4),*y=malloc(total*4),*bias=malloc(n*4),*p=malloc(total*4),*q=malloc(total*4);for(uint64_t i=0;i<total;i++)a[i]=v(i,1),b[i]=v(i,2),p[i]=q[i]=v(i,3);for(uint64_t i=0;i<n*n;i++)bm[i]=v(i,5);for(uint64_t i=0;i<n;i++)bias[i]=v(i,4);s.residual(a,b,x,total);d.residual(a,b,y,total);if(!eq(x,y,total))return 1;s.relu(a,x,total);d.relu(a,y,total);if(!eq(x,y,total))return 1;s.bias(a,bias,x,rows,n);d.bias(a,bias,y,rows,n);if(!eq(x,y,total))return 1;memset(x,1,total*4);memset(y,1,total*4);s.zero(0,x,total);d.zero(0,y,total);if(!eq(x,y,total))return 1;s.sgd(p,b,.1f,total);d.sgd(q,b,.1f,total);if(!eq(p,q,total))return 1;uint64_t m=3,k=n;s.matmul(a,bm,x,m,k,n);d.matmul(a,bm,y,m,k,n);if(!eq(x,y,m*n))return 1;free(a);free(b);free(bm);free(x);free(y);free(bias);free(p);free(q);}return 0;}
#ifndef TENSOR_DISPATCH_NO_MAIN
int main(void){Dispatch forced=scalar_dispatch(),automatic=auto_dispatch();if(check(forced)||check(automatic))return 1;printf("tensor kernel dispatch passed: compiled=%s auto=%s forced=scalar families=matmul,residual,bias,relu,zero,sgd tails=1/7/8/9 native_available=%d\n",NATIVE_NAME,automatic.backend,native_available());return 0;}
#endif
