#include <arm_neon.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mach/mach_time.h>
#ifndef AUTO_VECTORIZE
#define AUTO_VECTORIZE 1
#endif

static void matmul_scalar(const float*a,const float*b,float*out,uint64_t m,uint64_t k,uint64_t n){memset(out,0,m*n*4);for(uint64_t i=0;i<m;i++)for(uint64_t p=0;p<k;p++){float x=a[i*k+p];for(uint64_t j=0;j<n;j++)out[i*n+j]+=x*b[p*n+j];}}
static void matmul_neon(const float*a,const float*b,float*out,uint64_t m,uint64_t k,uint64_t n){memset(out,0,m*n*4);for(uint64_t i=0;i<m;i++)for(uint64_t p=0;p<k;p++){float32x4_t x=vdupq_n_f32(a[i*k+p]);uint64_t j=0;for(;j+4<=n;j+=4){float32x4_t y=vld1q_f32(b+p*n+j),z=vld1q_f32(out+i*n+j);vst1q_f32(out+i*n+j,vfmaq_f32(z,x,y));}for(;j<n;j++)out[i*n+j]+=a[i*k+p]*b[p*n+j];}}
static void residual_relu_scalar(const float*a,const float*b,float*out,uint64_t n){for(uint64_t i=0;i<n;i++){float x=a[i]+b[i];out[i]=x>0?x:0;}}
static void residual_relu_neon(const float*a,const float*b,float*out,uint64_t n){uint64_t i=0;float32x4_t zero=vdupq_n_f32(0);for(;i+4<=n;i+=4)vst1q_f32(out+i,vmaxq_f32(vaddq_f32(vld1q_f32(a+i),vld1q_f32(b+i)),zero));for(;i<n;i++){float x=a[i]+b[i];out[i]=x>0?x:0;}}
static float value(uint64_t i,uint64_t salt){return(float)((int)((i*43+salt*17)%37)-18)/23.0f;}
static int closef(float a,float b){return fabsf(a-b)<=3e-5f*fmaxf(1,fmaxf(fabsf(a),fabsf(b)));}
static int correctness(void){uint64_t dims[]={1,7,8,9,15,16,17};for(unsigned c=0;c<49;c++){uint64_t m=dims[c%7],k=dims[(c*3+1)%7],n=dims[(c*5+2)%7],ac=m*k,bc=k*n,oc=m*n;float*a=malloc(ac*4),*b=malloc(bc*4),*s=malloc(oc*4),*v=malloc(oc*4);for(uint64_t i=0;i<ac;i++)a[i]=value(i,1);for(uint64_t i=0;i<bc;i++)b[i]=value(i,2);matmul_scalar(a,b,s,m,k,n);matmul_neon(a,b,v,m,k,n);for(uint64_t i=0;i<oc;i++)if(!closef(s[i],v[i]))return 1;residual_relu_scalar(s,v,s,oc);residual_relu_neon(v,v,v,oc);for(uint64_t i=0;i<oc;i++)if(!closef(s[i],v[i]))return 1;free(a);free(b);free(s);free(v);}return 0;}
static double ms(uint64_t ticks){mach_timebase_info_data_t t;mach_timebase_info(&t);return(double)ticks*t.numer/t.denom/1e6;}
int main(void){if(correctness())return 1;const uint64_t m=128,k=128,n=128,reps=30;float*a=malloc(m*k*4),*b=malloc(k*n*4),*o=malloc(m*n*4);for(uint64_t i=0;i<m*k;i++)a[i]=value(i,3);for(uint64_t i=0;i<k*n;i++)b[i]=value(i,4);uint64_t t0=mach_absolute_time();for(unsigned r=0;r<reps;r++)matmul_scalar(a,b,o,m,k,n);uint64_t t1=mach_absolute_time();volatile float sink=o[17];uint64_t t2=mach_absolute_time();for(unsigned r=0;r<reps;r++)matmul_neon(a,b,o,m,k,n);uint64_t t3=mach_absolute_time();sink+=o[19];double scalar=ms(t1-t0),neon=ms(t3-t2);printf("tensor NEON spike passed: host=arm64 lanes_f32=4 compiler_vectorize=%s cases=49 tails=1/7/8/9 matmul_128_reps=%llu scalar_ms=%.3f neon_ms=%.3f speedup=%.2fx residual_relu=neon checksum=%.6f\n",AUTO_VECTORIZE?"on":"off",reps,scalar,neon,scalar/neon,(double)sink);free(a);free(b);free(o);return 0;}
