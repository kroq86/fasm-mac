#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

typedef struct { float *data, *grad; uint64_t rows, cols; } Tensor;
extern uint32_t tensor_cpu_features_f32(void);
extern int tensor_matmul_forward_f32(const Tensor *, const Tensor *, Tensor *);
extern int tensor_matmul_forward_auto_f32(const Tensor *, const Tensor *, Tensor *);
extern int tensor_matmul_forward_avx2_fma_f32(const Tensor *, const Tensor *, Tensor *);
extern int tensor_matmul_backward_f32(Tensor *, Tensor *, const Tensor *);
extern int tensor_matmul_backward_auto_f32(Tensor *, Tensor *, const Tensor *);
extern int tensor_matmul_backward_avx2_fma_f32(Tensor *, Tensor *, const Tensor *);

static int check(uint64_t m, uint64_t k, uint64_t n, int have_simd) {
    float *ad=calloc(m*k,4), *bd=calloc(k*n,4), *scalar=calloc(m*n,4);
    float *automatic=calloc(m*n,4), *explicit_simd=calloc(m*n,4);
    float *dout=calloc(m*n,4), *sag=calloc(m*k,4), *sbg=calloc(k*n,4);
    float *uag=calloc(m*k,4), *ubg=calloc(k*n,4), *vag=calloc(m*k,4), *vbg=calloc(k*n,4);
    if (!ad || !bd || !scalar || !automatic || !explicit_simd || !dout ||
        !sag || !sbg || !uag || !ubg || !vag || !vbg) return 1;
    for (uint64_t i=0;i<m*k;i++) ad[i]=((int)(i%11)-5)*0.071f;
    for (uint64_t i=0;i<k*n;i++) bd[i]=((int)(i%13)-6)*0.053f;
    for (uint64_t i=0;i<m*n;i++) dout[i]=((int)(i%7)-3)*0.037f;
    for (uint64_t i=0;i<m*k;i++) sag[i]=uag[i]=vag[i]=0.011f;
    for (uint64_t i=0;i<k*n;i++) sbg[i]=ubg[i]=vbg[i]=-0.013f;
    Tensor a={ad,0,m,k}, b={bd,0,k,n}, s={scalar,0,m,n}, u={automatic,0,m,n};
    Tensor v={explicit_simd,0,m,n};
    if (tensor_matmul_forward_f32(&a,&b,&s) || tensor_matmul_forward_auto_f32(&a,&b,&u)) return 1;
    if (have_simd && tensor_matmul_forward_avx2_fma_f32(&a,&b,&v)) return 1;
    for (uint64_t i=0;i<m*n;i++) {
        if (fabsf(scalar[i]-automatic[i]) > 2e-6f) {
            fprintf(stderr,"auto mismatch %llux%llux%llu i=%llu scalar=%g auto=%g\n",m,k,n,i,scalar[i],automatic[i]);
            return 1;
        }
        if (have_simd && fabsf(scalar[i]-explicit_simd[i]) > 2e-6f) {
            fprintf(stderr,"simd mismatch %llux%llux%llu i=%llu scalar=%g simd=%g\n",m,k,n,i,scalar[i],explicit_simd[i]);
            return 1;
        }
    }
    Tensor sa={ad,sag,m,k}, sb={bd,sbg,k,n}, so={scalar,dout,m,n};
    Tensor ua={ad,uag,m,k}, ub={bd,ubg,k,n}, uo={automatic,dout,m,n};
    Tensor va={ad,vag,m,k}, vb={bd,vbg,k,n}, vo={explicit_simd,dout,m,n};
    if (tensor_matmul_backward_f32(&sa,&sb,&so) || tensor_matmul_backward_auto_f32(&ua,&ub,&uo)) return 1;
    if (have_simd && tensor_matmul_backward_avx2_fma_f32(&va,&vb,&vo)) return 1;
    for (uint64_t i=0;i<m*k;i++) {
        if (fabsf(sag[i]-uag[i])>2e-6f || (have_simd && fabsf(sag[i]-vag[i])>2e-6f)) {
            fprintf(stderr,"dA SIMD mismatch %llux%llux%llu i=%llu scalar=%g auto=%g simd=%g\n",m,k,n,i,sag[i],uag[i],vag[i]);
            return 1;
        }
    }
    for (uint64_t i=0;i<k*n;i++) {
        if (fabsf(sbg[i]-ubg[i])>2e-6f || (have_simd && fabsf(sbg[i]-vbg[i])>2e-6f)) {
            fprintf(stderr,"dB SIMD mismatch %llux%llux%llu i=%llu scalar=%g auto=%g simd=%g\n",m,k,n,i,sbg[i],ubg[i],vbg[i]);
            return 1;
        }
    }
    free(ad);free(bd);free(scalar);free(automatic);free(explicit_simd);free(dout);
    free(sag);free(sbg);free(uag);free(ubg);free(vag);free(vbg);
    return 0;
}

int main(void) {
    int have_simd=(tensor_cpu_features_f32()&1)!=0;
    const uint64_t cases[][3]={{1,1,1},{2,7,7},{2,8,8},{2,9,9},{3,9,7},{3,7,9}};
    for (unsigned i=0;i<sizeof(cases)/sizeof(cases[0]);i++)
        if (check(cases[i][0],cases[i][1],cases[i][2],have_simd)) return 1;
    printf("tensor SIMD spike passed: forward+backward dispatch=%s tails=1/7/8/9\n",have_simd?"avx2+fma":"scalar");
    return 0;
}
