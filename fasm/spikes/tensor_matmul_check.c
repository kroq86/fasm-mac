#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    float *data;
    float *grad;
    uint64_t rows;
    uint64_t cols;
} Tensor;

extern int tensor_matmul_forward_f32(const Tensor *, const Tensor *, Tensor *);
extern int tensor_matmul_backward_f32(Tensor *, Tensor *, const Tensor *);
extern int tensor_matmul_backward_lhs_f32(Tensor *, Tensor *, const Tensor *);
extern int tensor_matmul_backward_rhs_f32(Tensor *, Tensor *, const Tensor *);
extern int tensor_matmul_backward_both_f32(Tensor *, Tensor *, const Tensor *);

static float loss(const Tensor *out, const float *seed) {
    float sum = 0.0f;
    for (uint64_t i = 0; i < out->rows * out->cols; ++i) sum += out->data[i] * seed[i];
    return sum;
}

static int check_case(uint64_t m, uint64_t k, uint64_t n) {
    const uint64_t ac = m * k, bc = k * n, oc = m * n;
    float *ad = calloc(ac, sizeof(float)), *ag = calloc(ac, sizeof(float));
    float *bd = calloc(bc, sizeof(float)), *bg = calloc(bc, sizeof(float));
    float *od = calloc(oc, sizeof(float)), *og = calloc(oc, sizeof(float));
    if (!ad || !ag || !bd || !bg || !od || !og) return 1;
    for (uint64_t i = 0; i < ac; ++i) ad[i] = ((int)(i % 7) - 3) * 0.17f;
    for (uint64_t i = 0; i < bc; ++i) bd[i] = ((int)(i % 5) - 2) * 0.13f;
    for (uint64_t i = 0; i < oc; ++i) og[i] = ((int)(i % 3) - 1) * 0.19f;
    Tensor a = {ad, ag, m, k}, b = {bd, bg, k, n}, out = {od, og, m, n};
    if (tensor_matmul_forward_f32(&a, &b, &out) || tensor_matmul_backward_f32(&a, &b, &out)) return 1;

    const float eps = 0.001f, tolerance = 0.00012f;
    for (uint64_t i = 0; i < ac; ++i) {
        float saved = ad[i];
        ad[i] = saved + eps; tensor_matmul_forward_f32(&a, &b, &out); float plus = loss(&out, og);
        ad[i] = saved - eps; tensor_matmul_forward_f32(&a, &b, &out); float minus = loss(&out, og);
        ad[i] = saved;
        float numeric = (plus - minus) / (2.0f * eps);
        if (fabsf(numeric - ag[i]) > tolerance) {
            fprintf(stderr, "dA mismatch %llux%llux%llu i=%llu analytic=%g numeric=%g\n",
                    m, k, n, i, ag[i], numeric);
            return 1;
        }
    }
    for (uint64_t i = 0; i < bc; ++i) {
        float saved = bd[i];
        bd[i] = saved + eps; tensor_matmul_forward_f32(&a, &b, &out); float plus = loss(&out, og);
        bd[i] = saved - eps; tensor_matmul_forward_f32(&a, &b, &out); float minus = loss(&out, og);
        bd[i] = saved;
        float numeric = (plus - minus) / (2.0f * eps);
        if (fabsf(numeric - bg[i]) > tolerance) {
            fprintf(stderr, "dB mismatch %llux%llux%llu i=%llu analytic=%g numeric=%g\n",
                    m, k, n, i, bg[i], numeric);
            return 1;
        }
    }
    free(ad); free(ag); free(bd); free(bg); free(od); free(og);
    return 0;
}

static int check_accumulation(void) {
    float ad = 2.0f, ag = 0.0f, bd = 3.0f, bg = 0.0f, od = 0.0f, og = 4.0f;
    Tensor a = {&ad, &ag, 1, 1}, b = {&bd, &bg, 1, 1}, out = {&od, &og, 1, 1};
    if (tensor_matmul_forward_f32(&a, &b, &out)) return 1;
    if (tensor_matmul_backward_f32(&a, &b, &out)) return 1;
    if (tensor_matmul_backward_f32(&a, &b, &out)) return 1;
    if (ag != 24.0f || bg != 16.0f) {
        fprintf(stderr, "gradient accumulation mismatch: dA=%g dB=%g\n", ag, bg);
        return 1;
    }
    return 0;
}

static int check_shared_operand(void) {
    float xd = 2.0f, xg = 0.0f, od = 0.0f, og = 1.0f;
    Tensor x = {&xd, &xg, 1, 1}, out = {&od, &og, 1, 1};
    if (tensor_matmul_forward_f32(&x, &x, &out)) return 1;
    if (tensor_matmul_backward_f32(&x, &x, &out)) return 1;
    if (xg != 4.0f) {
        fprintf(stderr, "shared-operand gradient mismatch: got=%g want=4\n", xg);
        return 1;
    }
    return 0;
}

static int check_selective(void) {
    float ad[6]={1,2,3,4,5,6},bd[6]={.5f,-1,2,.25f,-.5f,3};
    float full_ag[6]={0},full_bg[6]={0},lhs_ag[6]={0},rhs_bg[6]={0};
    float od[4]={0},og[4]={1,-2,.5f,3};
    Tensor af={ad,full_ag,2,3},bf={bd,full_bg,3,2},out={od,og,2,2};
    if(tensor_matmul_forward_f32(&af,&bf,&out)||tensor_matmul_backward_both_f32(&af,&bf,&out))return 1;
    Tensor al={ad,lhs_ag,2,3},b_no_grad={bd,0,3,2};
    if(tensor_matmul_backward_lhs_f32(&al,&b_no_grad,&out))return 1;
    Tensor a_no_grad={ad,0,2,3},br={bd,rhs_bg,3,2};
    if(tensor_matmul_backward_rhs_f32(&a_no_grad,&br,&out))return 1;
    for(int i=0;i<6;i++)if(fabsf(lhs_ag[i]-full_ag[i])>1e-6f||fabsf(rhs_bg[i]-full_bg[i])>1e-6f)return 1;
    if(tensor_matmul_backward_lhs_f32(&a_no_grad,&br,&out)!=-1)return 1;
    if(tensor_matmul_backward_rhs_f32(&al,&b_no_grad,&out)!=-1)return 1;
    return 0;
}

int main(void) {
    const uint64_t cases[][3] = {{1,1,1}, {2,7,3}, {2,8,3}, {2,9,3}, {3,2,4}};
    for (size_t i = 0; i < sizeof(cases)/sizeof(cases[0]); ++i)
        if (check_case(cases[i][0], cases[i][1], cases[i][2])) return 1;
    if (check_accumulation() || check_shared_operand() || check_selective()) return 1;

    float x = 1.0f, g = 0.0f;
    Tensor a = {&x, &g, 1, 1}, b = {&x, &g, 2, 1}, out = {&x, &g, 1, 1};
    if (tensor_matmul_forward_f32(&a, &b, &out) != -1) {
        fputs("shape mismatch was accepted\n", stderr);
        return 1;
    }
    puts("tensor matmul spike passed: gradients/tails/shapes/accumulation/sharing/selective-lhs-rhs-both");
    return 0;
}
