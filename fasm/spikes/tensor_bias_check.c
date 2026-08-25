#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

typedef struct { float *data,*grad; uint64_t rows,cols; } Tensor;
extern int tensor_bias_add_forward_f32(const Tensor *,const Tensor *,Tensor *);
extern int tensor_bias_add_backward_f32(Tensor *,Tensor *,const Tensor *);

static float loss(const Tensor *out,const float *seed) {
    float s=0; for(uint64_t i=0;i<out->rows*out->cols;i++) s+=out->data[i]*seed[i]; return s;
}
static int check(uint64_t m,uint64_t n) {
    float *xd=calloc(m*n,4),*xg=calloc(m*n,4),*bd=calloc(n,4),*bg=calloc(n,4);
    float *od=calloc(m*n,4),*og=calloc(m*n,4);
    if(!xd||!xg||!bd||!bg||!od||!og)return 1;
    for(uint64_t i=0;i<m*n;i++){xd[i]=((int)(i%7)-3)*.11f;og[i]=((int)(i%5)-2)*.07f;xg[i]=.01f;}
    for(uint64_t i=0;i<n;i++){bd[i]=((int)(i%3)-1)*.13f;bg[i]=-.02f;}
    Tensor x={xd,xg,m,n},b={bd,bg,1,n},out={od,og,m,n};
    if(tensor_bias_add_forward_f32(&x,&b,&out)||tensor_bias_add_backward_f32(&x,&b,&out))return 1;
    const float eps=.001f,tol=.00008f;
    for(uint64_t i=0;i<m*n;i++){
        float v=xd[i];xd[i]=v+eps;tensor_bias_add_forward_f32(&x,&b,&out);float p=loss(&out,og);
        xd[i]=v-eps;tensor_bias_add_forward_f32(&x,&b,&out);float q=loss(&out,og);xd[i]=v;
        float num=(p-q)/(2*eps);if(fabsf((xg[i]-.01f)-num)>tol)return 1;
    }
    for(uint64_t i=0;i<n;i++){
        float v=bd[i];bd[i]=v+eps;tensor_bias_add_forward_f32(&x,&b,&out);float p=loss(&out,og);
        bd[i]=v-eps;tensor_bias_add_forward_f32(&x,&b,&out);float q=loss(&out,og);bd[i]=v;
        float num=(p-q)/(2*eps);if(fabsf((bg[i]+.02f)-num)>tol)return 1;
    }
    free(xd);free(xg);free(bd);free(bg);free(od);free(og);return 0;
}
int main(void){
    if(check(1,1)||check(2,7)||check(2,8)||check(2,9))return 1;
    float x=0,g=0;Tensor a={&x,&g,2,2},bad={&x,&g,2,1},out={&x,&g,2,2};
    if(tensor_bias_add_forward_f32(&a,&bad,&out)!=-1)return 1;
    puts("tensor bias spike passed: forward/backward/finite-difference/1-7-8-9/shapes");return 0;
}
