#include <Accelerate/Accelerate.h>
#include <stddef.h>
#include <string.h>

void gpt2_matmul_bias_accelerate(const float *a, int rows, int inner,
        const float *weights, const float *bias, int columns, float *out) {
    for (int row = 0; row < rows; row++)
        memcpy(out + row * columns, bias, (size_t)columns * sizeof(float));
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                rows, columns, inner, 1.0f, a, inner, weights, columns,
                1.0f, out, columns);
}
