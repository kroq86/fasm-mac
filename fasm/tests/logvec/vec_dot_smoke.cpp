#include "../../apps/logvec/vector_core.hpp"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

using logvec::lb_vec_dot_f32;
using logvec::lb_vec_has_avx2;
using logvec::lb_vec_norm_f32;

namespace {

float ref_dot(const float* a, const float* b, std::size_t len) {
    float sum = 0.0f;
    for (std::size_t i = 0; i < len; ++i) {
        sum += a[i] * b[i];
    }
    return sum;
}

float ref_norm(const float* v, std::size_t len) {
    return std::sqrt(ref_dot(v, v, len));
}

bool approx_eq(float a, float b, float tol) {
    return std::fabs(a - b) <= tol;
}

bool check_dim(std::uint32_t dim) {
    std::vector<float> a(dim);
    std::vector<float> b(dim);
    for (std::uint32_t i = 0; i < dim; ++i) {
        a[i] = static_cast<float>(i + 1) * 0.01f;
        b[i] = static_cast<float>(dim - i) * 0.02f;
    }
    const float dot = lb_vec_dot_f32(a.data(), b.data(), dim);
    const float want_dot = ref_dot(a.data(), b.data(), dim);
    if (!approx_eq(dot, want_dot, 1e-4f * std::max(1.0f, std::fabs(want_dot)))) {
        std::fprintf(stderr, "dot mismatch dim=%u got=%f want=%f\n", dim, dot, want_dot);
        return false;
    }
    const float n = lb_vec_norm_f32(a.data(), dim);
    const float want_n = ref_norm(a.data(), dim);
    if (!approx_eq(n, want_n, 1e-4f * std::max(1.0f, want_n))) {
        std::fprintf(stderr, "norm mismatch dim=%u got=%f want=%f\n", dim, n, want_n);
        return false;
    }
    return true;
}

}  // namespace

int main() {
    const std::uint32_t dims[] = {4, 7, 768};
    for (const std::uint32_t dim : dims) {
        if (!check_dim(dim)) {
            return 1;
        }
    }
    std::printf("OK vec_dot_smoke avx2=%d\n", lb_vec_has_avx2());
    return 0;
}
