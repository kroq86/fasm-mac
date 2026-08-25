#include "../../apps/eml_core.hpp"
#include "../../apps/eml_sr/eml.hpp"

#include <cmath>
#include <cstdio>
#include <cstdlib>

extern "C" double lb_eml_f64(double x, double y);

namespace {

bool check_pair(double x, double y) {
    const double got = eml_core::lb_eml_f64(x, y);
    const double want = std::real(eml_sr::eml({x, 0.0}, {y, 0.0}));
    if (!eml_sr::near(got, want, 1e-9)) {
        std::fprintf(stderr, "eml_core mismatch x=%g y=%g got=%g want=%g\n", x, y, got, want);
        return false;
    }
    return true;
}

bool check_unary(const char* name, double got, double want, double tol = 1e-9) {
    if (!eml_sr::near(got, want, tol)) {
        std::fprintf(stderr, "%s mismatch got=%g want=%g\n", name, got, want);
        return false;
    }
    return true;
}

bool check_binary(const char* name, double got, double want, double tol = 1e-9) {
    return check_unary(name, got, want, tol);
}

}  // namespace

int main() {
    const double xs[] = {0.1, 0.5, 1.0, 2.0};
    for (const double x : xs) {
        if (!check_pair(x, 1.0)) {
            return 1;
        }
        const double exp_x = std::exp(x);
        if (!eml_sr::near(eml_core::lb_eml_f64(x, 1.0), exp_x, 1e-9)) {
            std::fprintf(stderr, "eml_core exp identity failed x=%g\n", x);
            return 1;
        }
    }
    if (!check_pair(1.0, std::exp(1.0))) {
        return 1;
    }
    const double positives[] = {0.25, 0.5, 1.0, 2.0, std::exp(1.0)};
    for (const double x : positives) {
        if (!check_unary("lb_eml_exp_f64", eml_core::lb_eml_exp_f64(x), std::exp(x))) {
            return 1;
        }
        if (!check_unary("lb_eml_log_f64", eml_core::lb_eml_log_f64(x), std::log(x))) {
            return 1;
        }
        if (!check_unary("lb_eml_neg_f64", eml_core::lb_eml_neg_f64(x), -x)) {
            return 1;
        }
        if (!check_unary("lb_eml_inv_f64", eml_core::lb_eml_inv_f64(x), 1.0 / x)) {
            return 1;
        }
        if (!check_unary("lb_eml_half_f64", eml_core::lb_eml_half_f64(x), x / 2.0)) {
            return 1;
        }
    }
    const double pairs[][2] = {
        {0.25, 0.5},
        {0.5, 1.0},
        {1.0, 2.0},
        {2.0, std::exp(1.0)},
    };
    for (const auto& pair : pairs) {
        const double a = pair[0];
        const double b = pair[1];
        if (!check_binary("lb_eml_sub_f64", eml_core::lb_eml_sub_f64(a, b), a - b)) {
            return 1;
        }
        if (!check_binary("lb_eml_add_f64", eml_core::lb_eml_add_f64(a, b), a + b)) {
            return 1;
        }
        if (!check_binary("lb_eml_mul_f64", eml_core::lb_eml_mul_f64(a, b), a * b)) {
            return 1;
        }
        if (!check_binary("lb_eml_div_f64", eml_core::lb_eml_div_f64(a, b), a / b)) {
            return 1;
        }
    }
    std::printf("OK eml_core lb_eml_f64\n");
    return 0;
}
