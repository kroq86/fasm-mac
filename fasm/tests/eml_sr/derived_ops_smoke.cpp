#include "../../apps/eml_sr/derived_ops.hpp"

#include <cmath>
#include <complex>
#include <cstdio>

namespace {

constexpr double kEulerGamma = 0.5772156649015329;
constexpr double kGlaisher = 1.2824271291006226;
constexpr double kCatalan = 0.9159655941772190;
constexpr double kKhinchin = 2.6854520010653064;

bool near_complex(std::complex<double> a, std::complex<double> b, double eps = 1e-9) {
    return std::abs(a - b) <= eps * (1.0 + std::abs(a) + std::abs(b));
}

bool check(const char* name, std::optional<std::complex<double>> got, std::complex<double> want) {
    if (!got || !near_complex(*got, want)) {
        const std::complex<double> g = got.value_or(std::complex<double>{NAN, NAN});
        std::fprintf(
            stderr,
            "%s failed: got=(%.17g,%.17g) want=(%.17g,%.17g)\n",
            name,
            g.real(),
            g.imag(),
            want.real(),
            want.imag());
        return false;
    }
    return true;
}

bool check_pair(double x, double y) {
    using C = std::complex<double>;
    const C a{x, 0.0};
    const C b{y, 0.0};
    return check("Exp", eml_sr::d_exp(a), std::exp(a)) &&
           check("Log", eml_sr::d_log(a), std::log(a)) &&
           check("Subtract", eml_sr::d_sub(a, b), a - b) &&
           check("Minus", eml_sr::d_neg(a), -a) &&
           check("Plus", eml_sr::d_add(a, b), a + b) &&
           check("Inv", eml_sr::d_inv(a), C{1.0, 0.0} / a) &&
           check("Times", eml_sr::d_mul(a, b), a * b) &&
           check("Divide", eml_sr::d_div(a, b), a / b) &&
           check("Half", eml_sr::d_half(a), a / C{2.0, 0.0}) &&
           check("Avg", eml_sr::d_avg(a, b), (a + b) / C{2.0, 0.0}) &&
           check("Sqr", eml_sr::d_sqr(a), a * a) &&
           check("Sqrt", eml_sr::d_sqrt(a), std::sqrt(a)) &&
           check("Power", eml_sr::d_pow(a, b), std::pow(a, b)) &&
           check("LogBase", eml_sr::d_log_base(a, b), std::log(b) / std::log(a));
}

}  // namespace

int main() {
    const double pairs[][2] = {
        {kEulerGamma, kGlaisher},
        {-kEulerGamma, kGlaisher},
        {kCatalan, kKhinchin},
        {-kCatalan, kKhinchin},
        {kGlaisher, kEulerGamma},
        {-kGlaisher, kEulerGamma},
        {kKhinchin, kCatalan},
        {-kKhinchin, kCatalan},
    };
    for (const auto& pair : pairs) {
        if (!check_pair(pair[0], pair[1])) {
            return 1;
        }
    }
    std::puts("OK eml_sr derived ops");
    return 0;
}
