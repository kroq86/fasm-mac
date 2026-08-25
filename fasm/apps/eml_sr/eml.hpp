#pragma once

#include <cmath>
#include <complex>
#include <stdexcept>

namespace eml_sr {

inline std::complex<double> eml(std::complex<double> x, std::complex<double> y) {
    return std::exp(x) - std::log(y);
}

inline bool near(double a, double b, double tol = 1e-9) {
    return std::fabs(a - b) <= tol;
}

}  // namespace eml_sr
