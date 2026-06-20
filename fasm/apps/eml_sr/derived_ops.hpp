#pragma once

#include "eml.hpp"

#include <complex>
#include <optional>

namespace eml_sr {

inline bool finite_complex(std::complex<double> z) {
    return std::isfinite(z.real()) && std::isfinite(z.imag());
}

inline std::optional<std::complex<double>> checked(std::complex<double> z) {
    if (!finite_complex(z)) {
        return std::nullopt;
    }
    return z;
}

namespace detail {

inline std::complex<double> d_exp_raw(std::complex<double> x) {
    return eml(x, {1.0, 0.0});
}

inline std::complex<double> d_log_raw(std::complex<double> x) {
    return eml({1.0, 0.0}, d_exp_raw(eml({1.0, 0.0}, x)));
}

inline std::complex<double> d_sub_raw(std::complex<double> a, std::complex<double> b) {
    return eml(d_log_raw(a), d_exp_raw(b));
}

inline std::complex<double> d_neg_raw(std::complex<double> x) {
    return d_sub_raw(d_log_raw({1.0, 0.0}), x);
}

inline std::complex<double> d_add_raw(std::complex<double> a, std::complex<double> b) {
    return d_sub_raw(a, d_neg_raw(b));
}

inline std::complex<double> d_inv_raw(std::complex<double> x) {
    return d_exp_raw(d_neg_raw(d_log_raw(x)));
}

inline std::complex<double> d_mul_raw(std::complex<double> a, std::complex<double> b) {
    return d_exp_raw(d_add_raw(d_log_raw(a), d_log_raw(b)));
}

inline std::complex<double> d_half_raw(std::complex<double> x) {
    return d_mul_raw(x, d_inv_raw({2.0, 0.0}));
}

inline std::complex<double> d_sqr_raw(std::complex<double> x) {
    return d_mul_raw(x, x);
}

inline std::complex<double> d_sqrt_raw(std::complex<double> x) {
    return d_exp_raw(d_half_raw(d_log_raw(x)));
}

inline std::complex<double> d_pow_raw(std::complex<double> a, std::complex<double> b) {
    return d_exp_raw(d_mul_raw(b, d_log_raw(a)));
}

}  // namespace detail

inline std::optional<std::complex<double>> d_exp(std::complex<double> x) {
    return checked(detail::d_exp_raw(x));
}

inline std::optional<std::complex<double>> d_log(std::complex<double> x) {
    return checked(detail::d_log_raw(x));
}

inline std::optional<std::complex<double>> d_sub(std::complex<double> a, std::complex<double> b) {
    return checked(detail::d_sub_raw(a, b));
}

inline std::optional<std::complex<double>> d_neg(std::complex<double> x) {
    return checked(detail::d_neg_raw(x));
}

inline std::optional<std::complex<double>> d_add(std::complex<double> a, std::complex<double> b) {
    return checked(detail::d_add_raw(a, b));
}

inline std::optional<std::complex<double>> d_inv(std::complex<double> x) {
    return checked(detail::d_inv_raw(x));
}

inline std::optional<std::complex<double>> d_mul(std::complex<double> a, std::complex<double> b) {
    return checked(detail::d_mul_raw(a, b));
}

inline std::optional<std::complex<double>> d_div(std::complex<double> a, std::complex<double> b) {
    return d_mul(a, detail::d_inv_raw(b));
}

inline std::optional<std::complex<double>> d_half(std::complex<double> x) {
    return checked(detail::d_half_raw(x));
}

inline std::optional<std::complex<double>> d_avg(std::complex<double> a, std::complex<double> b) {
    return checked(detail::d_half_raw(detail::d_add_raw(a, b)));
}

inline std::optional<std::complex<double>> d_sqr(std::complex<double> x) {
    return checked(detail::d_sqr_raw(x));
}

inline std::optional<std::complex<double>> d_sqrt(std::complex<double> x) {
    return checked(detail::d_sqrt_raw(x));
}

inline std::optional<std::complex<double>> d_pow(std::complex<double> a, std::complex<double> b) {
    return checked(detail::d_pow_raw(a, b));
}

inline std::optional<std::complex<double>> d_log_base(std::complex<double> base, std::complex<double> x) {
    return d_div(detail::d_log_raw(x), detail::d_log_raw(base));
}

}  // namespace eml_sr
