#pragma once

#include "witness_verify.hpp"

#include <optional>
#include <string>

namespace eml_sr::compiler {

inline std::string eml_call(const std::string& a, const std::string& b) {
    return "EML[" + a + ", " + b + "]";
}

inline std::string c_exp(const std::string& x) {
    return eml_call(x, "1");
}

inline std::string c_log(const std::string& x) {
    return eml_call("1", c_exp(eml_call("1", x)));
}

inline std::string c_zero() {
    return c_log("1");
}

inline std::string c_sub(const std::string& a, const std::string& b) {
    return eml_call(c_log(a), c_exp(b));
}

inline std::string c_neg(const std::string& x) {
    return c_sub(c_zero(), x);
}

inline std::string c_neg_one() {
    return c_sub(c_zero(), "1");
}

inline std::string c_two() {
    return c_sub("1", c_neg_one());
}

inline std::string c_add(const std::string& a, const std::string& b) {
    return c_sub(a, c_neg(b));
}

inline std::string c_inv(const std::string& x) {
    return c_exp(c_neg(c_log(x)));
}

inline std::string c_mul(const std::string& a, const std::string& b) {
    return c_exp(c_add(c_log(a), c_log(b)));
}

inline std::string c_div(const std::string& a, const std::string& b) {
    return c_mul(a, c_inv(b));
}

inline std::string c_half(const std::string& x) {
    return c_div(x, c_two());
}

inline std::string c_avg(const std::string& a, const std::string& b) {
    return c_half(c_add(a, b));
}

inline std::string c_sqr(const std::string& x) {
    return c_mul(x, x);
}

inline std::string c_sqrt(const std::string& x) {
    return c_exp(c_half(c_log(x)));
}

inline std::string c_pow(const std::string& a, const std::string& b) {
    return c_exp(c_mul(b, c_log(a)));
}

inline std::string c_log_base(const std::string& base, const std::string& x) {
    return c_div(c_log(x), c_log(base));
}

inline std::string c_pi() {
    return c_sqrt(c_neg(c_sqr(c_log(c_neg_one()))));
}

inline std::string c_i() {
    return c_sqrt(c_neg_one());
}

inline std::string c_hypot(const std::string& a, const std::string& b) {
    return c_sqrt(c_add(c_sqr(a), c_sqr(b)));
}

inline std::string c_logistic_sigmoid(const std::string& x) {
    return c_inv(eml_call(c_neg(x), c_exp(c_neg_one())));
}

inline std::string c_cosh(const std::string& x) {
    return c_avg(c_exp(x), c_exp(c_neg(x)));
}

inline std::string c_sinh(const std::string& x) {
    return eml_call(x, c_exp(c_cosh(x)));
}

inline std::string c_tanh(const std::string& x) {
    return c_div(c_sinh(x), c_cosh(x));
}

inline std::string c_cos(const std::string& x) {
    return c_cosh(c_sqrt(c_neg(c_sqr(x))));
}

inline std::string c_sin(const std::string& x) {
    return c_cos(c_sub(x, c_half(c_pi())));
}

inline std::string c_tan(const std::string& x) {
    return c_div(c_sin(x), c_cos(x));
}

inline std::string c_arcsinh(const std::string& x) {
    return c_log(c_add(x, c_sqrt(c_add(c_sqr(x), "1"))));
}

inline std::string c_arccosh(const std::string& x) {
    return c_log(c_add(x, c_mul(c_sqrt(c_add(x, "1")), c_sqrt(c_sub(x, "1")))));
}

inline std::string c_arctanh(const std::string& x) {
    return c_half(c_sub(c_log(c_add("1", x)), c_log(c_sub("1", x))));
}

inline std::string c_arcsin(const std::string& x) {
    return c_mul(c_neg(c_i()), c_log(c_add(c_mul(c_i(), x), c_sqrt(c_sub("1", c_sqr(x))))));
}

inline std::string c_arccos(const std::string& x) {
    return c_sub(c_half(c_pi()), c_arcsin(x));
}

inline std::string c_arctan(const std::string& x) {
    return c_mul(
        c_mul(c_i(), c_half("1")),
        c_sub(c_log(c_sub("1", c_mul(c_i(), x))), c_log(c_add("1", c_mul(c_i(), x)))));
}

inline std::optional<std::string> compile_expr(const witness::Expr& expr);

inline std::optional<std::string> compile_atom(const std::string& name) {
    if (name == "0") {
        return c_zero();
    }
    if (name == "-1") {
        return c_neg_one();
    }
    if (name == "2") {
        return c_two();
    }
    if (name == "E") {
        return c_exp("1");
    }
    if (name == "Pi") {
        return c_pi();
    }
    return name;
}

inline std::optional<std::string> compile_unary(const std::string& name, const std::string& x) {
    if (name == "Exp") {
        return c_exp(x);
    }
    if (name == "Log") {
        return c_log(x);
    }
    if (name == "Minus") {
        return c_neg(x);
    }
    if (name == "Inv") {
        return c_inv(x);
    }
    if (name == "Half") {
        return c_half(x);
    }
    if (name == "Sqr") {
        return c_sqr(x);
    }
    if (name == "Sqrt") {
        return c_sqrt(x);
    }
    if (name == "LogisticSigmoid") {
        return c_logistic_sigmoid(x);
    }
    if (name == "Cosh") {
        return c_cosh(x);
    }
    if (name == "Sinh") {
        return c_sinh(x);
    }
    if (name == "Tanh") {
        return c_tanh(x);
    }
    if (name == "Cos") {
        return c_cos(x);
    }
    if (name == "Sin") {
        return c_sin(x);
    }
    if (name == "Tan") {
        return c_tan(x);
    }
    if (name == "ArcSinh") {
        return c_arcsinh(x);
    }
    if (name == "ArcCosh") {
        return c_arccosh(x);
    }
    if (name == "ArcCos") {
        return c_arccos(x);
    }
    if (name == "ArcTanh") {
        return c_arctanh(x);
    }
    if (name == "ArcSin") {
        return c_arcsin(x);
    }
    if (name == "ArcTan") {
        return c_arctan(x);
    }
    return std::nullopt;
}

inline std::optional<std::string> compile_binary(
    const std::string& name,
    const std::string& a,
    const std::string& b) {
    if (name == "EML") {
        return eml_call(a, b);
    }
    if (name == "Subtract") {
        return c_sub(a, b);
    }
    if (name == "Plus") {
        return c_add(a, b);
    }
    if (name == "Times") {
        return c_mul(a, b);
    }
    if (name == "Divide") {
        return c_div(a, b);
    }
    if (name == "Avg") {
        return c_avg(a, b);
    }
    if (name == "Power") {
        return c_pow(a, b);
    }
    if (name == "Log") {
        return c_log_base(a, b);
    }
    if (name == "Hypot") {
        return c_hypot(a, b);
    }
    return std::nullopt;
}

inline std::optional<std::string> compile_expr(const witness::Expr& expr) {
    if (expr.is_atom()) {
        return compile_atom(expr.name);
    }
    if (expr.args.size() == 1) {
        auto x = compile_expr(expr.args[0]);
        if (!x) {
            return std::nullopt;
        }
        return compile_unary(expr.name, *x);
    }
    if (expr.args.size() == 2) {
        auto a = compile_expr(expr.args[0]);
        auto b = compile_expr(expr.args[1]);
        if (!a || !b) {
            return std::nullopt;
        }
        return compile_binary(expr.name, *a, *b);
    }
    return std::nullopt;
}

inline std::optional<std::string> compile_source(const std::string& source) {
    auto expr = witness::parse(source);
    if (!expr) {
        return std::nullopt;
    }
    return compile_expr(*expr);
}

inline bool is_pure_eml_expr(const witness::Expr& expr) {
    if (expr.is_atom()) {
        return true;
    }
    if (expr.name != "EML" || expr.args.size() != 2) {
        return false;
    }
    return is_pure_eml_expr(expr.args[0]) && is_pure_eml_expr(expr.args[1]);
}

inline bool is_pure_eml_source(const std::string& source) {
    auto expr = witness::parse(source);
    return expr && is_pure_eml_expr(*expr);
}

inline std::optional<witness::Value> eval_pure_eml_raw(
    const witness::Expr& expr,
    const witness::Env& env) {
    if (expr.is_atom()) {
        if (auto found = env.find(expr.name); found != env.end()) {
            return found->second;
        }
        return witness::constant_value(expr.name);
    }
    if (expr.name != "EML" || expr.args.size() != 2) {
        return std::nullopt;
    }
    auto a = eval_pure_eml_raw(expr.args[0], env);
    auto b = eval_pure_eml_raw(expr.args[1], env);
    if (!a || !b) {
        return std::nullopt;
    }
    return eml(*a, *b);
}

inline bool verify_compiled_identity(
    const std::string& target_source,
    const std::string& compiled_source,
    const std::vector<witness::Env>& envs,
    double eps = 1e-9,
    bool skip_nonreal_target = false) {
    const auto target = witness::parse(target_source);
    const auto compiled = witness::parse(compiled_source);
    if (!target || !compiled || !is_pure_eml_expr(*compiled)) {
        return false;
    }

    std::size_t tested = 0;
    for (const witness::Env& env : envs) {
        const auto target_value = witness::eval(*target, env);
        if (!target_value) {
            continue;
        }
        if (skip_nonreal_target && !witness::imag_is_zero(*target_value, eps)) {
            continue;
        }
        const auto compiled_value = eval_pure_eml_raw(*compiled, env);
        if (!compiled_value || !witness::finite(*compiled_value) ||
            !witness::near(*compiled_value, *target_value, eps)) {
            return false;
        }
        ++tested;
    }
    return tested > 0;
}

}  // namespace eml_sr::compiler
