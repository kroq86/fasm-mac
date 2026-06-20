#pragma once

#include "eml.hpp"

#include <cctype>
#include <cmath>
#include <complex>
#include <numbers>
#include <optional>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace eml_sr::witness {

constexpr double kEulerGamma = 0.5772156649015329;
constexpr double kCatalan = 0.9159655941772190;
constexpr double kGlaisher = 1.2824271291006226;
constexpr double kKhinchin = 2.6854520010653064;

struct Expr {
    std::string name;
    std::vector<Expr> args;

    [[nodiscard]] bool is_atom() const { return args.empty(); }
};

using Value = std::complex<double>;
using Env = std::unordered_map<std::string, Value>;

inline bool finite(Value v) {
    return std::isfinite(v.real()) && std::isfinite(v.imag());
}

inline std::optional<Value> checked(Value v) {
    if (!finite(v)) {
        return std::nullopt;
    }
    return v;
}

inline bool near(Value a, Value b, double eps = 1e-9) {
    return std::abs(a - b) <= eps * (1.0 + std::abs(a) + std::abs(b));
}

inline bool imag_is_zero(Value v, double eps = 1e-9) {
    return std::abs(v.imag()) <= eps * (1.0 + std::abs(v.real()) + std::abs(v.imag()));
}

class Parser {
public:
    explicit Parser(std::string source) : source_(std::move(source)) {}

    std::optional<Expr> parse() {
        auto expr = parse_expr();
        skip_ws();
        if (!expr || pos_ != source_.size()) {
            return std::nullopt;
        }
        return expr;
    }

private:
    void skip_ws() {
        while (pos_ < source_.size() &&
               std::isspace(static_cast<unsigned char>(source_[pos_]))) {
            ++pos_;
        }
    }

    std::optional<std::string> parse_name() {
        skip_ws();
        const std::size_t start = pos_;
        while (pos_ < source_.size()) {
            const char c = source_[pos_];
            if (c == '[' || c == ']' || c == ',' ||
                std::isspace(static_cast<unsigned char>(c))) {
                break;
            }
            ++pos_;
        }
        if (pos_ == start) {
            return std::nullopt;
        }
        return source_.substr(start, pos_ - start);
    }

    std::optional<Expr> parse_expr() {
        auto name = parse_name();
        if (!name) {
            return std::nullopt;
        }

        Expr out{*name, {}};
        skip_ws();
        if (pos_ >= source_.size() || source_[pos_] != '[') {
            return out;
        }
        ++pos_;
        while (true) {
            skip_ws();
            if (pos_ < source_.size() && source_[pos_] == ']') {
                ++pos_;
                return out;
            }
            auto arg = parse_expr();
            if (!arg) {
                return std::nullopt;
            }
            out.args.push_back(*arg);
            skip_ws();
            if (pos_ < source_.size() && source_[pos_] == ',') {
                ++pos_;
                continue;
            }
            if (pos_ < source_.size() && source_[pos_] == ']') {
                ++pos_;
                return out;
            }
            return std::nullopt;
        }
    }

    std::string source_;
    std::size_t pos_{0};
};

inline std::optional<Expr> parse(std::string source) {
    return Parser(std::move(source)).parse();
}

inline std::optional<Value> constant_value(const std::string& name) {
    if (name == "0") {
        return Value{0.0, 0.0};
    }
    if (name == "1") {
        return Value{1.0, 0.0};
    }
    if (name == "-1") {
        return Value{-1.0, 0.0};
    }
    if (name == "2") {
        return Value{2.0, 0.0};
    }
    if (name == "E") {
        return Value{std::numbers::e, 0.0};
    }
    if (name == "Pi") {
        return Value{std::numbers::pi, 0.0};
    }
    if (name == "EulerGamma") {
        return Value{kEulerGamma, 0.0};
    }
    if (name == "Glaisher") {
        return Value{kGlaisher, 0.0};
    }
    if (name == "Catalan") {
        return Value{kCatalan, 0.0};
    }
    if (name == "Khinchin") {
        return Value{kKhinchin, 0.0};
    }
    return std::nullopt;
}

inline std::optional<Value> eval_unary(const std::string& name, Value x) {
    constexpr Value kOne{1.0, 0.0};
    constexpr Value kHalf{0.5, 0.0};
    constexpr Value kI{0.0, 1.0};

    if (name == "Exp") {
        return checked(std::exp(x));
    }
    if (name == "Log") {
        return checked(std::log(x));
    }
    if (name == "Minus") {
        return checked(-x);
    }
    if (name == "Inv") {
        return checked(Value{1.0, 0.0} / x);
    }
    if (name == "Half") {
        return checked(x / Value{2.0, 0.0});
    }
    if (name == "Sqr") {
        return checked(x * x);
    }
    if (name == "Sqrt") {
        return checked(std::sqrt(x));
    }
    if (name == "LogisticSigmoid") {
        return checked(kOne / (kOne + std::exp(-x)));
    }
    if (name == "Cosh") {
        return checked(std::cosh(x));
    }
    if (name == "Sinh") {
        return checked(std::sinh(x));
    }
    if (name == "Tanh") {
        return checked(std::sinh(x) / std::cosh(x));
    }
    if (name == "Cos") {
        return checked(std::cos(x));
    }
    if (name == "Sin") {
        return checked(std::sin(x));
    }
    if (name == "Tan") {
        return checked(std::sin(x) / std::cos(x));
    }
    if (name == "ArcSinh") {
        return checked(std::log(x + std::sqrt(x * x + kOne)));
    }
    if (name == "ArcCosh") {
        return checked(std::log(x + std::sqrt(x + kOne) * std::sqrt(x - kOne)));
    }
    if (name == "ArcSin") {
        return checked(-kI * std::log(kI * x + std::sqrt(kOne - x * x)));
    }
    if (name == "ArcCos") {
        const auto asin_x = eval_unary("ArcSin", x);
        if (!asin_x) {
            return std::nullopt;
        }
        return checked(Value{std::numbers::pi / 2.0, 0.0} - *asin_x);
    }
    if (name == "ArcTan") {
        return checked(Value{0.0, 0.5} * (std::log(kOne - kI * x) - std::log(kOne + kI * x)));
    }
    if (name == "ArcTanh") {
        return checked(kHalf * (std::log(kOne + x) - std::log(kOne - x)));
    }
    return std::nullopt;
}

inline std::optional<Value> eval_binary(const std::string& name, Value a, Value b) {
    if (name == "EML") {
        return checked(eml(a, b));
    }
    if (name == "Subtract") {
        return checked(a - b);
    }
    if (name == "Plus") {
        return checked(a + b);
    }
    if (name == "Times") {
        return checked(a * b);
    }
    if (name == "Divide") {
        return checked(a / b);
    }
    if (name == "Avg") {
        return checked((a + b) / Value{2.0, 0.0});
    }
    if (name == "Power") {
        return checked(std::pow(a, b));
    }
    if (name == "Log") {
        return checked(std::log(b) / std::log(a));
    }
    if (name == "Hypot") {
        return checked(std::sqrt(a * a + b * b));
    }
    return std::nullopt;
}

inline std::optional<Value> eval(const Expr& expr, const Env& env) {
    if (expr.is_atom()) {
        if (auto found = env.find(expr.name); found != env.end()) {
            return checked(found->second);
        }
        return constant_value(expr.name);
    }
    if (expr.args.size() == 1) {
        auto x = eval(expr.args[0], env);
        if (!x) {
            return std::nullopt;
        }
        return eval_unary(expr.name, *x);
    }
    if (expr.args.size() == 2) {
        auto a = eval(expr.args[0], env);
        auto b = eval(expr.args[1], env);
        if (!a || !b) {
            return std::nullopt;
        }
        return eval_binary(expr.name, *a, *b);
    }
    return std::nullopt;
}

struct AnchorPair {
    double x{};
    double y{};
};

inline std::vector<AnchorPair> anchor_pairs() {
    return {
        {kEulerGamma, kGlaisher},
        {-kEulerGamma, kGlaisher},
        {kCatalan, kKhinchin},
        {-kCatalan, kKhinchin},
        {kGlaisher, kEulerGamma},
        {-kGlaisher, kEulerGamma},
        {kKhinchin, kCatalan},
        {-kKhinchin, kCatalan},
    };
}

inline Env unary_env(double x = kEulerGamma) {
    return {{"x", Value{x, 0.0}}};
}

inline std::vector<Env> unary_envs() {
    return {
        unary_env(kEulerGamma),
        unary_env(-kEulerGamma),
        unary_env(kCatalan),
        unary_env(-kCatalan),
        unary_env(kGlaisher),
        unary_env(-kGlaisher),
        unary_env(kKhinchin),
        unary_env(-kKhinchin),
    };
}

inline Env binary_env(double x, double y) {
    return {{"x", Value{x, 0.0}}, {"y", Value{y, 0.0}}};
}

inline bool verify_identity(
    const std::string& target_source,
    const std::string& witness_source,
    const std::vector<Env>& envs,
    double eps = 1e-9,
    bool skip_nonreal_target = false) {
    const auto target = parse(target_source);
    const auto witness = parse(witness_source);
    if (!target || !witness) {
        return false;
    }

    std::size_t tested = 0;
    for (const Env& env : envs) {
        const auto target_value = eval(*target, env);
        if (!target_value) {
            continue;
        }
        if (skip_nonreal_target && !imag_is_zero(*target_value, eps)) {
            continue;
        }
        const auto witness_value = eval(*witness, env);
        if (!witness_value || !near(*witness_value, *target_value, eps)) {
            return false;
        }
        ++tested;
    }
    return tested > 0;
}

}  // namespace eml_sr::witness
