#pragma once

#include "stats.hpp"
#include "tree.hpp"

#include <cmath>
#include <cstdint>
#include <limits>
#include <unordered_map>

namespace eml_sr {

enum class EvalDomain { Real, Complex };

struct EvalContext {
    EvalDomain domain{EvalDomain::Complex};
    SearchStats* stats{nullptr};
    std::unordered_map<std::uint64_t, std::complex<double>> cache;
    double imag_limit{1e-6};
    bool numeric_prune{true};

    static constexpr double kMaxAbs = 1e6;
};

inline std::uint64_t tree_content_hash(const Tree& tree) {
    std::uint64_t h = 14695981039346656037ULL;
    for (const Node& node : tree.nodes) {
        h ^= static_cast<std::uint64_t>(node.tag);
        h *= 1099511628211ULL;
        h ^= static_cast<std::uint64_t>(node.leaf);
        h *= 1099511628211ULL;
        const std::int64_t fq = static_cast<std::int64_t>(node.f_value * 10000.0);
        h ^= static_cast<std::uint64_t>(fq);
        h *= 1099511628211ULL;
    }
    return h;
}

inline std::uint64_t eval_cache_key(const Tree& tree, double x) {
    std::uint64_t h = tree_content_hash(tree);
    h ^= *reinterpret_cast<const std::uint64_t*>(&x);
    h *= 1099511628211ULL;
    return h;
}

inline bool passes_numeric_gate(const std::complex<double>& z, const EvalContext& ctx) {
    if (!std::isfinite(z.real()) || !std::isfinite(z.imag())) {
        return false;
    }
    if (std::abs(z.real()) > EvalContext::kMaxAbs || std::abs(z.imag()) > EvalContext::kMaxAbs) {
        return false;
    }
    if (ctx.domain == EvalDomain::Real && std::abs(z.imag()) > ctx.imag_limit) {
        return false;
    }
    return true;
}

inline double eml_real(double x, double y) {
    if (!(y > 0.0)) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    return std::exp(x) - std::log(y);
}

inline std::complex<double> eval_node_ctx(
    const Tree& tree,
    int idx,
    std::complex<double> x,
    EvalContext& ctx) {
    const Node& node = tree.nodes[static_cast<std::size_t>(idx)];
    if (node.tag == Node::Tag::Leaf) {
        if (node.leaf == LeafKind::One) {
            return {1.0, 0.0};
        }
        if (node.leaf == LeafKind::X) {
            return x;
        }
        return {node.f_value, 0.0};
    }
    const auto left = eval_node_ctx(tree, node.left, x, ctx);
    if (ctx.numeric_prune && !passes_numeric_gate(left, ctx)) {
        return {std::numeric_limits<double>::quiet_NaN(), 0.0};
    }
    const auto right = eval_node_ctx(tree, node.right, x, ctx);
    if (ctx.numeric_prune && !passes_numeric_gate(right, ctx)) {
        return {std::numeric_limits<double>::quiet_NaN(), 0.0};
    }
    if (ctx.stats) {
        ++ctx.stats->eml_calls;
    }
    if (ctx.domain == EvalDomain::Real) {
        return {eml_real(std::real(left), std::real(right)), 0.0};
    }
    return eml(left, right);
}

inline std::complex<double> eval_ctx(const Tree& tree, std::complex<double> x, EvalContext& ctx) {
    if (tree.empty()) {
        throw std::runtime_error("EmptyTree");
    }
    return eval_node_ctx(tree, tree.root, x, ctx);
}

inline double eval_real_ctx(const Tree& tree, double x, EvalContext& ctx) {
    const std::uint64_t key = eval_cache_key(tree, x);
    if (auto it = ctx.cache.find(key); it != ctx.cache.end()) {
        if (ctx.stats) {
            ++ctx.stats->cache_hits;
        }
        return std::real(it->second);
    }
    if (ctx.stats) {
        ++ctx.stats->cache_misses;
    }
    const std::complex<double> z = eval_ctx(tree, {x, 0.0}, ctx);
    if (ctx.numeric_prune && !passes_numeric_gate(z, ctx)) {
        ctx.cache[key] = {std::numeric_limits<double>::quiet_NaN(), 0.0};
        return std::numeric_limits<double>::quiet_NaN();
    }
    ctx.cache[key] = z;
    return std::real(z);
}

}  // namespace eml_sr
