#pragma once

#include "eval.hpp"
#include "tree.hpp"

#include <bit>
#include <cstdint>
#include <limits>
#include <optional>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace eml_sr {

using ExprId = std::uint32_t;

inline constexpr ExprId kInvalidExprId = std::numeric_limits<ExprId>::max();

enum class ExprKind : std::uint8_t { Leaf, Eml };

struct ExprNode {
    ExprKind kind{ExprKind::Leaf};
    LeafKind leaf{LeafKind::One};
    double f_value{1.0};
    ExprId left{kInvalidExprId};
    ExprId right{kInvalidExprId};
    std::uint32_t eml_count{0};
    std::uint32_t leaf_count{1};
    std::uint64_t hash{0};
    mutable std::string rpn_cache;
};

struct ExprKey {
    ExprKind kind{ExprKind::Leaf};
    LeafKind leaf{LeafKind::One};
    std::uint64_t f_bits{0};
    ExprId left{kInvalidExprId};
    ExprId right{kInvalidExprId};

    bool operator==(const ExprKey& other) const {
        return kind == other.kind && leaf == other.leaf && f_bits == other.f_bits &&
               left == other.left && right == other.right;
    }
};

struct ExprKeyHash {
    std::size_t operator()(const ExprKey& key) const {
        std::uint64_t h = 14695981039346656037ULL;
        auto mix = [&](std::uint64_t v) {
            h ^= v;
            h *= 1099511628211ULL;
        };
        mix(static_cast<std::uint64_t>(key.kind));
        mix(static_cast<std::uint64_t>(key.leaf));
        mix(key.f_bits);
        mix(key.left);
        mix(key.right);
        return static_cast<std::size_t>(h);
    }
};

inline std::uint64_t stable_mix(std::uint64_t h, std::uint64_t v) {
    h ^= v;
    h *= 1099511628211ULL;
    return h;
}

inline std::uint64_t f_value_key(double value) {
    return std::bit_cast<std::uint64_t>(value);
}

class ExprArena {
public:
    ExprId intern_leaf(LeafKind leaf, double f_value = 1.0) {
        ExprKey key{ExprKind::Leaf, leaf, f_value_key(f_value), kInvalidExprId, kInvalidExprId};
        if (auto it = ids_.find(key); it != ids_.end()) {
            return it->second;
        }
        ExprNode node{};
        node.kind = ExprKind::Leaf;
        node.leaf = leaf;
        node.f_value = f_value;
        node.left = kInvalidExprId;
        node.right = kInvalidExprId;
        node.eml_count = 0;
        node.leaf_count = 1;
        node.hash = stable_mix(stable_mix(stable_mix(14695981039346656037ULL, 0), static_cast<std::uint64_t>(leaf)), key.f_bits);
        return push_node(key, node);
    }

    ExprId intern_eml(ExprId left, ExprId right) {
        ExprKey key{ExprKind::Eml, LeafKind::One, 0, left, right};
        if (auto it = ids_.find(key); it != ids_.end()) {
            return it->second;
        }
        const ExprNode& l = node(left);
        const ExprNode& r = node(right);
        ExprNode out{};
        out.kind = ExprKind::Eml;
        out.left = left;
        out.right = right;
        out.eml_count = l.eml_count + r.eml_count + 1;
        out.leaf_count = l.leaf_count + r.leaf_count;
        out.hash = stable_mix(stable_mix(stable_mix(14695981039346656037ULL, 1), l.hash), r.hash);
        return push_node(key, out);
    }

    [[nodiscard]] const ExprNode& node(ExprId id) const {
        return nodes_[static_cast<std::size_t>(id)];
    }

    [[nodiscard]] std::size_t size() const {
        return nodes_.size();
    }

    [[nodiscard]] std::string to_rpn(ExprId id) const {
        const ExprNode& n = node(id);
        if (!n.rpn_cache.empty()) {
            return n.rpn_cache;
        }
        if (n.kind == ExprKind::Leaf) {
            n.rpn_cache = n.leaf == LeafKind::One ? "1" : n.leaf == LeafKind::X ? "x" : "f";
            return n.rpn_cache;
        }
        n.rpn_cache = to_rpn(n.left) + " " + to_rpn(n.right) + " eml";
        return n.rpn_cache;
    }

private:
    ExprId push_node(const ExprKey& key, const ExprNode& node) {
        const ExprId id = static_cast<ExprId>(nodes_.size());
        nodes_.push_back(node);
        ids_.emplace(key, id);
        return id;
    }

    std::vector<ExprNode> nodes_;
    std::unordered_map<ExprKey, ExprId, ExprKeyHash> ids_;
};

struct ArenaEvalKey {
    ExprId id{kInvalidExprId};
    std::uint64_t x_bits{0};
    EvalDomain domain{EvalDomain::Complex};
    bool numeric_prune{true};

    bool operator==(const ArenaEvalKey& other) const {
        return id == other.id && x_bits == other.x_bits && domain == other.domain &&
               numeric_prune == other.numeric_prune;
    }
};

struct ArenaEvalKeyHash {
    std::size_t operator()(const ArenaEvalKey& key) const {
        std::uint64_t h = 14695981039346656037ULL;
        h = stable_mix(h, key.id);
        h = stable_mix(h, key.x_bits);
        h = stable_mix(h, static_cast<std::uint64_t>(key.domain));
        h = stable_mix(h, key.numeric_prune ? 1U : 0U);
        return static_cast<std::size_t>(h);
    }
};

struct ArenaEvalMemo {
    std::unordered_map<ArenaEvalKey, std::complex<double>, ArenaEvalKeyHash> cache;
};

inline std::complex<double> eval_expr(
    const ExprArena& arena,
    ExprId id,
    std::complex<double> x,
    EvalContext& ctx,
    ArenaEvalMemo& memo) {
    const ArenaEvalKey key{id, std::bit_cast<std::uint64_t>(x.real()), ctx.domain, ctx.numeric_prune};
    if (x.imag() == 0.0) {
        if (auto it = memo.cache.find(key); it != memo.cache.end()) {
            if (ctx.stats) {
                ++ctx.stats->cache_hits;
            }
            return it->second;
        }
        if (ctx.stats) {
            ++ctx.stats->cache_misses;
        }
    }

    const ExprNode& n = arena.node(id);
    std::complex<double> out{};
    if (n.kind == ExprKind::Leaf) {
        if (n.leaf == LeafKind::One) {
            out = {1.0, 0.0};
        } else if (n.leaf == LeafKind::X) {
            out = x;
        } else {
            out = {n.f_value, 0.0};
        }
    } else {
        const auto left = eval_expr(arena, n.left, x, ctx, memo);
        if (ctx.numeric_prune && !passes_numeric_gate(left, ctx)) {
            out = {std::numeric_limits<double>::quiet_NaN(), 0.0};
        } else {
            const auto right = eval_expr(arena, n.right, x, ctx, memo);
            if (ctx.numeric_prune && !passes_numeric_gate(right, ctx)) {
                out = {std::numeric_limits<double>::quiet_NaN(), 0.0};
            } else {
                if (ctx.stats) {
                    ++ctx.stats->eml_calls;
                }
                out = ctx.domain == EvalDomain::Real
                          ? std::complex<double>{eml_real(std::real(left), std::real(right)), 0.0}
                          : eml(left, right);
            }
        }
    }
    if (x.imag() == 0.0) {
        memo.cache[key] = out;
    }
    return out;
}

inline std::complex<double> eval_expr(
    const ExprArena& arena,
    ExprId id,
    std::complex<double> x,
    EvalContext& ctx) {
    ArenaEvalMemo memo{};
    return eval_expr(arena, id, x, ctx, memo);
}

inline double eval_expr_real(
    const ExprArena& arena,
    ExprId id,
    double x,
    EvalContext& ctx,
    ArenaEvalMemo& memo) {
    const std::complex<double> z = eval_expr(arena, id, {x, 0.0}, ctx, memo);
    if (ctx.numeric_prune && !passes_numeric_gate(z, ctx)) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    return std::real(z);
}

inline ExprId tree_to_expr(const Tree& tree, int idx, ExprArena& arena) {
    const Node& n = tree.nodes[static_cast<std::size_t>(idx)];
    if (n.tag == Node::Tag::Leaf) {
        return arena.intern_leaf(n.leaf, n.f_value);
    }
    return arena.intern_eml(tree_to_expr(tree, n.left, arena), tree_to_expr(tree, n.right, arena));
}

inline ExprId tree_to_expr(const Tree& tree, ExprArena& arena) {
    if (tree.empty()) {
        return kInvalidExprId;
    }
    return tree_to_expr(tree, tree.root, arena);
}

inline int expr_to_tree_node(const ExprArena& arena, ExprId id, Tree& tree) {
    const ExprNode& n = arena.node(id);
    if (n.kind == ExprKind::Leaf) {
        const int idx = static_cast<int>(tree.nodes.size());
        tree.nodes.push_back(Node{Node::Tag::Leaf, n.leaf, n.f_value, -1, -1});
        return idx;
    }
    const int left = expr_to_tree_node(arena, n.left, tree);
    const int right = expr_to_tree_node(arena, n.right, tree);
    const int idx = static_cast<int>(tree.nodes.size());
    tree.nodes.push_back(Node{Node::Tag::Eml, LeafKind::One, 1.0, left, right});
    return idx;
}

inline Tree expr_to_tree(const ExprArena& arena, ExprId id) {
    Tree tree{};
    if (id == kInvalidExprId) {
        return tree;
    }
    tree.root = expr_to_tree_node(arena, id, tree);
    return tree;
}

}  // namespace eml_sr
