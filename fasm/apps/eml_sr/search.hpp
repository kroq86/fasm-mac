#pragma once

#include "eval.hpp"
#include "stats.hpp"
#include "tree.hpp"

#include <cmath>
#include <cstdlib>
#include <functional>
#include <limits>
#include <string>
#include <utility>
#include <vector>

namespace eml_sr {

struct DataPoint {
    double x{};
    double y{};
};

enum class SearchMethod { Enumerate, LegacyEnumerate, Adam };

struct SearchOptions {
    EvalDomain domain{EvalDomain::Complex};
    int jobs{1};
    bool profile{false};
    SearchMethod method{SearchMethod::Enumerate};
    int adam_epochs{2000};
    double adam_lr{0.05};
};

struct SearchResult {
    Tree tree{};
    double mse{std::numeric_limits<double>::infinity()};
    std::string rpn;
    SearchStats stats{};
};

inline void collect_leaf_indices(const Tree& tree, int idx, std::vector<int>& out) {
    const Node& node = tree.nodes[static_cast<std::size_t>(idx)];
    if (node.tag == Node::Tag::Leaf) {
        out.push_back(idx);
        return;
    }
    collect_leaf_indices(tree, node.left, out);
    collect_leaf_indices(tree, node.right, out);
}

inline void collect_f_leaf_indices(const Tree& tree, int idx, std::vector<int>& out) {
    const Node& node = tree.nodes[static_cast<std::size_t>(idx)];
    if (node.tag == Node::Tag::Leaf) {
        if (node.leaf == LeafKind::F) {
            out.push_back(idx);
        }
        return;
    }
    collect_f_leaf_indices(tree, node.left, out);
    collect_f_leaf_indices(tree, node.right, out);
}

inline double mse_for_tree_ctx(
    const Tree& tree,
    const std::vector<DataPoint>& data,
    EvalContext& ctx,
    double prune_above = std::numeric_limits<double>::infinity(),
    std::size_t point_limit = 0) {
    if (tree.empty() || data.empty()) {
        return std::numeric_limits<double>::infinity();
    }
    const std::size_t n = point_limit == 0 ? data.size() : std::min(point_limit, data.size());
    const double inv_n = 1.0 / static_cast<double>(n);
    double sum = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        const double pred = eval_real_ctx(tree, data[i].x, ctx);
        if (!std::isfinite(pred)) {
            if (ctx.stats) {
                ++ctx.stats->pruned_by_numeric;
            }
            return std::numeric_limits<double>::infinity();
        }
        const double d = pred - data[i].y;
        sum += d * d;
        if (sum * inv_n > prune_above) {
            if (ctx.stats) {
                ++ctx.stats->pruned_by_partial_mse;
            }
            return std::numeric_limits<double>::infinity();
        }
    }
    return sum * inv_n;
}

inline void optimize_f_params_ctx(
    Tree& tree,
    const std::vector<DataPoint>& data,
    EvalContext& ctx,
    double prune_above = std::numeric_limits<double>::infinity()) {
    std::vector<int> f_leaves;
    collect_f_leaf_indices(tree, tree.root, f_leaves);
    if (f_leaves.empty()) {
        return;
    }

    for (int pass = 0; pass < 8; ++pass) {
        bool improved = false;
        for (const int leaf_idx : f_leaves) {
            Node& node = tree.nodes[static_cast<std::size_t>(leaf_idx)];
            const double start = node.f_value;
            double best_f = start;
            double best_mse = mse_for_tree_ctx(tree, data, ctx, prune_above);
            const double probes[] = {
                start * 0.25, start * 0.5, start * 2.0, start * 4.0,
                start + 0.1,  start - 0.1,  start + 1.0, start - 1.0,
                0.5, 1.0, 2.0, 3.0, 4.0, 5.0,
            };
            for (const double trial_raw : probes) {
                const double trial = trial_raw < 1e-6 ? 1e-6 : trial_raw;
                node.f_value = trial;
                const double trial_mse = mse_for_tree_ctx(tree, data, ctx, best_mse);
                if (trial_mse < best_mse) {
                    best_mse = trial_mse;
                    best_f = trial;
                    improved = true;
                    if (best_mse <= prune_above) {
                        prune_above = best_mse;
                    }
                }
            }
            node.f_value = best_f;
        }
        if (!improved) {
            break;
        }
    }
}

inline Tree with_leaf_assignment(const Tree& base, std::uint32_t assignment) {
    Tree t = base;
    std::vector<int> leaves;
    collect_leaf_indices(t, t.root, leaves);
    for (std::size_t i = 0; i < leaves.size(); ++i) {
        const std::uint32_t kind = (assignment / static_cast<std::uint32_t>(std::pow(3, i))) % 3U;
        Node& node = t.nodes[static_cast<std::size_t>(leaves[i])];
        if (kind == 0U) {
            node.leaf = LeafKind::One;
        } else if (kind == 1U) {
            node.leaf = LeafKind::X;
        } else {
            node.leaf = LeafKind::F;
            node.f_value = 1.0;
        }
    }
    return t;
}

inline bool assignment_has_f(std::uint32_t assignment, std::size_t leaf_count) {
    for (std::size_t i = 0; i < leaf_count; ++i) {
        if ((assignment / static_cast<std::uint32_t>(std::pow(3, i))) % 3U == 2U) {
            return true;
        }
    }
    return false;
}

inline void gen_shapes(int eml_nodes, const std::function<void(const Tree&)>& emit) {
    if (eml_nodes == 0) {
        emit(leaf_tree(LeafKind::One));
        emit(leaf_tree(LeafKind::X));
        emit(leaf_tree(LeafKind::F));
        return;
    }
    for (int left_eml = 0; left_eml < eml_nodes; ++left_eml) {
        const int right_eml = eml_nodes - 1 - left_eml;
        gen_shapes(left_eml, [&](const Tree& left_shape) {
            gen_shapes(right_eml, [&](const Tree& right_shape) {
                emit(eml_tree(left_shape, right_shape));
            });
        });
    }
}

inline std::array<double, 4> probe_x_values(const std::vector<DataPoint>& data) {
    std::array<double, 4> xs{};
    if (data.empty()) {
        return xs;
    }
    xs[0] = data[0].x;
    xs[1] = data.size() > 1 ? data[1].x : data[0].x;
    xs[2] = data.size() > 2 ? data[2].x : data[0].x;
    xs[3] = data.size() > 3 ? data[3].x : data[0].x;
    return xs;
}

inline std::uint64_t signature_hash(
    const Tree& tree,
    const std::array<double, 4>& probe_x,
    EvalContext& ctx) {
    std::uint64_t h = 14695981039346656037ULL;
    for (const double x : probe_x) {
        const double v = eval_real_ctx(tree, x, ctx);
        const std::int64_t q = static_cast<std::int64_t>(v * 1e6);
        h ^= static_cast<std::uint64_t>(q);
        h *= 1099511628211ULL;
    }
    return h;
}

inline bool use_legacy_search() {
    const char* env = std::getenv("EML_SR_LEGACY");
    return env != nullptr && env[0] != '\0' && env[0] != '0';
}

SearchResult pool_search_best(
    const std::vector<DataPoint>& data,
    int max_eml_nodes,
    const SearchOptions& opts,
    double goal_mse = 0.0);

SearchResult search_best_legacy(
    const std::vector<DataPoint>& data,
    int max_eml_nodes,
    const SearchOptions& opts,
    double goal_mse = 0.0);

SearchResult adam_search_best(
    const std::vector<DataPoint>& data,
    int max_eml_nodes,
    const SearchOptions& opts,
    double goal_mse = 0.0);

SearchResult arena_search_best(
    const std::vector<DataPoint>& data,
    int max_eml_nodes,
    const SearchOptions& opts,
    double goal_mse = 0.0);

inline SearchResult search_best(
    const std::vector<DataPoint>& data,
    int max_eml_nodes,
    const SearchOptions& opts = {},
    double goal_mse = 0.0) {
    if (opts.method == SearchMethod::Adam) {
        return adam_search_best(data, max_eml_nodes, opts, goal_mse);
    }
    if (opts.method == SearchMethod::LegacyEnumerate) {
        return search_best_legacy(data, max_eml_nodes, opts, goal_mse);
    }
    return arena_search_best(data, max_eml_nodes, opts, goal_mse);
}

}  // namespace eml_sr
