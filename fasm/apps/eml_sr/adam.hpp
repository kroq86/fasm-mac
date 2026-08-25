#pragma once

#include "search.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <functional>
#include <limits>
#include <random>
#include <string>
#include <vector>

namespace eml_sr {

struct AdamState {
    double beta1{0.9};
    double beta2{0.999};
    double epsilon{1e-8};
    std::vector<double> m;
    std::vector<double> v;
    int step{0};
};

struct ParamLeaf {
    double logit_one{0.0};
    double logit_x{0.0};
    double logit_f{0.0};
    double f_value{1.0};
};

inline void softmax3(double a, double b, double c, double temp, double& o, double& x, double& f) {
    const double inv = 1.0 / temp;
    const double ma = std::max({a * inv, b * inv, c * inv});
    const double ea = std::exp(a * inv - ma);
    const double eb = std::exp(b * inv - ma);
    const double ec = std::exp(c * inv - ma);
    const double s = ea + eb + ec;
    o = ea / s;
    x = eb / s;
    f = ec / s;
}

inline Tree left_heavy_shape(int eml_nodes) {
    if (eml_nodes <= 0) {
        return leaf_tree(LeafKind::X);
    }
    return eml_tree(left_heavy_shape(eml_nodes - 1), leaf_tree(LeafKind::One));
}

inline double eval_param_tree(
    const Tree& shape,
    const std::vector<ParamLeaf>& leaves,
    double x,
    double temperature) {
    std::size_t leaf_idx = 0;
    std::function<double(int)> walk = [&](int idx) -> double {
        const Node& node = shape.nodes[static_cast<std::size_t>(idx)];
        if (node.tag == Node::Tag::Leaf) {
            const ParamLeaf& pl = leaves[leaf_idx++];
            double w_one = 0.0;
            double w_x = 0.0;
            double w_f = 0.0;
            softmax3(pl.logit_one, pl.logit_x, pl.logit_f, temperature, w_one, w_x, w_f);
            return w_one * 1.0 + w_x * x + w_f * pl.f_value;
        }
        const double left = walk(node.left);
        const double right = walk(node.right);
        return eml_real(left, right);
    };
    return walk(shape.root);
}

inline double mse_param_tree(
    const Tree& shape,
    const std::vector<ParamLeaf>& leaves,
    const std::vector<DataPoint>& data,
    double temperature) {
    double sum = 0.0;
    for (const DataPoint& pt : data) {
        const double pred = eval_param_tree(shape, leaves, pt.x, temperature);
        if (!std::isfinite(pred)) {
            return std::numeric_limits<double>::infinity();
        }
        const double d = pred - pt.y;
        sum += d * d;
    }
    return sum / static_cast<double>(data.size());
}

inline Tree snap_param_tree(const Tree& shape, const std::vector<ParamLeaf>& leaves) {
    Tree out = shape;
    std::size_t leaf_idx = 0;
    std::function<void(int)> walk = [&](int idx) {
        Node& node = out.nodes[static_cast<std::size_t>(idx)];
        if (node.tag == Node::Tag::Leaf) {
            const ParamLeaf& pl = leaves[leaf_idx++];
            double w_one = 0.0;
            double w_x = 0.0;
            double w_f = 0.0;
            softmax3(pl.logit_one, pl.logit_x, pl.logit_f, 1e-6, w_one, w_x, w_f);
            if (w_x >= w_one && w_x >= w_f) {
                node.leaf = LeafKind::X;
            } else if (w_f >= w_one) {
                node.leaf = LeafKind::F;
                node.f_value = pl.f_value;
            } else {
                node.leaf = LeafKind::One;
            }
            return;
        }
        walk(node.left);
        walk(node.right);
    };
    walk(out.root);
    return out;
}

inline void pack_params(const std::vector<ParamLeaf>& leaves, std::vector<double>& out) {
    out.clear();
    out.reserve(leaves.size() * 4);
    for (const ParamLeaf& pl : leaves) {
        out.push_back(pl.logit_one);
        out.push_back(pl.logit_x);
        out.push_back(pl.logit_f);
        out.push_back(pl.f_value);
    }
}

inline void unpack_params(const std::vector<double>& packed, std::vector<ParamLeaf>& leaves) {
    for (std::size_t i = 0; i < leaves.size(); ++i) {
        const std::size_t base = i * 4;
        leaves[i].logit_one = packed[base + 0];
        leaves[i].logit_x = packed[base + 1];
        leaves[i].logit_f = packed[base + 2];
        leaves[i].f_value = std::max(1e-6, packed[base + 3]);
    }
}

inline void adam_step(std::vector<double>& params, const std::vector<double>& grad, AdamState& state, double lr) {
    if (state.m.size() != params.size()) {
        state.m.assign(params.size(), 0.0);
        state.v.assign(params.size(), 0.0);
    }
    ++state.step;
    const double bc1 = 1.0 - std::pow(state.beta1, state.step);
    const double bc2 = 1.0 - std::pow(state.beta2, state.step);
    for (std::size_t i = 0; i < params.size(); ++i) {
        state.m[i] = state.beta1 * state.m[i] + (1.0 - state.beta1) * grad[i];
        state.v[i] = state.beta2 * state.v[i] + (1.0 - state.beta2) * grad[i] * grad[i];
        const double m_hat = state.m[i] / bc1;
        const double v_hat = state.v[i] / bc2;
        params[i] -= lr * m_hat / (std::sqrt(v_hat) + state.epsilon);
    }
}

inline std::size_t collect_leaf_count(const Tree& tree) {
    std::vector<int> leaves;
    collect_leaf_indices(tree, tree.root, leaves);
    return leaves.size();
}

inline SearchResult adam_search_best(
    const std::vector<DataPoint>& data,
    int max_eml_nodes,
    const SearchOptions& opts,
    double goal_mse) {
    SearchResult merged{};
    for (int eml_nodes = 1; eml_nodes <= max_eml_nodes; ++eml_nodes) {
        SearchResult trial{};
        EvalContext ctx{};
        ctx.domain = opts.domain;
        ctx.stats = opts.profile ? &trial.stats : nullptr;

        const Tree shape = left_heavy_shape(eml_nodes);
        std::vector<ParamLeaf> leaves(collect_leaf_count(shape));
        std::mt19937 rng(42 + static_cast<std::uint32_t>(eml_nodes));
        std::uniform_real_distribution<double> dist(-0.3, 0.3);
        for (ParamLeaf& pl : leaves) {
            pl.logit_one = dist(rng);
            pl.logit_x = dist(rng) + 0.5;
            pl.logit_f = dist(rng);
            pl.f_value = 1.0;
        }

        std::vector<double> params;
        pack_params(leaves, params);
        AdamState adam_state{};

        for (int epoch = 0; epoch < opts.adam_epochs; ++epoch) {
            const double temp = std::max(0.05, 1.0 - static_cast<double>(epoch) / static_cast<double>(opts.adam_epochs));
            unpack_params(params, leaves);
            const double base_mse = mse_param_tree(shape, leaves, data, temp);
            if (!std::isfinite(base_mse)) {
                continue;
            }

            std::vector<double> grad(params.size(), 0.0);
            constexpr double kEps = 1e-4;
            for (std::size_t i = 0; i < params.size(); ++i) {
                params[i] += kEps;
                unpack_params(params, leaves);
                const double plus = mse_param_tree(shape, leaves, data, temp);
                params[i] -= 2.0 * kEps;
                unpack_params(params, leaves);
                const double minus = mse_param_tree(shape, leaves, data, temp);
                params[i] += kEps;
                grad[i] = (plus - minus) / (2.0 * kEps);
            }

            adam_step(params, grad, adam_state, opts.adam_lr);
        }

        unpack_params(params, leaves);
        Tree snapped = snap_param_tree(shape, leaves);
        optimize_f_params_ctx(snapped, data, ctx, trial.mse);
        const double mse = mse_for_tree_ctx(snapped, data, ctx, trial.mse);
        trial.tree = snapped;
        trial.mse = mse;
        trial.rpn = to_rpn(snapped);
        ++trial.stats.forms_seen;
        ++trial.stats.candidates_evaled;

        if (trial.mse < merged.mse) {
            merged = std::move(trial);
        }
        if (goal_mse > 0.0 && merged.mse <= goal_mse) {
            break;
        }
    }
    return merged;
}

}  // namespace eml_sr
