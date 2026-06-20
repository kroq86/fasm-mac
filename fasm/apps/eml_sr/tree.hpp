#pragma once

#include "eml.hpp"

#include <cstdint>
#include <iomanip>
#include <ostream>
#include <sstream>
#include <string>
#include <vector>

namespace eml_sr {

enum class LeafKind : std::uint8_t { One, X, F };

struct Node {
    enum class Tag : std::uint8_t { Leaf, Eml } tag{Tag::Leaf};
    LeafKind leaf{LeafKind::One};
    double f_value{1.0};
    int left{-1};
    int right{-1};
};

struct Tree {
    std::vector<Node> nodes;
    int root{-1};

    [[nodiscard]] bool empty() const { return root < 0; }
    [[nodiscard]] std::size_t eml_count() const {
        std::size_t n = 0;
        for (const auto& node : nodes) {
            if (node.tag == Node::Tag::Eml) {
                ++n;
            }
        }
        return n;
    }
};

inline std::complex<double> eval_node(const Tree& tree, int idx, std::complex<double> x) {
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
    const auto left = eval_node(tree, node.left, x);
    const auto right = eval_node(tree, node.right, x);
    return eml(left, right);
}

inline std::complex<double> eval(const Tree& tree, std::complex<double> x) {
    if (tree.empty()) {
        throw std::runtime_error("EmptyTree");
    }
    return eval_node(tree, tree.root, x);
}

inline double eval_real(const Tree& tree, double x) {
    return std::real(eval(tree, {x, 0.0}));
}

inline void append_rpn_node(const Tree& tree, int idx, std::vector<std::string>& out) {
    const Node& node = tree.nodes[static_cast<std::size_t>(idx)];
    if (node.tag == Node::Tag::Leaf) {
        out.push_back(node.leaf == LeafKind::One ? "1"
                      : node.leaf == LeafKind::X ? "x"
                                                 : "f");
        return;
    }
    append_rpn_node(tree, node.left, out);
    append_rpn_node(tree, node.right, out);
    out.push_back("eml");
}

inline std::string to_rpn(const Tree& tree) {
    if (tree.empty()) {
        return {};
    }
    std::vector<std::string> tokens;
    append_rpn_node(tree, tree.root, tokens);
    std::ostringstream ss;
    for (std::size_t i = 0; i < tokens.size(); ++i) {
        if (i > 0) {
            ss << ' ';
        }
        ss << tokens[i];
    }
    return ss.str();
}

inline std::string format_f_value(double value) {
    std::ostringstream ss;
    ss << std::setprecision(4) << value;
    return ss.str();
}

struct DotOptions {
    bool annotate_values{false};
    double eval_x{1.0};
};

inline double eval_real_at_node(const Tree& tree, int idx, double x) {
    return std::real(eval_node(tree, idx, {x, 0.0}));
}

inline void dot_node(
    const Tree& tree,
    int idx,
    std::ostream& out,
    int& next_id,
    const std::string& my_id,
    const DotOptions& opts) {
    const Node& node = tree.nodes[static_cast<std::size_t>(idx)];
    if (node.tag == Node::Tag::Leaf) {
        const double w = opts.annotate_values ? 0.75 : 0.55;
        const double h = opts.annotate_values ? 0.38 : 0.32;
        out << "  " << my_id << " [shape=box, style=\"filled,rounded\", penwidth=1.2, width=" << w
            << ", height=" << h << ", fixedsize=true";
        if (node.leaf == LeafKind::One) {
            out << ", label=\"1\", fillcolor=\"#e5e7eb\"";
        } else if (node.leaf == LeafKind::X) {
            if (opts.annotate_values) {
                out << ", label=\"x\\ninput=" << format_f_value(opts.eval_x) << "\"";
            } else {
                out << ", label=\"x\"";
            }
            out << ", fillcolor=\"#dcfce7\"";
        } else {
            out << ", label=\"f=" << format_f_value(node.f_value) << "\", fillcolor=\"#ffedd5\"";
        }
        out << "];\n";
        return;
    }
    const double sz = opts.annotate_values ? 1.05 : 0.85;
    out << "  " << my_id << " [shape=circle, style=\"filled\", fillcolor=\"#dbeafe\", "
           "penwidth=1.4, width=" << sz << ", height=" << sz << ", fixedsize=true, label=\"";
    if (opts.annotate_values) {
        const double value = eval_real_at_node(tree, idx, opts.eval_x);
        out << "eml\\n≈" << format_f_value(value);
    } else {
        out << "eml";
    }
    out << "\"];\n";
    const std::string left_id = "n" + std::to_string(next_id++);
    const std::string right_id = "n" + std::to_string(next_id++);
    out << "  " << my_id << " -> " << left_id
        << " [label=\"  exp(L)  \", color=\"#2563eb\", fontcolor=\"#2563eb\", penwidth=1.3];\n";
    out << "  " << my_id << " -> " << right_id
        << " [label=\"  ln(R)  \", color=\"#dc2626\", fontcolor=\"#dc2626\", penwidth=1.3];\n";
    dot_node(tree, node.left, out, next_id, left_id, opts);
    dot_node(tree, node.right, out, next_id, right_id, opts);
}

inline std::string to_dot(const Tree& tree, const DotOptions& opts = {}) {
    if (tree.empty()) {
        return "digraph EMLTree {}\n";
    }
    std::ostringstream out;
    out << "digraph EMLTree {\n";
    out << "  graph [rankdir=TB, bgcolor=\"white\", fontname=\"Helvetica\", fontsize=12,\n";
    out << "         nodesep=0.55, ranksep=0.75, splines=true, overlap=false, pad=0.35];\n";
    out << "  node [fontname=\"Helvetica\", fontsize=11];\n";
    out << "  edge [fontname=\"Helvetica\", fontsize=10];\n";
    out << "  labelloc=t;\n";
    if (opts.annotate_values) {
        out << "  label=\"EML tree @ x=" << format_f_value(opts.eval_x)
            << "  |  eml(L,R)=exp(L)-ln(R)  |  eml_nodes=" << tree.eml_count() << "\";\n";
    } else {
        out << "  label=\"EML tree: eml(L,R)=exp(L)-ln(R)  |  eml_nodes=" << tree.eml_count()
            << "\";\n";
    }
    int next_id = 1;
    dot_node(tree, tree.root, out, next_id, "n0", opts);
    out << "}\n";
    return out.str();
}

inline Tree leaf_tree(LeafKind kind, double f_value = 1.0) {
    Tree t{};
    t.nodes.push_back(Node{Node::Tag::Leaf, kind, f_value, -1, -1});
    t.root = 0;
    return t;
}

inline Tree eml_tree(const Tree& left, const Tree& right) {
    Tree t{};
    t.nodes = left.nodes;
    const int right_offset = static_cast<int>(t.nodes.size());
    for (Node node : right.nodes) {
        if (node.left >= 0) {
            node.left += right_offset;
        }
        if (node.right >= 0) {
            node.right += right_offset;
        }
        t.nodes.push_back(node);
    }
    const int left_root = left.root;
    const int right_root = right.root + right_offset;
    const int eml_idx = static_cast<int>(t.nodes.size());
    t.nodes.push_back(Node{Node::Tag::Eml, LeafKind::One, 1.0, left_root, right_root});
    t.root = eml_idx;
    return t;
}

inline Tree preset_exp_tree() {
    return eml_tree(leaf_tree(LeafKind::X), leaf_tree(LeafKind::One));
}

// Paper: ln(x) = eml(1, eml(eml(1,x), 1))
inline Tree preset_ln_tree() {
    const Tree inner = eml_tree(leaf_tree(LeafKind::One), leaf_tree(LeafKind::X));
    const Tree mid = eml_tree(inner, leaf_tree(LeafKind::One));
    return eml_tree(leaf_tree(LeafKind::One), mid);
}

}  // namespace eml_sr
