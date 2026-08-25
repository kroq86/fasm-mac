#include "adam.hpp"
#include "bindings_api.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <stdexcept>

namespace py = pybind11;

namespace {

eml_sr::SearchMethod parse_method(const std::string& method) {
    if (method == "adam") {
        return eml_sr::SearchMethod::Adam;
    }
    if (method == "enumerate") {
        return eml_sr::SearchMethod::Enumerate;
    }
    if (method == "legacy-enumerate") {
        return eml_sr::SearchMethod::LegacyEnumerate;
    }
    throw std::invalid_argument("method must be 'enumerate' or 'adam'");
}

eml_sr::EvalDomain parse_domain(const std::string& domain) {
    if (domain == "real") {
        return eml_sr::EvalDomain::Real;
    }
    if (domain == "complex") {
        return eml_sr::EvalDomain::Complex;
    }
    throw std::invalid_argument("domain must be 'real' or 'complex'");
}

std::vector<eml_sr::DataPoint> make_points(const std::vector<double>& x, const std::vector<double>& y) {
    if (x.size() != y.size()) {
        throw std::runtime_error("x and y must have the same length");
    }
    if (x.empty()) {
        throw std::runtime_error("empty input");
    }
    std::vector<eml_sr::DataPoint> out;
    out.reserve(x.size());
    for (std::size_t i = 0; i < x.size(); ++i) {
        out.push_back({x[i], y[i]});
    }
    return out;
}

}  // namespace

PYBIND11_MODULE(_core, m) {
    m.doc() = "EML symbolic regression core (C++ engine)";

    py::class_<eml_sr::SearchStats>(m, "SearchStats")
        .def_readonly("forms_seen", &eml_sr::SearchStats::forms_seen)
        .def_readonly("candidates_evaled", &eml_sr::SearchStats::candidates_evaled)
        .def_readonly("eml_calls", &eml_sr::SearchStats::eml_calls)
        .def_readonly("cache_hits", &eml_sr::SearchStats::cache_hits)
        .def_readonly("cache_misses", &eml_sr::SearchStats::cache_misses)
        .def_readonly("best_update_count", &eml_sr::SearchStats::best_update_count);

    py::class_<eml_sr::SearchResult>(m, "FitResult")
        .def_readonly("mse", &eml_sr::SearchResult::mse)
        .def_readonly("rpn", &eml_sr::SearchResult::rpn)
        .def_property_readonly("eml_nodes", [](const eml_sr::SearchResult& r) {
            return static_cast<int>(r.tree.eml_count());
        })
        .def_readonly("stats", &eml_sr::SearchResult::stats)
        .def("predict", [](const eml_sr::SearchResult& r, double x) {
            return eml_sr::predict_tree(r.tree, x);
        })
        .def("predict_many", [](const eml_sr::SearchResult& r, const std::vector<double>& xs) {
            std::vector<double> out;
            out.reserve(xs.size());
            for (const double x : xs) {
                out.push_back(eml_sr::predict_tree(r.tree, x));
            }
            return out;
        })
        .def("to_dot", [](const eml_sr::SearchResult& r) {
            return eml_sr::tree_to_dot(r.tree);
        });

    m.def(
        "fit",
        [](const std::vector<double>& x,
           const std::vector<double>& y,
           int max_depth,
           const std::string& method,
           const std::string& domain,
           int jobs,
           bool profile,
           int epochs,
           double lr) {
            eml_sr::FitConfig config{};
            config.max_depth = max_depth;
            config.method = parse_method(method);
            config.domain = parse_domain(domain);
            config.jobs = jobs;
            config.profile = profile;
            config.adam_epochs = epochs;
            config.adam_lr = lr;
            return eml_sr::fit_api(make_points(x, y), config);
        },
        py::arg("x"),
        py::arg("y"),
        py::arg("max_depth") = 3,
        py::arg("method") = "enumerate",
        py::arg("domain") = "complex",
        py::arg("jobs") = 1,
        py::arg("profile") = false,
        py::arg("epochs") = 2000,
        py::arg("lr") = 0.05);

    m.def(
        "eml",
        [](double x, double y) {
            const auto z = eml_sr::eml({x, 0.0}, {y, 0.0});
            return std::real(z);
        },
        py::arg("x"),
        py::arg("y"));
}
