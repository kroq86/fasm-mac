#include "search.hpp"
#include "pool_search.hpp"
#include "arena_search.hpp"
#include "adam.hpp"

#include <cmath>
#include <cctype>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

struct BenchLatencyRow {
    std::uint64_t count{};
    std::string layer;
    double ms{};
};

struct CliOptions {
    eml_sr::SearchOptions search{};
    bool print_profile{false};
};

void usage() {
    std::cerr
        << "usage:\n"
        << "  eml_sr verify\n"
        << "  eml_sr show --preset exp|ln [--dot-eval X] [--dot PATH]\n"
        << "  eml_sr recover --target exp|ln|poly [--max-depth N] [--method enumerate|legacy-enumerate|adam]\n"
        << "      [--domain real|complex] [--jobs N] [--profile] [--epochs N] [--dot PATH]\n"
        << "  eml_sr fit-bench [--max-depth N] [--method enumerate|legacy-enumerate|adam] [--domain real|complex]\n"
        << "      [--jobs N] [--profile] [--epochs N] [--dot PATH] [--points PATH]\n"
        << "    stdin: optional x y rows and/or bench_perf lines (count layer ms)\n"
        << "\n"
        << "  exp = minimal tree (1 eml node).  ln = paper depth-3 tree (3 eml nodes).\n"
        << "  poly = search result (~90s at depth 4).  Use show --preset ln for full demo.\n";
}

std::string trim_copy(std::string s) {
    while (!s.empty() && std::isspace(static_cast<unsigned char>(s.front()))) {
        s.erase(s.begin());
    }
    while (!s.empty() && std::isspace(static_cast<unsigned char>(s.back()))) {
        s.pop_back();
    }
    return s;
}

bool try_parse_bench_line(const std::string& line, BenchLatencyRow& out) {
    std::istringstream ss(line);
    std::uint64_t count = 0;
    std::string layer;
    double ms = 0.0;
    if (!(ss >> count >> layer >> ms)) {
        return false;
    }
    if (count == 0 || layer.empty() || !std::isfinite(ms)) {
        return false;
    }
    out.count = count;
    out.layer = layer;
    out.ms = ms;
    return true;
}

std::vector<eml_sr::DataPoint> target_exp() {
    return {
        {0.1, std::exp(0.1)},
        {0.5, std::exp(0.5)},
        {1.0, std::exp(1.0)},
    };
}

std::vector<eml_sr::DataPoint> target_poly() {
    return {
        {1.0, 2.0},
        {2.0, 5.0},
        {3.0, 10.0},
        {4.0, 17.0},
    };
}

std::vector<eml_sr::DataPoint> target_ln() {
    return {
        {1.0, 0.0},
        {2.0, std::log(2.0)},
        {3.0, std::log(3.0)},
        {std::exp(1.0), 1.0},
    };
}

void write_dot_file(const std::string& path, const eml_sr::Tree& tree, const eml_sr::DotOptions& opts) {
    const std::string dot = eml_sr::to_dot(tree, opts);
    if (path == "-") {
        std::cout << dot;
        return;
    }
    std::ofstream out(path);
    if (!out) {
        throw std::runtime_error("failed to write dot: " + path);
    }
    out << dot;
}

void print_result(const eml_sr::SearchResult& result, bool profile) {
    std::cout << "mse=" << result.mse << '\n';
    std::cout << "eml_nodes=" << result.tree.eml_count() << '\n';
    std::cout << "rpn=" << result.rpn << '\n';
    if (profile) {
        result.stats.print(std::cout, result.mse);
    }
}

CliOptions parse_common_flags(int argc, char** argv, int start_idx) {
    CliOptions out{};
    for (int i = start_idx; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--method" && i + 1 < argc) {
            const std::string method = argv[++i];
            if (method == "adam") {
                out.search.method = eml_sr::SearchMethod::Adam;
            } else if (method == "enumerate") {
                out.search.method = eml_sr::SearchMethod::Enumerate;
            } else if (method == "legacy-enumerate") {
                out.search.method = eml_sr::SearchMethod::LegacyEnumerate;
            } else {
                throw std::runtime_error("BadMethod");
            }
        } else if (arg == "--domain" && i + 1 < argc) {
            const std::string domain = argv[++i];
            if (domain == "real") {
                out.search.domain = eml_sr::EvalDomain::Real;
            } else if (domain == "complex") {
                out.search.domain = eml_sr::EvalDomain::Complex;
            } else {
                throw std::runtime_error("BadDomain");
            }
        } else if (arg == "--jobs" && i + 1 < argc) {
            out.search.jobs = std::stoi(argv[++i]);
        } else if (arg == "--epochs" && i + 1 < argc) {
            out.search.adam_epochs = std::stoi(argv[++i]);
        } else if (arg == "--profile") {
            out.print_profile = true;
            out.search.profile = true;
        } else if (arg == "--dot" || arg == "--dot-eval" || arg == "--max-depth" || arg == "--target" ||
                   arg == "--points" || arg == "--preset") {
            continue;
        }
    }
    return out;
}

int run_verify() {
    const double xs[] = {0.1, 0.5, 1.0, 2.0};
    for (const double x : xs) {
        const auto got = eml_sr::eml({x, 0.0}, {1.0, 0.0});
        const double expected = std::exp(x);
        if (!eml_sr::near(std::real(got), expected, 1e-9)) {
            throw std::runtime_error("VerifyEmlFailed");
        }
    }
    const eml_sr::Tree ln_tree = eml_sr::preset_ln_tree();
    const double ln_xs[] = {1.0, 2.0, 3.0, std::exp(1.0)};
    for (const double x : ln_xs) {
        const double got = eml_sr::eval_real(ln_tree, x);
        const double expected = std::log(x);
        if (!eml_sr::near(got, expected, 1e-9)) {
            throw std::runtime_error("VerifyLnWitnessFailed");
        }
    }
    if (std::isfinite(eml_sr::eml_real(0.0, -1.0))) {
        throw std::runtime_error("VerifyRealDomainFailed");
    }
    std::cout << "OK eml(x,1) ~= exp(x)\n";
    return 0;
}

int run_show(int argc, char** argv) {
    std::string preset;
    std::string dot_path;
    eml_sr::DotOptions dot_opts{};
    dot_opts.annotate_values = true;
    dot_opts.eval_x = 2.0;

    for (int i = 2; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--preset" && i + 1 < argc) {
            preset = argv[++i];
        } else if (arg == "--dot-eval" && i + 1 < argc) {
            dot_opts.eval_x = std::stod(argv[++i]);
            dot_opts.annotate_values = true;
        } else if (arg == "--dot" && i + 1 < argc) {
            dot_path = argv[++i];
        } else {
            throw std::runtime_error("Usage");
        }
    }
    if (preset.empty()) {
        throw std::runtime_error("Usage");
    }

    eml_sr::Tree tree{};
    std::string name;
    if (preset == "exp") {
        tree = eml_sr::preset_exp_tree();
        name = "exp(x)=eml(x,1)";
    } else if (preset == "ln") {
        tree = eml_sr::preset_ln_tree();
        name = "ln(x)=eml(1,eml(eml(1,x),1))";
    } else {
        throw std::runtime_error("BadPreset");
    }

    std::cout << "preset=" << preset << '\n';
    std::cout << "formula=" << name << '\n';
    std::cout << "eml_nodes=" << tree.eml_count() << '\n';
    std::cout << "rpn=" << eml_sr::to_rpn(tree) << '\n';
    std::cout << "eval@" << dot_opts.eval_x << '=' << eml_sr::eval_real(tree, dot_opts.eval_x) << '\n';
    if (!dot_path.empty()) {
        write_dot_file(dot_path, tree, dot_opts);
    }
    return 0;
}

int run_recover(int argc, char** argv) {
    std::string target;
    std::string dot_path;
    int max_depth = 3;
    eml_sr::DotOptions dot_opts{};
    CliOptions cli{};

    for (int i = 2; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--target" && i + 1 < argc) {
            target = argv[++i];
        } else if (arg == "--max-depth" && i + 1 < argc) {
            max_depth = std::stoi(argv[++i]);
        } else if (arg == "--dot-eval" && i + 1 < argc) {
            dot_opts.eval_x = std::stod(argv[++i]);
            dot_opts.annotate_values = true;
        } else if (arg == "--dot" && i + 1 < argc) {
            dot_path = argv[++i];
        } else if (arg == "--method" && i + 1 < argc) {
            const std::string method = argv[++i];
            if (method == "adam") {
                cli.search.method = eml_sr::SearchMethod::Adam;
            } else if (method == "enumerate") {
                cli.search.method = eml_sr::SearchMethod::Enumerate;
            } else if (method == "legacy-enumerate") {
                cli.search.method = eml_sr::SearchMethod::LegacyEnumerate;
            } else {
                throw std::runtime_error("BadMethod");
            }
        } else if (arg == "--domain" && i + 1 < argc) {
            const std::string domain = argv[++i];
            if (domain == "real") {
                cli.search.domain = eml_sr::EvalDomain::Real;
            } else if (domain == "complex") {
                cli.search.domain = eml_sr::EvalDomain::Complex;
            } else {
                throw std::runtime_error("BadDomain");
            }
        } else if (arg == "--jobs" && i + 1 < argc) {
            cli.search.jobs = std::stoi(argv[++i]);
        } else if (arg == "--epochs" && i + 1 < argc) {
            cli.search.adam_epochs = std::stoi(argv[++i]);
        } else if (arg == "--profile") {
            cli.print_profile = true;
            cli.search.profile = true;
        } else {
            throw std::runtime_error("Usage");
        }
    }
    if (target.empty() || max_depth < 1) {
        throw std::runtime_error("Usage");
    }

    std::vector<eml_sr::DataPoint> data;
    double mse_limit = 1e-4;
    if (target == "exp") {
        data = target_exp();
        mse_limit = 1e-6;
    } else if (target == "ln") {
        data = target_ln();
        mse_limit = 1e-6;
        if (max_depth < 3) {
            max_depth = 3;
        }
    } else if (target == "poly") {
        data = target_poly();
        mse_limit = 0.2;
    } else {
        throw std::runtime_error("BadTarget");
    }

    const double search_goal = (target == "exp" || target == "ln") ? 1e-9 : 0.0;
    const eml_sr::SearchResult result = eml_sr::search_best(data, max_depth, cli.search, search_goal);
    if (!std::isfinite(result.mse) || result.mse > mse_limit) {
        throw std::runtime_error("RecoverFailed");
    }
    print_result(result, cli.print_profile);
    if (!dot_path.empty()) {
        if (!dot_opts.annotate_values && !data.empty()) {
            dot_opts.eval_x = data.front().x;
            dot_opts.annotate_values = true;
        }
        write_dot_file(dot_path, result.tree, dot_opts);
    }
    return 0;
}

std::vector<eml_sr::DataPoint> read_points(std::istream& in) {
    std::vector<eml_sr::DataPoint> out;
    double x = 0.0;
    double y = 0.0;
    while (in >> x >> y) {
        out.push_back({x, y});
    }
    if (out.empty()) {
        throw std::runtime_error("EmptyPoints");
    }
    return out;
}

struct FitInput {
    std::vector<eml_sr::DataPoint> points;
    std::vector<BenchLatencyRow> latency;
};

FitInput read_fit_input(const std::string& points_path) {
    FitInput out{};
    if (!points_path.empty()) {
        std::ifstream in(points_path);
        if (!in) {
            throw std::runtime_error("failed to open points: " + points_path);
        }
        out.points = read_points(in);
    }

    std::string line;
    while (std::getline(std::cin, line)) {
        line = trim_copy(line);
        if (line.empty()) {
            continue;
        }
        BenchLatencyRow row{};
        if (try_parse_bench_line(line, row)) {
            out.latency.push_back(row);
            continue;
        }
        std::istringstream ls(line);
        double x = 0.0;
        double y = 0.0;
        if (ls >> x >> y) {
            out.points.push_back({x, y});
        }
    }

    if (out.points.empty()) {
        if (!out.latency.empty()) {
            out.points = target_poly();
        } else if (points_path.empty()) {
            throw std::runtime_error("EmptyPoints");
        }
    }
    return out;
}

void print_latency_rows(const std::vector<BenchLatencyRow>& rows) {
    for (const BenchLatencyRow& row : rows) {
        std::cout << "latency_count=" << row.count << ' '
                  << "latency_layer=" << row.layer << ' '
                  << "latency_ms=" << row.ms << '\n';
    }
}

int run_fit_bench(int argc, char** argv) {
    std::string dot_path;
    std::string points_path;
    int max_depth = 3;
    CliOptions cli{};

    for (int i = 2; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--max-depth" && i + 1 < argc) {
            max_depth = std::stoi(argv[++i]);
        } else if (arg == "--dot" && i + 1 < argc) {
            dot_path = argv[++i];
        } else if (arg == "--points" && i + 1 < argc) {
            points_path = argv[++i];
        } else if (arg == "--method" && i + 1 < argc) {
            const std::string method = argv[++i];
            if (method == "adam") {
                cli.search.method = eml_sr::SearchMethod::Adam;
            } else if (method == "enumerate") {
                cli.search.method = eml_sr::SearchMethod::Enumerate;
            } else if (method == "legacy-enumerate") {
                cli.search.method = eml_sr::SearchMethod::LegacyEnumerate;
            } else {
                throw std::runtime_error("BadMethod");
            }
        } else if (arg == "--domain" && i + 1 < argc) {
            const std::string domain = argv[++i];
            if (domain == "real") {
                cli.search.domain = eml_sr::EvalDomain::Real;
            } else if (domain == "complex") {
                cli.search.domain = eml_sr::EvalDomain::Complex;
            } else {
                throw std::runtime_error("BadDomain");
            }
        } else if (arg == "--jobs" && i + 1 < argc) {
            cli.search.jobs = std::stoi(argv[++i]);
        } else if (arg == "--epochs" && i + 1 < argc) {
            cli.search.adam_epochs = std::stoi(argv[++i]);
        } else if (arg == "--profile") {
            cli.print_profile = true;
            cli.search.profile = true;
        } else {
            throw std::runtime_error("Usage");
        }
    }

    const FitInput fit = read_fit_input(points_path);
    const std::vector<eml_sr::DataPoint>& data = fit.points;

    double mean_y = 0.0;
    for (const auto& pt : data) {
        mean_y += pt.y;
    }
    mean_y /= static_cast<double>(data.size());
    double baseline = 0.0;
    for (const auto& pt : data) {
        const double d = pt.y - mean_y;
        baseline += d * d;
    }
    baseline /= static_cast<double>(data.size());

    const eml_sr::SearchResult result = eml_sr::search_best(data, max_depth, cli.search);
    if (!std::isfinite(result.mse) || result.mse >= baseline) {
        throw std::runtime_error("FitBenchFailed");
    }
    print_result(result, cli.print_profile);
    std::cout << "baseline_mse=" << baseline << '\n';
    print_latency_rows(fit.latency);
    if (!dot_path.empty()) {
        write_dot_file(dot_path, result.tree, {});
    }
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        if (argc < 2) {
            usage();
            return 1;
        }
        const std::string cmd = argv[1];
        if (cmd == "verify") {
            return run_verify();
        }
        if (cmd == "show") {
            return run_show(argc, argv);
        }
        if (cmd == "recover") {
            return run_recover(argc, argv);
        }
        if (cmd == "fit-bench") {
            return run_fit_bench(argc, argv);
        }
        usage();
        return 1;
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
