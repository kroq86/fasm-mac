#include "../../apps/eml_sr/pool_search.hpp"
#include "../../apps/eml_sr/arena_search.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <optional>
#include <string>
#include <vector>

#include <sys/resource.h>

namespace {

struct CaseDef {
    std::string name;
    std::vector<eml_sr::DataPoint> data;
    int depth{};
};

struct BenchRow {
    double ms{};
    eml_sr::SearchResult result;
};

std::vector<eml_sr::DataPoint> target_exp() {
    return {
        {0.1, std::exp(0.1)},
        {0.5, std::exp(0.5)},
        {1.0, std::exp(1.0)},
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

std::vector<eml_sr::DataPoint> target_poly() {
    return {
        {1.0, 2.0},
        {2.0, 5.0},
        {3.0, 10.0},
        {4.0, 17.0},
    };
}

template <typename Fn>
BenchRow time_search(Fn&& fn, const std::vector<eml_sr::DataPoint>& data, int depth) {
    eml_sr::SearchOptions opts{};
    opts.profile = true;
    const auto start = std::chrono::steady_clock::now();
    eml_sr::SearchResult result = fn(data, depth, opts, 0.0);
    const auto end = std::chrono::steady_clock::now();
    const double ms = std::chrono::duration<double, std::milli>(end - start).count();
    return BenchRow{ms, std::move(result)};
}

BenchRow best_of(std::vector<BenchRow> rows) {
    return *std::min_element(rows.begin(), rows.end(), [](const BenchRow& a, const BenchRow& b) {
        return a.ms < b.ms;
    });
}

void print_row(const std::string& case_name, const std::string& method, const BenchRow& row) {
    const eml_sr::SearchStats& s = row.result.stats;
    std::cout << "case=" << case_name
              << " method=" << method
              << " ms=" << row.ms
              << " mse=" << row.result.mse
              << " eml_nodes=" << row.result.tree.eml_count()
              << " forms=" << s.forms_seen
              << " candidates=" << s.candidates_evaled
              << " eml_calls=" << s.eml_calls
              << " pruned=" << s.pruned_total()
              << " cache_hit=" << s.cache_hits
              << " cache_miss=" << s.cache_misses
              << " rpn=" << row.result.rpn
              << '\n';
}

}  // namespace

int main(int argc, char** argv) {
    int repeats = 3;
    bool include_poly = false;
    std::string method_filter = "both";
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--repeats" && i + 1 < argc) {
            repeats = std::atoi(argv[++i]);
        } else if (arg == "--poly") {
            include_poly = true;
        } else if (arg == "--method" && i + 1 < argc) {
            method_filter = argv[++i];
            if (method_filter != "both" && method_filter != "legacy" && method_filter != "arena") {
                std::cerr << "bad method filter\n";
                return 1;
            }
        } else {
            std::cerr << "usage: arena_compare_bench [--repeats N] [--poly] [--method legacy|arena|both]\n";
            return 1;
        }
    }
    if (repeats < 1) {
        repeats = 1;
    }

    std::vector<CaseDef> cases{
        {"exp_d1", target_exp(), 1},
        {"ln_d3", target_ln(), 3},
    };
    if (include_poly) {
        cases.push_back({"poly_d4", target_poly(), 4});
    }

    for (const CaseDef& c : cases) {
        std::vector<BenchRow> legacy_rows;
        std::vector<BenchRow> arena_rows;
        legacy_rows.reserve(static_cast<std::size_t>(repeats));
        arena_rows.reserve(static_cast<std::size_t>(repeats));
        for (int i = 0; i < repeats; ++i) {
            if (method_filter == "both" || method_filter == "legacy") {
                legacy_rows.push_back(time_search(eml_sr::search_best_legacy, c.data, c.depth));
            }
            if (method_filter == "both" || method_filter == "arena") {
                arena_rows.push_back(time_search(eml_sr::arena_search_best, c.data, c.depth));
            }
        }
        std::optional<BenchRow> legacy;
        std::optional<BenchRow> arena;
        if (!legacy_rows.empty()) {
            legacy = best_of(std::move(legacy_rows));
            print_row(c.name, "legacy", *legacy);
        }
        if (!arena_rows.empty()) {
            arena = best_of(std::move(arena_rows));
            print_row(c.name, "arena", *arena);
        }
        if (legacy && arena) {
            std::cout << "case=" << c.name
                      << " arena_vs_legacy=" << (arena->ms / legacy->ms)
                      << '\n';
        }
    }
    rusage usage{};
    if (getrusage(RUSAGE_SELF, &usage) == 0) {
        std::cout << "process_maxrss=" << usage.ru_maxrss << '\n';
    }
    return 0;
}
