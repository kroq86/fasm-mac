#pragma once

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

namespace logvec {

inline double medianMs(std::vector<double>& samples) {
    if (samples.empty()) {
        return 0.0;
    }
    std::sort(samples.begin(), samples.end());
    return samples[samples.size() / 2];
}

inline double elapsedMs(
    const std::chrono::steady_clock::time_point& t0,
    const std::chrono::steady_clock::time_point& t1) {
    return std::chrono::duration<double, std::milli>(t1 - t0).count();
}

struct BenchResult {
    double median_ms{};
    double min_ms{};
    std::size_t count{};
    std::uint32_t dim{};
    std::string layer;
    std::string extra;
};

inline void printBenchLine(const BenchResult& r) {
    std::cout << std::fixed << std::setprecision(3)
              << "layer=" << r.layer
              << " count=" << r.count
              << " dim=" << r.dim
              << " median_ms=" << r.median_ms
              << " min_ms=" << r.min_ms;
    if (!r.extra.empty()) {
        std::cout << ' ' << r.extra;
    }
    std::cout << '\n';
}

template <typename Fn>
BenchResult runTimed(
    std::uint32_t dim,
    std::uint64_t count,
    std::string layer,
    std::uint32_t warmup,
    std::uint32_t iters,
    Fn fn) {
    for (std::uint32_t i = 0; i < warmup; ++i) {
        fn();
    }
    std::vector<double> samples;
    samples.reserve(iters);
    for (std::uint32_t i = 0; i < iters; ++i) {
        const auto t0 = std::chrono::steady_clock::now();
        fn();
        const auto t1 = std::chrono::steady_clock::now();
        samples.push_back(elapsedMs(t0, t1));
    }
    BenchResult out{};
    out.layer = std::move(layer);
    out.count = static_cast<std::size_t>(count);
    out.dim = dim;
    out.median_ms = medianMs(samples);
    out.min_ms = samples.front();
    return out;
}

}  // namespace logvec
