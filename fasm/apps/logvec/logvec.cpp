#include "logbus_client.hpp"
#include "bench_util.hpp"

#include <cstring>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <string>
#include <vector>

namespace {

enum class LogbusRecordResult { Ok, Eof, BadCrc };

LogbusRecordResult readLogbusRecord(std::ifstream& in, std::vector<std::uint8_t>& payload_out) {
    std::array<std::uint8_t, 8> hdr{};
    in.read(reinterpret_cast<char*>(hdr.data()), static_cast<std::streamsize>(hdr.size()));
    const auto hn = static_cast<std::size_t>(in.gcount());
    if (hn == 0) {
        return LogbusRecordResult::Eof;
    }
    if (hn != hdr.size()) {
        throw std::runtime_error("TruncatedRecord");
    }
    const std::uint32_t plen = logvec::readU32Le(hdr.data());
    const std::uint32_t want_crc = logvec::readU32Le(hdr.data() + 4);
    payload_out.resize(plen);
    if (plen > 0) {
        in.read(reinterpret_cast<char*>(payload_out.data()), static_cast<std::streamsize>(plen));
        if (static_cast<std::size_t>(in.gcount()) != plen) {
            throw std::runtime_error("TruncatedRecord");
        }
    }
    if (logvec::crc32c(payload_out) != want_crc) {
        return LogbusRecordResult::BadCrc;
    }
    return LogbusRecordResult::Ok;
}

void buildFromPayloadDir(const std::filesystem::path& dir_path, logvec::IndexBuilder& builder) {
    std::vector<std::string> names;
    for (const auto& entry : std::filesystem::directory_iterator(dir_path)) {
        if (!entry.is_regular_file()) {
            continue;
        }
        names.push_back(entry.path().filename().string());
    }
    std::sort(names.begin(), names.end());
    std::uint64_t i = 0;
    for (const auto& name : names) {
        logvec::IngestRecord rec{};
        rec.topic_record_offset = i;
        rec.payload = logvec::readWholeFile(dir_path / name, 16 * 1024 * 1024);
        builder.append(rec);
        ++i;
    }
}

void buildFromDirLogPayloads(
    const std::filesystem::path& root,
    const std::string& topic,
    logvec::IndexBuilder& builder) {
    const auto dir_path = root / "topics" / topic;
    std::vector<std::string> logs;
    for (const auto& entry : std::filesystem::directory_iterator(dir_path)) {
        if (!entry.is_regular_file()) {
            continue;
        }
        const auto name = entry.path().filename().string();
        if (name.size() >= 4 && name.compare(name.size() - 4, 4, ".log") == 0) {
            logs.push_back(name);
        }
    }
    std::sort(logs.begin(), logs.end());
    std::vector<std::uint8_t> payload_buf;
    std::uint64_t topic_record_offset = 0;
    for (const auto& name : logs) {
        std::ifstream file(dir_path / name, std::ios::binary);
        if (!file) {
            throw std::runtime_error("failed to open log file: " + name);
        }
        while (true) {
            switch (readLogbusRecord(file, payload_buf)) {
            case LogbusRecordResult::Ok:
                builder.append({topic_record_offset, payload_buf});
                ++topic_record_offset;
                break;
            case LogbusRecordResult::Eof:
                goto next_log;
            case LogbusRecordResult::BadCrc:
                throw std::runtime_error("BadCrc");
            }
        }
    next_log:;
    }
}

void usage() {
    std::cerr
        << "usage:\n"
        << "  logvec search --index PATH --query PATH --top K\n"
        << "  logvec bench --index PATH --query PATH [--top K] [--iters N]\n"
        << "    [--layer dot|topk|search|io] [--simd auto|scalar|avx2]\n"
        << "    [--cold] [--threads N] [--breakdown]\n"
        << "  logvec build-index --payload-dir DIR --out PATH\n"
        << "  logvec build-index --host H --port P --topic TOPIC --out PATH\n"
        << "  logvec build-index --dir DATA --topic TOPIC --out PATH\n";
}

int runSearch(int argc, char** argv) {
    std::filesystem::path index_path;
    std::filesystem::path query_path;
    std::uint32_t top_k = 5;
    for (int i = 2; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--index" && i + 1 < argc) {
            index_path = argv[++i];
        } else if (arg == "--query" && i + 1 < argc) {
            query_path = argv[++i];
        } else if (arg == "--top" && i + 1 < argc) {
            top_k = static_cast<std::uint32_t>(std::stoul(argv[++i]));
        } else {
            throw std::runtime_error("Usage");
        }
    }
    if (index_path.empty() || query_path.empty()) {
        throw std::runtime_error("Usage");
    }
    const logvec::VectorIndex index = logvec::VectorIndex::load(index_path);
    const auto qbytes = logvec::readWholeFile(query_path, 64 * 1024);
    if (qbytes.size() != static_cast<std::size_t>(index.dim()) * 4) {
        throw std::runtime_error("QueryDimMismatch");
    }
    const auto* qptr = reinterpret_cast<const float*>(qbytes.data());
    const std::span<const float> query{qptr, index.dim()};
    const std::vector<logvec::SearchHit> hits = index.search(query, top_k);
    for (const auto& hit : hits) {
        std::cout << hit.doc_id << ' ' << std::fixed << std::setprecision(6) << hit.score << '\n';
    }
    return 0;
}

enum class BenchLayer {
    Dot,
    Topk,
    Search,
    Io,
};

BenchLayer parseLayer(const std::string& s) {
    if (s == "dot") {
        return BenchLayer::Dot;
    }
    if (s == "topk") {
        return BenchLayer::Topk;
    }
    if (s == "io") {
        return BenchLayer::Io;
    }
    return BenchLayer::Search;
}

logvec::SimdMode parseSimd(const std::string& s) {
    if (s == "scalar") {
        return logvec::SimdMode::Scalar;
    }
    if (s == "avx2") {
        return logvec::SimdMode::Avx2;
    }
    return logvec::SimdMode::Auto;
}

std::span<const float> loadQuery(
    const std::filesystem::path& query_path,
    std::uint32_t dim,
    std::vector<float>& storage) {
    const auto qbytes = logvec::readWholeFile(query_path, 64 * 1024);
    if (qbytes.size() != static_cast<std::size_t>(dim) * 4) {
        throw std::runtime_error("QueryDimMismatch");
    }
    storage.resize(dim);
    std::memcpy(storage.data(), qbytes.data(), qbytes.size());
    return {storage.data(), dim};
}

int runBench(int argc, char** argv) {
    std::filesystem::path index_path;
    std::filesystem::path query_path;
    std::uint32_t top_k = 8;
    std::uint32_t iters = 50;
    std::uint32_t threads = 1;
    BenchLayer layer = BenchLayer::Search;
    logvec::SimdMode simd = logvec::SimdMode::Auto;
    bool cold = false;
    bool breakdown = false;

    for (int i = 2; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--index" && i + 1 < argc) {
            index_path = argv[++i];
        } else if (arg == "--query" && i + 1 < argc) {
            query_path = argv[++i];
        } else if (arg == "--top" && i + 1 < argc) {
            top_k = static_cast<std::uint32_t>(std::stoul(argv[++i]));
        } else if (arg == "--iters" && i + 1 < argc) {
            iters = static_cast<std::uint32_t>(std::stoul(argv[++i]));
        } else if (arg == "--layer" && i + 1 < argc) {
            layer = parseLayer(argv[++i]);
        } else if (arg == "--simd" && i + 1 < argc) {
            simd = parseSimd(argv[++i]);
        } else if (arg == "--threads" && i + 1 < argc) {
            threads = static_cast<std::uint32_t>(std::stoul(argv[++i]));
        } else if (arg == "--cold") {
            cold = true;
        } else if (arg == "--breakdown") {
            breakdown = true;
        } else {
            throw std::runtime_error("Usage");
        }
    }
    if (index_path.empty() || query_path.empty() || top_k == 0 || iters == 0 || threads == 0 || threads > 4) {
        throw std::runtime_error("Usage");
    }
    if (layer == BenchLayer::Io && query_path.empty()) {
        (void)query_path;
    }

    logvec::applySimdMode(simd);

    if (layer == BenchLayer::Io) {
        const logvec::VectorIndex meta = logvec::VectorIndex::load(index_path);
        const auto result = logvec::runTimed(
            meta.dim(),
            meta.count(),
            "io",
            0,
            iters,
            [&]() { (void)logvec::VectorIndex::load(index_path); });
        logvec::BenchResult line = result;
        line.extra = std::string("cold=") + (cold ? "1" : "0");
        logvec::printBenchLine(line);
        return 0;
    }

    std::vector<float> query_storage;
    logvec::VectorIndex index = logvec::VectorIndex::load(index_path);
    const auto query = loadQuery(query_path, index.dim(), query_storage);

    const auto dot_fn = [&](const float* a, const float* b, std::uint64_t len) -> float {
        switch (simd) {
        case logvec::SimdMode::Scalar:
            return logvec::lb_vec_dot_f32_scalar(a, b, len);
        case logvec::SimdMode::Avx2:
            return logvec::lb_vec_dot_f32_avx2(a, b, len);
        case logvec::SimdMode::Auto:
            return logvec::lb_vec_dot_f32(a, b, len);
        }
        return logvec::lb_vec_dot_f32(a, b, len);
    };

    if (layer == BenchLayer::Dot) {
        const auto result = logvec::runTimed(
            index.dim(),
            index.count(),
            "dot",
            5,
            iters,
            [&]() {
                if (cold) {
                    index = logvec::VectorIndex::load(index_path);
                }
                (void)index.benchDotScan(query, dot_fn);
            });
        logvec::BenchResult line = result;
        line.extra = std::string("simd=") + (simd == logvec::SimdMode::Scalar   ? "scalar"
                                             : simd == logvec::SimdMode::Avx2 ? "avx2"
                                                                              : "auto")
                     + " avx2=" + std::to_string(logvec::lb_vec_has_avx2());
        logvec::printBenchLine(line);
        return 0;
    }

    if (layer == BenchLayer::Topk) {
        std::vector<std::uint32_t> idx;
        std::vector<float> scores;
        const auto result = logvec::runTimed(
            index.dim(),
            index.count(),
            "topk",
            5,
            iters,
            [&]() {
                if (cold) {
                    index = logvec::VectorIndex::load(index_path);
                }
                if (index.benchTopk(query, top_k, threads, idx, scores) != 0) {
                    throw std::runtime_error("BadIndexNorm");
                }
            });
        logvec::BenchResult line = result;
        line.extra = "top=" + std::to_string(top_k) + " threads=" + std::to_string(threads);
        logvec::printBenchLine(line);
        return 0;
    }

    if (breakdown) {
        logvec::SearchBreakdown total{};
        std::vector<double> qnorm_samples;
        std::vector<double> topk_samples;
        std::vector<double> resolve_samples;
        qnorm_samples.reserve(iters);
        topk_samples.reserve(iters);
        resolve_samples.reserve(iters);
        for (std::uint32_t w = 0; w < 5; ++w) {
            if (cold) {
                index = logvec::VectorIndex::load(index_path);
            }
            (void)index.searchWithBreakdown(query, top_k, threads);
        }
        for (std::uint32_t i = 0; i < iters; ++i) {
            if (cold) {
                index = logvec::VectorIndex::load(index_path);
            }
            const auto [hits, parts] = index.searchWithBreakdown(query, top_k, threads);
            (void)hits;
            qnorm_samples.push_back(parts.qnorm_ms);
            topk_samples.push_back(parts.topk_ms);
            resolve_samples.push_back(parts.resolve_ms);
        }
        logvec::BenchResult line{};
        line.layer = "search";
        line.count = static_cast<std::size_t>(index.count());
        line.dim = index.dim();
        line.median_ms = logvec::medianMs(qnorm_samples) + logvec::medianMs(topk_samples)
                         + logvec::medianMs(resolve_samples);
        line.min_ms = qnorm_samples.front() + topk_samples.front() + resolve_samples.front();
        line.extra = "top=" + std::to_string(top_k) + " threads=" + std::to_string(threads)
                     + " qnorm_ms=" + std::to_string(logvec::medianMs(qnorm_samples))
                     + " topk_ms=" + std::to_string(logvec::medianMs(topk_samples))
                     + " resolve_ms=" + std::to_string(logvec::medianMs(resolve_samples))
                     + " cold=" + (cold ? "1" : "0");
        logvec::printBenchLine(line);
        return 0;
    }

    const auto result = logvec::runTimed(
        index.dim(),
        index.count(),
        "search",
        5,
        iters,
        [&]() {
            if (cold) {
                index = logvec::VectorIndex::load(index_path);
            }
            (void)index.search(query, top_k, threads);
        });
    logvec::BenchResult line = result;
    line.extra = "top=" + std::to_string(top_k) + " threads=" + std::to_string(threads)
                 + " avx2=" + std::to_string(logvec::lb_vec_has_avx2()) + " cold=" + (cold ? "1" : "0");
    logvec::printBenchLine(line);
    return 0;
}

int runBuildIndex(int argc, char** argv) {
    std::filesystem::path out_path;
    std::filesystem::path payload_dir;
    std::string host;
    std::uint16_t port = 9092;
    std::filesystem::path data_dir;
    std::string topic;
    for (int i = 2; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--out" && i + 1 < argc) {
            out_path = argv[++i];
        } else if (arg == "--payload-dir" && i + 1 < argc) {
            payload_dir = argv[++i];
        } else if (arg == "--host" && i + 1 < argc) {
            host = argv[++i];
        } else if (arg == "--port" && i + 1 < argc) {
            port = static_cast<std::uint16_t>(std::stoul(argv[++i]));
        } else if (arg == "--dir" && i + 1 < argc) {
            data_dir = argv[++i];
        } else if (arg == "--topic" && i + 1 < argc) {
            topic = argv[++i];
        } else {
            throw std::runtime_error("Usage");
        }
    }
    if (out_path.empty()) {
        throw std::runtime_error("Usage");
    }

    logvec::IndexBuilder builder;
    builder.open(out_path);
    if (!payload_dir.empty()) {
        buildFromPayloadDir(payload_dir, builder);
    } else if (!host.empty() && !topic.empty()) {
        logvec::LogbusClient client(host, port);
        client.fetchAndConsume(topic, [&](logvec::IngestRecord rec) {
            builder.append(rec);
        });
    } else if (!data_dir.empty() && !topic.empty()) {
        buildFromDirLogPayloads(data_dir, topic, builder);
    } else {
        throw std::runtime_error("Usage");
    }
    builder.finalize();
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
        if (cmd == "search") {
            return runSearch(argc, argv);
        }
        if (cmd == "bench") {
            return runBench(argc, argv);
        }
        if (cmd == "build-index") {
            return runBuildIndex(argc, argv);
        }
        usage();
        return 1;
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
