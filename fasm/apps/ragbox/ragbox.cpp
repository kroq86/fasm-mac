#include "../logvec/bench_util.hpp"
#include "../logvec/vector_index.hpp"
#include "chunker.hpp"
#include "doctor.hpp"
#include "manifest.hpp"
#include "ollama_client.hpp"
#include "snippet.hpp"

#include <chrono>
#include <cstring>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace {

std::vector<std::uint8_t> makePayload(
    std::uint32_t dim,
    std::span<const float> vector,
    std::uint64_t doc_id) {
    std::vector<std::uint8_t> out;
    out.reserve(4 + dim * 4 + 8);
    logvec::writeU32Le(out, dim);
    const auto* bytes = reinterpret_cast<const std::uint8_t*>(vector.data());
    out.insert(out.end(), bytes, bytes + dim * sizeof(float));
    logvec::writeU64Le(out, doc_id);
    return out;
}

std::string jsonEscapeOut(std::string_view s) {
    std::string out;
    out.reserve(s.size() + 8);
    for (const char c : s) {
        switch (c) {
        case '\\':
            out += "\\\\";
            break;
        case '"':
            out += "\\\"";
            break;
        case '\n':
            out += "\\n";
            break;
        case '\r':
            out += "\\r";
            break;
        case '\t':
            out += "\\t";
            break;
        default:
            out += c;
            break;
        }
    }
    return out;
}

void usage() {
    std::cerr
        << "usage:\n"
        << "  ragbox build --root PATH --out PATH [--manifest PATH]\n"
        << "    [--chunk-size N] [--overlap N] [--ollama URL] [--model MODEL]\n"
        << "    [--dry-run] [--embed-text]\n"
        << "  ragbox search --index PATH --query TEXT [--top K] [--json]\n"
        << "    [--manifest PATH] [--ollama URL] [--model MODEL] [--query-file PATH]\n"
        << "    [--snippet-len N]\n"
        << "  ragbox bench --index PATH --manifest PATH --query-file PATH\n"
        << "    [--top K] [--iters N] [--snippet-len N] [--threads N] [--breakdown]\n"
        << "  ragbox doctor [--index PATH] [--manifest PATH] [--ollama URL] [--model MODEL]\n"
        << "    [--skip-ollama]\n";
}

int runBuild(int argc, char** argv) {
    std::filesystem::path root;
    std::filesystem::path out_path;
    std::filesystem::path manifest_path;
    std::size_t chunk_size = 800;
    std::size_t overlap = 100;
    std::string ollama_url = "http://127.0.0.1:11434";
    std::string model = "nomic-embed-text";
    bool dry_run = false;
    bool embed_text = false;

    for (int i = 2; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--root" && i + 1 < argc) {
            root = argv[++i];
        } else if (arg == "--out" && i + 1 < argc) {
            out_path = argv[++i];
        } else if (arg == "--manifest" && i + 1 < argc) {
            manifest_path = argv[++i];
        } else if (arg == "--chunk-size" && i + 1 < argc) {
            chunk_size = static_cast<std::size_t>(std::stoull(argv[++i]));
        } else if (arg == "--overlap" && i + 1 < argc) {
            overlap = static_cast<std::size_t>(std::stoull(argv[++i]));
        } else if (arg == "--ollama" && i + 1 < argc) {
            ollama_url = argv[++i];
        } else if (arg == "--model" && i + 1 < argc) {
            model = argv[++i];
        } else if (arg == "--dry-run") {
            dry_run = true;
        } else if (arg == "--embed-text") {
            embed_text = true;
        } else {
            throw std::runtime_error("Usage");
        }
    }
    if (root.empty() || out_path.empty()) {
        throw std::runtime_error("Usage");
    }
    if (manifest_path.empty()) {
        manifest_path = ragbox::defaultManifestPath(out_path);
    }

    const std::vector<ragbox::Chunk> chunks =
        ragbox::chunkDirectory(root, chunk_size, overlap);
    if (chunks.empty()) {
        throw std::runtime_error("EmptyInput");
    }

    if (dry_run) {
        const ragbox::Manifest manifest = ragbox::manifestFromChunks(
            chunks, 0, model, chunk_size, overlap, root);
        ragbox::writeManifest(manifest_path, manifest, embed_text);
        std::cout << "dry-run chunks=" << chunks.size() << " manifest=" << manifest_path << '\n';
        return 0;
    }

    logvec::IndexBuilder builder;
    builder.open(out_path);
    std::uint32_t dim = 0;
    bool has_dim = false;
    for (const auto& chunk : chunks) {
        const std::vector<float> embedding = ragbox::ollamaEmbed(ollama_url, model, chunk.text);
        if (!has_dim) {
            dim = static_cast<std::uint32_t>(embedding.size());
            has_dim = true;
        } else if (embedding.size() != dim) {
            throw std::runtime_error("DimMismatch");
        }
        logvec::IngestRecord rec{};
        rec.topic_record_offset = chunk.doc_id;
        rec.payload = makePayload(dim, embedding, chunk.doc_id);
        builder.append(rec);
    }
    builder.finalize();

    const ragbox::Manifest manifest =
        ragbox::manifestFromChunks(chunks, dim, model, chunk_size, overlap, root);
    ragbox::writeManifest(manifest_path, manifest, embed_text);
    std::cout << "built index=" << out_path << " chunks=" << chunks.size() << " dim=" << dim << '\n';
    return 0;
}

int runSearch(int argc, char** argv) {
    std::filesystem::path index_path;
    std::filesystem::path manifest_path;
    std::filesystem::path query_file;
    std::string query_text;
    std::string ollama_url = "http://127.0.0.1:11434";
    std::string model = "nomic-embed-text";
    std::uint32_t top_k = 8;
    bool json_out = false;
    std::size_t snippet_len = 200;

    for (int i = 2; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--index" && i + 1 < argc) {
            index_path = argv[++i];
        } else if (arg == "--manifest" && i + 1 < argc) {
            manifest_path = argv[++i];
        } else if (arg == "--query" && i + 1 < argc) {
            query_text = argv[++i];
        } else if (arg == "--query-file" && i + 1 < argc) {
            query_file = argv[++i];
        } else if (arg == "--ollama" && i + 1 < argc) {
            ollama_url = argv[++i];
        } else if (arg == "--model" && i + 1 < argc) {
            model = argv[++i];
        } else if (arg == "--top" && i + 1 < argc) {
            top_k = static_cast<std::uint32_t>(std::stoul(argv[++i]));
        } else if (arg == "--json") {
            json_out = true;
        } else if (arg == "--snippet-len" && i + 1 < argc) {
            snippet_len = static_cast<std::size_t>(std::stoull(argv[++i]));
        } else {
            throw std::runtime_error("Usage");
        }
    }
    if (index_path.empty() || (query_text.empty() && query_file.empty()) || top_k == 0) {
        throw std::runtime_error("Usage");
    }
    if (manifest_path.empty()) {
        manifest_path = ragbox::defaultManifestPath(index_path);
    }

    const logvec::VectorIndex index = logvec::VectorIndex::load(index_path);
    const ragbox::Manifest manifest = ragbox::loadManifest(manifest_path);
    ragbox::validateManifestIndex(manifest, index.dim(), index.count());

    std::vector<float> query_vec;
    if (!query_file.empty()) {
        const auto qbytes = logvec::readWholeFile(query_file, 64 * 1024);
        if (qbytes.size() != static_cast<std::size_t>(index.dim()) * 4) {
            throw std::runtime_error("QueryDimMismatch");
        }
        query_vec.resize(index.dim());
        std::memcpy(query_vec.data(), qbytes.data(), qbytes.size());
    } else {
        query_vec = ragbox::ollamaEmbed(ollama_url, model, query_text);
        if (query_vec.size() != index.dim()) {
            throw std::runtime_error("QueryDimMismatch");
        }
    }

    const std::vector<logvec::SearchHit> hits = index.search(query_vec, top_k);

    if (json_out) {
        std::cout << "[\n";
        for (std::size_t i = 0; i < hits.size(); ++i) {
            const auto& hit = hits[i];
            const ragbox::ManifestRecord* rec = ragbox::findRecord(manifest, hit.doc_id);
            std::string path;
            std::size_t offset = 0;
            std::string snippet;
            if (rec != nullptr) {
                path = rec->path;
                offset = rec->offset;
                snippet = ragbox::loadSnippet(manifest, *rec, snippet_len);
            }
            std::cout << "  {"
                      << "\"doc_id\":" << hit.doc_id << ","
                      << "\"score\":" << std::fixed << std::setprecision(6) << hit.score << ","
                      << "\"path\":\"" << jsonEscapeOut(path) << "\","
                      << "\"offset\":" << offset << ","
                      << "\"snippet\":\"" << jsonEscapeOut(snippet) << "\""
                      << "}";
            if (i + 1 < hits.size()) {
                std::cout << ',';
            }
            std::cout << '\n';
        }
        std::cout << "]\n";
    } else {
        for (const auto& hit : hits) {
            const ragbox::ManifestRecord* rec = ragbox::findRecord(manifest, hit.doc_id);
            if (rec != nullptr) {
                std::cout << rec->path << ':' << rec->offset << ' ' << std::fixed << std::setprecision(6)
                          << hit.score << '\n';
            } else {
                std::cout << hit.doc_id << ' ' << std::fixed << std::setprecision(6) << hit.score << '\n';
            }
        }
    }
    return 0;
}

int runBench(int argc, char** argv) {
    std::filesystem::path index_path;
    std::filesystem::path manifest_path;
    std::filesystem::path query_file;
    std::uint32_t top_k = 8;
    std::uint32_t iters = 50;
    std::uint32_t threads = 1;
    std::size_t snippet_len = 200;
    bool breakdown = false;

    for (int i = 2; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--index" && i + 1 < argc) {
            index_path = argv[++i];
        } else if (arg == "--manifest" && i + 1 < argc) {
            manifest_path = argv[++i];
        } else if (arg == "--query-file" && i + 1 < argc) {
            query_file = argv[++i];
        } else if (arg == "--top" && i + 1 < argc) {
            top_k = static_cast<std::uint32_t>(std::stoul(argv[++i]));
        } else if (arg == "--iters" && i + 1 < argc) {
            iters = static_cast<std::uint32_t>(std::stoul(argv[++i]));
        } else if (arg == "--threads" && i + 1 < argc) {
            threads = static_cast<std::uint32_t>(std::stoul(argv[++i]));
        } else if (arg == "--snippet-len" && i + 1 < argc) {
            snippet_len = static_cast<std::size_t>(std::stoull(argv[++i]));
        } else if (arg == "--breakdown") {
            breakdown = true;
        } else {
            throw std::runtime_error("Usage");
        }
    }
    if (index_path.empty() || query_file.empty() || top_k == 0 || iters == 0 || threads == 0 || threads > 4) {
        throw std::runtime_error("Usage");
    }
    if (manifest_path.empty()) {
        manifest_path = ragbox::defaultManifestPath(index_path);
    }

    const auto t_manifest0 = std::chrono::steady_clock::now();
    const ragbox::Manifest manifest = ragbox::loadManifest(manifest_path);
    const auto t_manifest1 = std::chrono::steady_clock::now();
    const logvec::VectorIndex index = logvec::VectorIndex::load(index_path);
    ragbox::validateManifestIndex(manifest, index.dim(), index.count());

    const auto qbytes = logvec::readWholeFile(query_file, 64 * 1024);
    if (qbytes.size() != static_cast<std::size_t>(index.dim()) * 4) {
        throw std::runtime_error("QueryDimMismatch");
    }
    std::vector<float> query(index.dim());
    std::memcpy(query.data(), qbytes.data(), qbytes.size());

    auto run_once = [&]() {
        const std::vector<logvec::SearchHit> hits = index.search(query, top_k, threads);
        for (const auto& hit : hits) {
            const ragbox::ManifestRecord* rec = ragbox::findRecord(manifest, hit.doc_id);
            if (rec != nullptr) {
                (void)ragbox::loadSnippet(manifest, *rec, snippet_len);
            }
        }
    };

    if (breakdown) {
        std::vector<double> search_ms;
        std::vector<double> join_ms;
        std::vector<double> snippet_ms;
        search_ms.reserve(iters);
        join_ms.reserve(iters);
        snippet_ms.reserve(iters);
        for (std::uint32_t w = 0; w < 5; ++w) {
            run_once();
        }
        for (std::uint32_t i = 0; i < iters; ++i) {
            const auto t0 = std::chrono::steady_clock::now();
            const std::vector<logvec::SearchHit> hits = index.search(query, top_k, threads);
            const auto t1 = std::chrono::steady_clock::now();
            for (const auto& hit : hits) {
                const ragbox::ManifestRecord* rec = ragbox::findRecord(manifest, hit.doc_id);
                (void)rec;
            }
            const auto t2 = std::chrono::steady_clock::now();
            for (const auto& hit : hits) {
                const ragbox::ManifestRecord* rec = ragbox::findRecord(manifest, hit.doc_id);
                if (rec != nullptr) {
                    (void)ragbox::loadSnippet(manifest, *rec, snippet_len);
                }
            }
            const auto t3 = std::chrono::steady_clock::now();
            search_ms.push_back(logvec::elapsedMs(t0, t1));
            join_ms.push_back(logvec::elapsedMs(t1, t2));
            snippet_ms.push_back(logvec::elapsedMs(t2, t3));
        }
        logvec::BenchResult line{};
        line.layer = "ragbox";
        line.count = static_cast<std::size_t>(index.count());
        line.dim = index.dim();
        line.median_ms = logvec::medianMs(search_ms) + logvec::medianMs(join_ms) + logvec::medianMs(snippet_ms);
        line.min_ms = search_ms.front() + join_ms.front() + snippet_ms.front();
        line.extra = "top=" + std::to_string(top_k) + " threads=" + std::to_string(threads)
                     + " manifest_load_ms=" + std::to_string(logvec::elapsedMs(t_manifest0, t_manifest1))
                     + " search_ms=" + std::to_string(logvec::medianMs(search_ms))
                     + " join_ms=" + std::to_string(logvec::medianMs(join_ms))
                     + " snippet_ms=" + std::to_string(logvec::medianMs(snippet_ms));
        logvec::printBenchLine(line);
        return 0;
    }

    for (std::uint32_t w = 0; w < 5; ++w) {
        run_once();
    }
    std::vector<double> samples;
    samples.reserve(iters);
    for (std::uint32_t i = 0; i < iters; ++i) {
        const auto t0 = std::chrono::steady_clock::now();
        run_once();
        const auto t1 = std::chrono::steady_clock::now();
        samples.push_back(logvec::elapsedMs(t0, t1));
    }
    logvec::BenchResult line{};
    line.layer = "ragbox";
    line.count = static_cast<std::size_t>(index.count());
    line.dim = index.dim();
    line.median_ms = logvec::medianMs(samples);
    line.min_ms = samples.front();
    line.extra = "top=" + std::to_string(top_k) + " threads=" + std::to_string(threads)
                 + " manifest_load_ms=" + std::to_string(logvec::elapsedMs(t_manifest0, t_manifest1));
    logvec::printBenchLine(line);
    return 0;
}

int runDoctor(int argc, char** argv) {
    ragbox::DoctorOptions opts{};
    for (int i = 2; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--index" && i + 1 < argc) {
            opts.index_path = argv[++i];
        } else if (arg == "--manifest" && i + 1 < argc) {
            opts.manifest_path = argv[++i];
        } else if (arg == "--ollama" && i + 1 < argc) {
            opts.ollama_url = argv[++i];
        } else if (arg == "--model" && i + 1 < argc) {
            opts.model = argv[++i];
        } else if (arg == "--skip-ollama") {
            opts.skip_ollama = true;
        } else {
            throw std::runtime_error("Usage");
        }
    }
    if (!opts.index_path.empty() && opts.manifest_path.empty()) {
        opts.manifest_path = ragbox::defaultManifestPath(opts.index_path);
    }
    return ragbox::runDoctor(opts);
}

}  // namespace

int main(int argc, char** argv) {
    try {
        if (argc < 2) {
            usage();
            return 1;
        }
        const std::string cmd = argv[1];
        if (cmd == "build") {
            return runBuild(argc, argv);
        }
        if (cmd == "search") {
            return runSearch(argc, argv);
        }
        if (cmd == "bench") {
            return runBench(argc, argv);
        }
        if (cmd == "doctor") {
            return runDoctor(argc, argv);
        }
        usage();
        return 1;
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
