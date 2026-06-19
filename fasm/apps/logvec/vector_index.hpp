#pragma once

#include "mapped_file.hpp"
#include "vector_core.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <limits>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace logvec {

struct SearchHit {
    std::uint64_t doc_id{};
    float score{};
};

struct SearchBreakdown {
    double qnorm_ms{};
    double topk_ms{};
    double resolve_ms{};
};

enum class SimdMode {
    Auto,
    Scalar,
    Avx2,
};

inline void applySimdMode(SimdMode mode) {
    switch (mode) {
    case SimdMode::Scalar:
        lb_vec_set_simd_mode(1);
        break;
    case SimdMode::Avx2:
        lb_vec_set_simd_mode(2);
        break;
    case SimdMode::Auto:
        lb_vec_set_simd_mode(0);
        break;
    }
}

inline bool sameHits(const std::vector<SearchHit>& a, const std::vector<SearchHit>& b, float tol = 1e-5f) {
    if (a.size() != b.size()) {
        return false;
    }
    for (std::size_t i = 0; i < a.size(); ++i) {
        if (a[i].doc_id != b[i].doc_id) {
            return false;
        }
        if (std::fabs(a[i].score - b[i].score) > tol) {
            return false;
        }
    }
    return true;
}

struct IndexItem {
    std::uint64_t doc_id{};
    std::span<const float> vector;
};

struct IndexEntry {
    std::uint64_t doc_id{};
    float norm{};
    std::size_t vector_off{};
};

struct IngestRecord {
    std::uint64_t topic_record_offset{};
    std::vector<std::uint8_t> payload;
};

struct ParsedPayload {
    std::uint32_t dim{};
    std::uint64_t doc_id{};
    std::span<const float> vector;
};

inline std::uint32_t readU32Le(const std::uint8_t* p) {
    std::uint32_t v{};
    std::memcpy(&v, p, sizeof(v));
    return v;
}

inline std::uint64_t readU64Le(const std::uint8_t* p) {
    std::uint64_t v{};
    std::memcpy(&v, p, sizeof(v));
    return v;
}

inline void writeU32Le(std::vector<std::uint8_t>& out, std::uint32_t v) {
    std::uint8_t buf[4];
    std::memcpy(buf, &v, sizeof(v));
    out.insert(out.end(), buf, buf + 4);
}

inline void writeU64Le(std::vector<std::uint8_t>& out, std::uint64_t v) {
    std::uint8_t buf[8];
    std::memcpy(buf, &v, sizeof(v));
    out.insert(out.end(), buf, buf + 8);
}

inline std::vector<std::uint8_t> readWholeFile(
    const std::filesystem::path& path,
    std::size_t limit) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        throw std::runtime_error("failed to open file: " + path.string());
    }
    in.seekg(0, std::ios::end);
    const auto size = static_cast<std::size_t>(in.tellg());
    if (size > limit) {
        throw std::runtime_error("file too large: " + path.string());
    }
    in.seekg(0, std::ios::beg);
    std::vector<std::uint8_t> data(size);
    if (size > 0 && !in.read(reinterpret_cast<char*>(data.data()), static_cast<std::streamsize>(size))) {
        throw std::runtime_error("failed to read file: " + path.string());
    }
    return data;
}

inline void writeWholeFile(const std::filesystem::path& path, std::span<const std::uint8_t> data) {
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    if (!out) {
        throw std::runtime_error("failed to open file for write: " + path.string());
    }
    if (!data.empty() &&
        !out.write(reinterpret_cast<const char*>(data.data()), static_cast<std::streamsize>(data.size()))) {
        throw std::runtime_error("failed to write file: " + path.string());
    }
}

inline float normalizeUnitVector(std::span<const float> vector, std::uint32_t dim, float* unit_out) {
    const float norm = lb_vec_norm_f32(vector.data(), dim);
    if (norm == 0.0f) {
        throw std::runtime_error("ZeroNorm");
    }
    for (std::uint32_t i = 0; i < dim; ++i) {
        unit_out[i] = vector[i] / norm;
    }
    return lb_vec_norm_f32(unit_out, dim);
}

class VectorIndex {
public:
    static constexpr std::array<std::uint8_t, 8> kMagic = {'L', 'O', 'G', 'V', 'E', 'C', '1', 0};
    static constexpr std::uint32_t kVersion = 1;

    static constexpr std::size_t kMaxIndexBytes = 512 * 1024 * 1024;

    static VectorIndex load(const std::filesystem::path& path) {
        try {
            auto mapped = std::make_shared<MappedFile>(path, kMaxIndexBytes);
            const auto [dim, count] = validateHeader(mapped->bytes());
            return VectorIndex{dim, count, std::move(mapped), {}};
        } catch (const std::exception&) {
            return parse(readWholeFile(path, kMaxIndexBytes));
        }
    }

    static VectorIndex parse(std::vector<std::uint8_t> data) {
        const auto [dim, count] = validateHeader(data);
        return VectorIndex{dim, count, nullptr, std::move(data)};
    }

    static std::vector<std::uint8_t> buildBytes(std::uint32_t dim, std::span<const IndexItem> items) {
        std::vector<std::uint8_t> out;
        out.reserve(4096);
        out.insert(out.end(), kMagic.begin(), kMagic.end());
        writeU32Le(out, kVersion);
        writeU32Le(out, dim);
        writeU64Le(out, items.size());
        writeU64Le(out, 0);
        writeU64Le(out, 0);
        std::vector<float> unit(dim);
        for (const auto& item : items) {
            writeU64Le(out, item.doc_id);
            const float unit_norm = normalizeUnitVector(item.vector, dim, unit.data());
            std::uint32_t norm_bits{};
            std::memcpy(&norm_bits, &unit_norm, sizeof(norm_bits));
            writeU32Le(out, norm_bits);
            writeU32Le(out, 0);
            const auto* bytes = reinterpret_cast<const std::uint8_t*>(unit.data());
            out.insert(out.end(), bytes, bytes + dim * sizeof(float));
        }
        return out;
    }

    static void write(const std::filesystem::path& path, std::uint32_t dim, std::span<const IndexItem> items) {
        const auto bytes = buildBytes(dim, items);
        writeWholeFile(path, bytes);
    }

    [[nodiscard]] std::uint32_t dim() const { return dim_; }
    [[nodiscard]] std::uint64_t count() const { return count_; }

    [[nodiscard]] IndexEntry entry(std::uint64_t i) const {
        const std::size_t off = headerSize() + static_cast<std::size_t>(i) * recordSize(dim_);
        const auto* p = data_.data() + off;
        IndexEntry ent{};
        ent.doc_id = readU64Le(p);
        std::uint32_t norm_bits = readU32Le(p + 8);
        std::memcpy(&ent.norm, &norm_bits, sizeof(ent.norm));
        ent.vector_off = off + 16;
        return ent;
    }

    [[nodiscard]] std::span<const float> vectorSpan(const IndexEntry& ent) const {
        const auto* ptr = reinterpret_cast<const float*>(data_.data() + ent.vector_off);
        return {ptr, dim_};
    }

    [[nodiscard]] std::span<const std::uint8_t> bytes() const { return data_; }
    [[nodiscard]] const std::uint8_t* recordsPtr() const { return data_.data() + headerSize(); }
    [[nodiscard]] std::uint64_t recordStride() const { return recordSize(dim_); }

    [[nodiscard]] float benchDotScan(
        std::span<const float> query,
        const std::function<float(const float*, const float*, std::uint64_t)>& dot_fn) const {
        if (query.size() != dim_) {
            throw std::runtime_error("QueryDimMismatch");
        }
        if (count_ == 0) {
            return 0.0f;
        }
        const auto* records = recordsPtr();
        const std::uint64_t stride = recordStride();
        volatile float acc = 0.0f;
        for (std::uint64_t i = 0; i < count_; ++i) {
            const auto* rec = records + static_cast<std::size_t>(i) * stride;
            const auto* vec = reinterpret_cast<const float*>(rec + 16);
            acc += dot_fn(query.data(), vec, dim_);
        }
        return acc;
    }

    [[nodiscard]] int benchTopk(
        std::span<const float> query,
        std::size_t top_k,
        std::size_t threads,
        std::vector<std::uint32_t>& idx_out,
        std::vector<float>& score_out) const {
        if (query.size() != dim_) {
            throw std::runtime_error("QueryDimMismatch");
        }
        const float qnorm = lb_vec_norm_f32(query.data(), dim_);
        if (qnorm == 0.0f) {
            throw std::runtime_error("ZeroNorm");
        }
        if (count_ == 0) {
            return 0;
        }
        const std::uint64_t k = std::min(static_cast<std::uint64_t>(top_k), count_);
        if (threads <= 1) {
            idx_out.resize(static_cast<std::size_t>(k));
            score_out.resize(static_cast<std::size_t>(k));
            return lb_vec_topk_cosine_lv(
                query.data(),
                recordsPtr(),
                count_,
                dim_,
                k,
                recordStride(),
                idx_out.data(),
                score_out.data());
        }
        const auto partial = runTopkParallel(query, k, threads);
        idx_out = partial.first;
        score_out = partial.second;
        return 0;
    }

    [[nodiscard]] std::vector<SearchHit> search(
        std::span<const float> query,
        std::size_t top_k,
        std::size_t threads = 1) const {
        if (threads <= 1) {
            return searchSerial(query, top_k);
        }
        return searchParallel(query, top_k, threads);
    }

    [[nodiscard]] std::pair<std::vector<SearchHit>, SearchBreakdown> searchWithBreakdown(
        std::span<const float> query,
        std::size_t top_k,
        std::size_t threads = 1) const {
        SearchBreakdown breakdown{};
        if (query.size() != dim_) {
            throw std::runtime_error("QueryDimMismatch");
        }
        const auto t0 = std::chrono::steady_clock::now();
        const float qnorm = lb_vec_norm_f32(query.data(), dim_);
        volatile float touch = lb_vec_dot_f32(query.data(), query.data(), dim_);
        (void)touch;
        const auto t1 = std::chrono::steady_clock::now();
        if (qnorm == 0.0f) {
            throw std::runtime_error("ZeroNorm");
        }
        if (count_ == 0) {
            return {{}, breakdown};
        }

        const std::uint64_t k = std::min(static_cast<std::uint64_t>(top_k), count_);
        std::vector<std::uint32_t> idx_out(static_cast<std::size_t>(k));
        std::vector<float> score_out(static_cast<std::size_t>(k));

        const auto t2 = std::chrono::steady_clock::now();
        if (threads <= 1) {
            if (lb_vec_topk_cosine_lv(
                    query.data(),
                    recordsPtr(),
                    count_,
                    dim_,
                    k,
                    recordStride(),
                    idx_out.data(),
                    score_out.data()) != 0) {
                throw std::runtime_error("BadIndexNorm");
            }
        } else {
            const auto partial = runTopkParallel(query, k, threads);
            idx_out = partial.first;
            score_out = partial.second;
        }
        const auto t3 = std::chrono::steady_clock::now();

        std::vector<SearchHit> hits(static_cast<std::size_t>(k));
        for (std::uint64_t i = 0; i < k; ++i) {
            const std::uint32_t row = idx_out[static_cast<std::size_t>(i)];
            if (row >= count_) {
                throw std::runtime_error("TopkFailed");
            }
            hits[static_cast<std::size_t>(i)] = {entry(row).doc_id, score_out[static_cast<std::size_t>(i)]};
        }
        std::sort(hits.begin(), hits.end(), [](const SearchHit& a, const SearchHit& b) {
            if (a.score > b.score) {
                return true;
            }
            if (a.score < b.score) {
                return false;
            }
            return a.doc_id < b.doc_id;
        });
        const auto t4 = std::chrono::steady_clock::now();

        breakdown.qnorm_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        breakdown.topk_ms = std::chrono::duration<double, std::milli>(t3 - t2).count();
        breakdown.resolve_ms = std::chrono::duration<double, std::milli>(t4 - t3).count();
        return {hits, breakdown};
    }

private:
    [[nodiscard]] std::vector<SearchHit> searchSerial(std::span<const float> query, std::size_t top_k) const {
        return searchWithBreakdown(query, top_k, 1).first;
    }

    [[nodiscard]] std::vector<SearchHit> searchParallel(
        std::span<const float> query,
        std::size_t top_k,
        std::size_t threads) const {
        return searchWithBreakdown(query, top_k, threads).first;
    }

    [[nodiscard]] std::pair<std::vector<std::uint32_t>, std::vector<float>> runTopkParallel(
        std::span<const float> query,
        std::uint64_t k,
        std::size_t threads) const {
        const std::size_t tcount = std::min<std::size_t>(std::max<std::size_t>(threads, 1), 4);
        if (tcount == 1 || count_ == 0) {
            std::vector<std::uint32_t> idx(static_cast<std::size_t>(k));
            std::vector<float> scores(static_cast<std::size_t>(k));
            if (lb_vec_topk_cosine_lv(
                    query.data(),
                    recordsPtr(),
                    count_,
                    dim_,
                    k,
                    recordStride(),
                    idx.data(),
                    scores.data()) != 0) {
                throw std::runtime_error("BadIndexNorm");
            }
            return {idx, scores};
        }

        struct Partial {
            std::uint64_t start{};
            std::uint64_t len{};
            std::vector<std::uint32_t> idx;
            std::vector<float> scores;
            int status{0};
        };

        const std::uint64_t base = count_ / tcount;
        const std::uint64_t rem = count_ % tcount;
        std::vector<Partial> parts(tcount);
        std::vector<std::thread> workers;
        workers.reserve(tcount);

        std::uint64_t offset = 0;
        for (std::size_t t = 0; t < tcount; ++t) {
            const std::uint64_t len = base + (t < rem ? 1 : 0);
            parts[t].start = offset;
            parts[t].len = len;
            parts[t].idx.resize(static_cast<std::size_t>(k));
            parts[t].scores.resize(static_cast<std::size_t>(k));
            offset += len;
        }

        for (std::size_t t = 0; t < tcount; ++t) {
            workers.emplace_back([this, &query, k, t, &parts]() {
                auto& part = parts[t];
                if (part.len == 0) {
                    part.status = 0;
                    return;
                }
                const auto* rec = recordsPtr() + static_cast<std::size_t>(part.start) * recordStride();
                part.status = lb_vec_topk_cosine_lv(
                    query.data(),
                    rec,
                    part.len,
                    dim_,
                    k,
                    recordStride(),
                    part.idx.data(),
                    part.scores.data());
                for (auto& idx : part.idx) {
                    idx += static_cast<std::uint32_t>(part.start);
                }
            });
        }
        for (auto& worker : workers) {
            worker.join();
        }
        for (const auto& part : parts) {
            if (part.status != 0) {
                throw std::runtime_error("BadIndexNorm");
            }
        }

        std::vector<std::pair<float, std::uint32_t>> merged;
        merged.reserve(static_cast<std::size_t>(k) * tcount);
        for (const auto& part : parts) {
            for (std::uint64_t i = 0; i < k && i < part.len; ++i) {
                merged.emplace_back(part.scores[static_cast<std::size_t>(i)], part.idx[static_cast<std::size_t>(i)]);
            }
        }
        std::sort(merged.begin(), merged.end(), [](const auto& a, const auto& b) {
            if (a.first > b.first) {
                return true;
            }
            if (a.first < b.first) {
                return false;
            }
            return a.second < b.second;
        });
        if (merged.size() > static_cast<std::size_t>(k)) {
            merged.resize(static_cast<std::size_t>(k));
        }

        std::vector<std::uint32_t> idx(merged.size());
        std::vector<float> scores(merged.size());
        for (std::size_t i = 0; i < merged.size(); ++i) {
            scores[i] = merged[i].first;
            idx[i] = merged[i].second;
        }
        return {idx, scores};
    }

    struct HeaderInfo {
        std::uint32_t dim{};
        std::uint64_t count{};
    };

    static HeaderInfo validateHeader(std::span<const std::uint8_t> data) {
        constexpr std::size_t header_size = kMagic.size() + 4 + 4 + 8 + 8 + 8;
        if (data.size() < header_size) {
            throw std::runtime_error("BadIndex");
        }
        if (!std::equal(kMagic.begin(), kMagic.end(), data.begin())) {
            throw std::runtime_error("BadMagic");
        }
        const auto* p = data.data() + kMagic.size();
        const std::uint32_t ver = readU32Le(p);
        if (ver != kVersion) {
            throw std::runtime_error("BadVersion");
        }
        const std::uint32_t dim = readU32Le(p + 4);
        if (dim < kDimMin || dim > kDimMax) {
            throw std::runtime_error("BadDim");
        }
        const std::uint64_t count = readU64Le(p + 8);
        const std::size_t need = header_size + static_cast<std::size_t>(count) * recordSize(dim);
        if (data.size() < need) {
            throw std::runtime_error("TruncatedIndex");
        }
        return {dim, count};
    }

    static constexpr std::size_t headerSize() { return kMagic.size() + 4 + 4 + 8 + 8 + 8; }
    static std::size_t recordSize(std::uint32_t dim) { return 16 + static_cast<std::size_t>(dim) * 4; }

    VectorIndex(
        std::uint32_t dim,
        std::uint64_t count,
        std::shared_ptr<MappedFile> mapped,
        std::vector<std::uint8_t> owned)
        : mapped_(std::move(mapped)), owned_(std::move(owned)), dim_(dim), count_(count) {
        if (mapped_) {
            data_ = mapped_->bytes();
        } else {
            data_ = owned_;
        }
    }

    std::shared_ptr<MappedFile> mapped_;
    std::vector<std::uint8_t> owned_;
    std::span<const std::uint8_t> data_;
    std::uint32_t dim_{};
    std::uint64_t count_{};
};

inline ParsedPayload parsePayload(std::span<const std::uint8_t> payload) {
    std::uint32_t dim{};
    std::uint64_t doc_id{};
    const float* vector_ptr = nullptr;
    if (lb_logvec_payload_validate(
            payload.data(),
            payload.size(),
            &dim,
            &doc_id,
            &vector_ptr) != 0) {
        throw std::runtime_error("BadPayload");
    }
    const std::span<const float> vector{vector_ptr, dim};
    const float norm = lb_vec_norm_f32(vector.data(), dim);
    if (norm == 0.0f) {
        throw std::runtime_error("ZeroNorm");
    }
    return {dim, doc_id, vector};
}

inline std::uint64_t resolveDocId(std::uint64_t explicit_id, std::uint64_t topic_record_offset) {
    if (explicit_id == kDocIdAuto) {
        return topic_record_offset;
    }
    return explicit_id;
}

class IndexBuilder {
public:
    IndexBuilder() = default;

    ~IndexBuilder() {
        if (out_.is_open()) {
            out_.close();
        }
    }

    IndexBuilder(const IndexBuilder&) = delete;
    IndexBuilder& operator=(const IndexBuilder&) = delete;

    void open(const std::filesystem::path& path) {
        if (out_.is_open()) {
            throw std::runtime_error("IndexBuilder already open");
        }
        path_ = path;
        out_.open(path, std::ios::binary | std::ios::trunc);
        if (!out_) {
            throw std::runtime_error("failed to open file for write: " + path.string());
        }
        writeHeaderPlaceholder();
    }

    void append(const IngestRecord& rec) {
        if (!out_.is_open()) {
            throw std::runtime_error("IndexBuilder not open");
        }
        const ParsedPayload parsed = parsePayload(rec.payload);
        if (!has_dim_) {
            dim_ = parsed.dim;
            has_dim_ = true;
            patchU32(kDimOffset, dim_);
        } else if (parsed.dim != dim_) {
            throw std::runtime_error("DimMismatch");
        }
        seekEnd();
        const std::uint64_t doc_id = resolveDocId(parsed.doc_id, rec.topic_record_offset);
        writeRecord(doc_id, parsed.vector);
        ++count_;
    }

    void finalize() {
        if (!out_.is_open()) {
            throw std::runtime_error("IndexBuilder not open");
        }
        if (!has_dim_ || count_ == 0) {
            throw std::runtime_error("EmptyInput");
        }
        patchU64(kCountOffset, count_);
        out_.close();
        finalized_ = true;
    }

private:
    static constexpr std::size_t kDimOffset = VectorIndex::kMagic.size() + 4;
    static constexpr std::size_t kCountOffset = VectorIndex::kMagic.size() + 4 + 4;

    void writeHeaderPlaceholder() {
        out_.write(reinterpret_cast<const char*>(VectorIndex::kMagic.data()), VectorIndex::kMagic.size());
        writeU32(VectorIndex::kVersion);
        writeU32(0);
        writeU64(0);
        writeU64(0);
        writeU64(0);
    }

    void writeRecord(std::uint64_t doc_id, std::span<const float> vector) {
        writeU64(doc_id);
        std::vector<float> unit(dim_);
        const float unit_norm = normalizeUnitVector(vector, dim_, unit.data());
        std::uint32_t norm_bits{};
        std::memcpy(&norm_bits, &unit_norm, sizeof(norm_bits));
        writeU32(norm_bits);
        writeU32(0);
        out_.write(
            reinterpret_cast<const char*>(unit.data()),
            static_cast<std::streamsize>(dim_ * sizeof(float)));
        if (!out_) {
            throw std::runtime_error("failed to write index record: " + path_.string());
        }
    }

    void writeU32(std::uint32_t v) {
        std::uint8_t buf[4];
        std::memcpy(buf, &v, sizeof(v));
        out_.write(reinterpret_cast<const char*>(buf), sizeof(buf));
    }

    void writeU64(std::uint64_t v) {
        std::uint8_t buf[8];
        std::memcpy(buf, &v, sizeof(v));
        out_.write(reinterpret_cast<const char*>(buf), sizeof(buf));
    }

    void patchU32(std::size_t offset, std::uint32_t v) {
        out_.seekp(static_cast<std::streamoff>(offset));
        writeU32(v);
    }

    void patchU64(std::size_t offset, std::uint64_t v) {
        out_.seekp(static_cast<std::streamoff>(offset));
        writeU64(v);
    }

    void seekEnd() {
        out_.seekp(0, std::ios::end);
    }

    std::ofstream out_;
    std::filesystem::path path_;
    std::uint32_t dim_{};
    std::uint64_t count_{};
    bool has_dim_{false};
    bool finalized_{false};
};

}  // namespace logvec
