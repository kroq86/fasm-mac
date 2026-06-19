#pragma once

#include "mapped_file.hpp"
#include "vector_core.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace logvec {

struct SearchHit {
    std::uint64_t doc_id{};
    float score{};
};

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

    [[nodiscard]] std::vector<SearchHit> search(std::span<const float> query, std::size_t top_k) const {
        if (query.size() != dim_) {
            throw std::runtime_error("QueryDimMismatch");
        }
        const float qnorm = lb_vec_norm_f32(query.data(), dim_);
        volatile float touch = lb_vec_dot_f32(query.data(), query.data(), dim_);
        (void)touch;
        if (qnorm == 0.0f) {
            throw std::runtime_error("ZeroNorm");
        }
        if (count_ == 0) {
            return {};
        }
        const std::uint64_t k = std::min(static_cast<std::uint64_t>(top_k), count_);
        std::vector<std::uint32_t> idx_out(k);
        std::vector<float> score_out(k);
        const auto* records = data_.data() + headerSize();
        const std::uint64_t record_stride = recordSize(dim_);

        if (lb_vec_topk_cosine_lv(
                query.data(),
                records,
                count_,
                dim_,
                k,
                record_stride,
                idx_out.data(),
                score_out.data()) != 0) {
            throw std::runtime_error("BadIndexNorm");
        }

        std::vector<SearchHit> hits(k);
        for (std::uint64_t i = 0; i < k; ++i) {
            const std::uint32_t row = idx_out[i];
            if (row >= count_) {
                throw std::runtime_error("TopkFailed");
            }
            hits[i] = {entry(row).doc_id, score_out[i]};
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
        return hits;
    }

private:
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
