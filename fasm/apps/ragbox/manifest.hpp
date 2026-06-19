#pragma once

#include "chunker.hpp"

#include <cctype>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace ragbox {

struct ManifestRecord {
    std::uint64_t doc_id{};
    std::string path;
    std::size_t offset{};
    std::size_t length{};
    std::string text;
};

struct Manifest {
    static constexpr std::uint32_t kVersion = 1;

    std::uint32_t version{kVersion};
    std::uint32_t dim{};
    std::string model;
    std::size_t chunk_size{};
    std::size_t overlap{};
    std::string root;
    std::vector<ManifestRecord> records;
};

inline std::string jsonEscape(std::string_view s) {
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

inline void skipWs(const std::string& s, std::size_t& i) {
    while (i < s.size() && (s[i] == ' ' || s[i] == '\n' || s[i] == '\r' || s[i] == '\t')) {
        ++i;
    }
}

inline bool expectChar(const std::string& s, std::size_t& i, char c) {
    skipWs(s, i);
    if (i >= s.size() || s[i] != c) {
        return false;
    }
    ++i;
    return true;
}

inline bool parseJsonString(const std::string& s, std::size_t& i, std::string& out) {
    skipWs(s, i);
    if (i >= s.size() || s[i] != '"') {
        return false;
    }
    ++i;
    out.clear();
    while (i < s.size()) {
        const char c = s[i++];
        if (c == '"') {
            return true;
        }
        if (c == '\\') {
            if (i >= s.size()) {
                return false;
            }
            const char esc = s[i++];
            switch (esc) {
            case '"':
                out += '"';
                break;
            case '\\':
                out += '\\';
                break;
            case 'n':
                out += '\n';
                break;
            case 'r':
                out += '\r';
                break;
            case 't':
                out += '\t';
                break;
            default:
                out += esc;
                break;
            }
            continue;
        }
        out += c;
    }
    return false;
}

inline bool parseJsonUInt(const std::string& s, std::size_t& i, std::uint64_t& out) {
    skipWs(s, i);
    if (i >= s.size() || !std::isdigit(static_cast<unsigned char>(s[i]))) {
        return false;
    }
    std::uint64_t v = 0;
    while (i < s.size() && std::isdigit(static_cast<unsigned char>(s[i]))) {
        v = v * 10 + static_cast<std::uint64_t>(s[i++] - '0');
    }
    out = v;
    return true;
}

inline bool findKey(const std::string& s, std::string_view key, std::size_t start, std::size_t& pos) {
    const std::string needle = std::string("\"") + std::string(key) + "\":";
    pos = s.find(needle, start);
    return pos != std::string::npos;
}

inline std::string readWholeText(const std::filesystem::path& path) {
    std::ifstream in(path);
    if (!in) {
        throw std::runtime_error("failed to open manifest: " + path.string());
    }
    std::ostringstream ss;
    ss << in.rdbuf();
    return ss.str();
}

inline Manifest loadManifest(const std::filesystem::path& path) {
    const std::string json = readWholeText(path);
    Manifest m{};
    std::size_t pos = 0;

    if (findKey(json, "version", 0, pos)) {
        pos += 10;
        std::uint64_t v = 0;
        if (!parseJsonUInt(json, pos, v)) {
            throw std::runtime_error("BadManifestVersion");
        }
        m.version = static_cast<std::uint32_t>(v);
    }
    if (findKey(json, "dim", 0, pos)) {
        pos += 6;
        std::uint64_t v = 0;
        if (!parseJsonUInt(json, pos, v)) {
            throw std::runtime_error("BadManifestDim");
        }
        m.dim = static_cast<std::uint32_t>(v);
    }
    if (findKey(json, "model", 0, pos)) {
        pos += 8;
        if (!parseJsonString(json, pos, m.model)) {
            throw std::runtime_error("BadManifestModel");
        }
    }
    if (findKey(json, "chunk_size", 0, pos)) {
        pos += 13;
        std::uint64_t v = 0;
        if (!parseJsonUInt(json, pos, v)) {
            throw std::runtime_error("BadManifestChunkSize");
        }
        m.chunk_size = static_cast<std::size_t>(v);
    }
    if (findKey(json, "overlap", 0, pos)) {
        pos += 10;
        std::uint64_t v = 0;
        if (!parseJsonUInt(json, pos, v)) {
            throw std::runtime_error("BadManifestOverlap");
        }
        m.overlap = static_cast<std::size_t>(v);
    }
    if (findKey(json, "root", 0, pos)) {
        pos += 7;
        if (!parseJsonString(json, pos, m.root)) {
            throw std::runtime_error("BadManifestRoot");
        }
    }

    const std::string records_key = "\"records\"";
    std::size_t records_pos = json.find(records_key);
    if (records_pos == std::string::npos) {
        throw std::runtime_error("BadManifestRecords");
    }
    std::size_t i = records_pos + records_key.size();
    if (!expectChar(json, i, ':') || !expectChar(json, i, '[')) {
        throw std::runtime_error("BadManifestRecords");
    }
    skipWs(json, i);
    while (i < json.size() && json[i] != ']') {
        if (!expectChar(json, i, '{')) {
            throw std::runtime_error("BadManifestRecord");
        }
        ManifestRecord rec{};
        while (i < json.size() && json[i] != '}') {
            std::string key;
            if (!parseJsonString(json, i, key)) {
                throw std::runtime_error("BadManifestRecordKey");
            }
            if (!expectChar(json, i, ':')) {
                throw std::runtime_error("BadManifestRecordKey");
            }
            if (key == "doc_id") {
                std::uint64_t v = 0;
                if (!parseJsonUInt(json, i, v)) {
                    throw std::runtime_error("BadManifestDocId");
                }
                rec.doc_id = v;
            } else if (key == "path") {
                if (!parseJsonString(json, i, rec.path)) {
                    throw std::runtime_error("BadManifestPath");
                }
            } else if (key == "offset") {
                std::uint64_t v = 0;
                if (!parseJsonUInt(json, i, v)) {
                    throw std::runtime_error("BadManifestOffset");
                }
                rec.offset = static_cast<std::size_t>(v);
            } else if (key == "length") {
                std::uint64_t v = 0;
                if (!parseJsonUInt(json, i, v)) {
                    throw std::runtime_error("BadManifestLength");
                }
                rec.length = static_cast<std::size_t>(v);
            } else if (key == "text") {
                if (!parseJsonString(json, i, rec.text)) {
                    throw std::runtime_error("BadManifestText");
                }
            } else {
                skipWs(json, i);
                if (i < json.size() && json[i] == '"') {
                    std::string skip;
                    if (!parseJsonString(json, i, skip)) {
                        throw std::runtime_error("BadManifestRecord");
                    }
                } else {
                    while (i < json.size() && json[i] != ',' && json[i] != '}') {
                        ++i;
                    }
                }
            }
            skipWs(json, i);
            if (i < json.size() && json[i] == ',') {
                ++i;
            }
        }
        if (!expectChar(json, i, '}')) {
            throw std::runtime_error("BadManifestRecord");
        }
        m.records.push_back(std::move(rec));
        skipWs(json, i);
        if (i < json.size() && json[i] == ',') {
            ++i;
        }
        skipWs(json, i);
    }

    if (m.version != Manifest::kVersion) {
        throw std::runtime_error("BadManifestVersion");
    }
    return m;
}

inline void writeManifest(const std::filesystem::path& path, const Manifest& m) {
    std::ofstream out(path, std::ios::trunc);
    if (!out) {
        throw std::runtime_error("failed to write manifest: " + path.string());
    }
    out << "{\n";
    out << "  \"version\": " << m.version << ",\n";
    out << "  \"dim\": " << m.dim << ",\n";
    out << "  \"model\": \"" << jsonEscape(m.model) << "\",\n";
    out << "  \"chunk_size\": " << m.chunk_size << ",\n";
    out << "  \"overlap\": " << m.overlap << ",\n";
    out << "  \"root\": \"" << jsonEscape(m.root) << "\",\n";
    out << "  \"records\": [\n";
    for (std::size_t i = 0; i < m.records.size(); ++i) {
        const auto& rec = m.records[i];
        out << "    {\n";
        out << "      \"doc_id\": " << rec.doc_id << ",\n";
        out << "      \"path\": \"" << jsonEscape(rec.path) << "\",\n";
        out << "      \"offset\": " << rec.offset << ",\n";
        out << "      \"length\": " << rec.length << ",\n";
        out << "      \"text\": \"" << jsonEscape(rec.text) << "\"\n";
        out << "    }";
        if (i + 1 < m.records.size()) {
            out << ',';
        }
        out << '\n';
    }
    out << "  ]\n";
    out << "}\n";
}

inline Manifest manifestFromChunks(
    const std::vector<Chunk>& chunks,
    std::uint32_t dim,
    const std::string& model,
    std::size_t chunk_size,
    std::size_t overlap,
    const std::filesystem::path& root) {
    Manifest m{};
    m.dim = dim;
    m.model = model;
    m.chunk_size = chunk_size;
    m.overlap = overlap;
    m.root = std::filesystem::absolute(root).string();
    m.records.reserve(chunks.size());
    for (const auto& chunk : chunks) {
        ManifestRecord rec{};
        rec.doc_id = chunk.doc_id;
        rec.path = chunk.path;
        rec.offset = chunk.offset;
        rec.length = chunk.text.size();
        rec.text = chunk.text;
        m.records.push_back(std::move(rec));
    }
    return m;
}

inline std::filesystem::path defaultManifestPath(const std::filesystem::path& index_path) {
    return index_path.string() + ".manifest.json";
}

inline const ManifestRecord* findRecord(const Manifest& m, std::uint64_t doc_id) {
    if (doc_id < m.records.size() && m.records[doc_id].doc_id == doc_id) {
        return &m.records[doc_id];
    }
    for (const auto& rec : m.records) {
        if (rec.doc_id == doc_id) {
            return &rec;
        }
    }
    return nullptr;
}

inline void validateManifestIndex(const Manifest& m, std::uint32_t index_dim, std::uint64_t index_count) {
    if (m.dim != index_dim) {
        throw std::runtime_error("ManifestDimMismatch");
    }
    if (m.records.size() != index_count) {
        throw std::runtime_error("ManifestCountMismatch");
    }
    for (std::size_t i = 0; i < m.records.size(); ++i) {
        if (m.records[i].doc_id != i) {
            throw std::runtime_error("ManifestDocIdGap");
        }
    }
}

}  // namespace ragbox
