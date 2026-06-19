#pragma once

#include "vector_index.hpp"

#include <array>
#include <cerrno>
#include <cstdint>
#include <cstring>
#include <functional>
#include <netdb.h>
#include <stdexcept>
#include <string>
#include <sys/socket.h>
#include <unistd.h>
#include <vector>

namespace logvec {

inline std::uint32_t crc32c(std::span<const std::uint8_t> data) {
    return lb_crc32c(data.data(), data.size());
}

inline std::uint64_t consumeLogbusBatchRecords(
    std::span<const std::uint8_t> batch,
    std::uint64_t base_offset,
    const std::function<void(IngestRecord)>& consume) {
    std::size_t pos = 0;
    std::uint64_t offset = base_offset;
    while (pos < batch.size()) {
        if (batch.size() - pos < 8) {
            throw std::runtime_error("TruncatedRecord");
        }
        const std::uint32_t plen = readU32Le(batch.data() + pos);
        const std::uint32_t want_crc = readU32Le(batch.data() + pos + 4);
        pos += 8;
        if (batch.size() - pos < plen) {
            throw std::runtime_error("TruncatedRecord");
        }
        const auto payload = batch.subspan(pos, plen);
        pos += plen;
        if (crc32c(payload) != want_crc) {
            throw std::runtime_error("BadCrc");
        }
        IngestRecord rec{};
        rec.topic_record_offset = offset;
        rec.payload.assign(payload.begin(), payload.end());
        consume(std::move(rec));
        ++offset;
    }
    return offset - base_offset;
}

inline std::uint64_t appendLogbusBatchRecords(
    std::span<const std::uint8_t> batch,
    std::uint64_t base_offset,
    std::vector<IngestRecord>& out) {
    return consumeLogbusBatchRecords(batch, base_offset, [&](IngestRecord rec) {
        out.push_back(std::move(rec));
    });
}

class LogbusClient {
public:
    LogbusClient(const std::string& host, std::uint16_t port) {
        addrinfo hints{};
        hints.ai_family = AF_UNSPEC;
        hints.ai_socktype = SOCK_STREAM;
        const std::string port_str = std::to_string(port);
        addrinfo* res = nullptr;
        const int gai = getaddrinfo(host.c_str(), port_str.c_str(), &hints, &res);
        if (gai != 0) {
            throw std::runtime_error(std::string("getaddrinfo failed: ") + gai_strerror(gai));
        }
        int fd = -1;
        for (addrinfo* it = res; it != nullptr; it = it->ai_next) {
            fd = socket(it->ai_family, it->ai_socktype, it->ai_protocol);
            if (fd < 0) {
                continue;
            }
            if (connect(fd, it->ai_addr, it->ai_addrlen) == 0) {
                break;
            }
            close(fd);
            fd = -1;
        }
        freeaddrinfo(res);
        if (fd < 0) {
            throw std::runtime_error("connect failed");
        }
        fd_ = fd;
    }

    LogbusClient(const LogbusClient&) = delete;
    LogbusClient& operator=(const LogbusClient&) = delete;

    ~LogbusClient() {
        if (fd_ >= 0) {
            close(fd_);
        }
    }

    void fetchAndConsume(const std::string& topic, const std::function<void(IngestRecord)>& consume) {
        std::uint64_t next_offset = 0;
        std::string line_buf;
        while (true) {
            const std::string off_str = std::to_string(next_offset);
            encode({"FETCHBATCH", topic, off_str, "1048576"});
            const std::vector<std::uint8_t> batch = readBulk(line_buf);
            if (batch.empty()) {
                break;
            }
            const std::uint64_t got = consumeLogbusBatchRecords(batch, next_offset, consume);
            if (got == 0) {
                break;
            }
            next_offset += got;
        }
    }

private:
    void writeAll(std::span<const std::uint8_t> bytes) {
        std::size_t sent = 0;
        while (sent < bytes.size()) {
            const ssize_t n = ::write(fd_, bytes.data() + sent, bytes.size() - sent);
            if (n < 0) {
                if (errno == EINTR) {
                    continue;
                }
                throw std::runtime_error("write failed");
            }
            if (n == 0) {
                throw std::runtime_error("write failed");
            }
            sent += static_cast<std::size_t>(n);
        }
    }

    void readExact(std::span<std::uint8_t> buf) {
        std::size_t got = 0;
        while (got < buf.size()) {
            const ssize_t n = ::read(fd_, buf.data() + got, buf.size() - got);
            if (n < 0) {
                if (errno == EINTR) {
                    continue;
                }
                throw std::runtime_error("read failed");
            }
            if (n == 0) {
                throw std::runtime_error("read failed");
            }
            got += static_cast<std::size_t>(n);
        }
    }

    [[nodiscard]] std::string readLine(std::string& buf) {
        buf.clear();
        while (true) {
            std::uint8_t byte{};
            readExact({&byte, 1});
            buf.push_back(static_cast<char>(byte));
            if (buf.size() >= 2 && buf[buf.size() - 2] == '\r' && buf[buf.size() - 1] == '\n') {
                return buf.substr(0, buf.size() - 2);
            }
        }
    }

    void encode(std::initializer_list<std::string> args) {
        std::string msg = "*" + std::to_string(args.size()) + "\r\n";
        for (const auto& arg : args) {
            msg += "$" + std::to_string(arg.size()) + "\r\n";
            msg += arg;
            msg += "\r\n";
        }
        writeAll({reinterpret_cast<const std::uint8_t*>(msg.data()), msg.size()});
    }

    [[nodiscard]] std::vector<std::uint8_t> readBulk(std::string& line_buf) {
        const std::string line = readLine(line_buf);
        if (line.empty() || line[0] == '-') {
            throw std::runtime_error("ProtocolError");
        }
        if (line[0] != '$') {
            throw std::runtime_error("ProtocolError");
        }
        const std::size_t size = std::stoull(line.substr(1));
        std::vector<std::uint8_t> payload(size);
        if (size > 0) {
            readExact(payload);
        }
        std::array<std::uint8_t, 2> crlf{};
        readExact(crlf);
        if (crlf[0] != '\r' || crlf[1] != '\n') {
            throw std::runtime_error("ProtocolError");
        }
        return payload;
    }

    int fd_{-1};
};

}  // namespace logvec
