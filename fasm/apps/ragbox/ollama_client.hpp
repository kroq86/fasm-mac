#pragma once

#include <cctype>
#include <cstdint>
#include <cstring>
#include <netdb.h>
#include <stdexcept>
#include <string>
#include <sys/socket.h>
#include <unistd.h>
#include <vector>

namespace ragbox {

struct ParsedUrl {
    std::string host;
    std::uint16_t port{};
    std::string path;
};

inline ParsedUrl parseHttpUrl(const std::string& url) {
    constexpr std::string_view prefix = "http://";
    if (url.size() < prefix.size() || url.compare(0, prefix.size(), prefix) != 0) {
        throw std::runtime_error("OllamaUrlMustBeHttp");
    }
    ParsedUrl out{};
    std::size_t start = prefix.size();
    std::size_t slash = url.find('/', start);
    const std::string host_port = slash == std::string::npos ? url.substr(start) : url.substr(start, slash - start);
    out.path = slash == std::string::npos ? "/" : url.substr(slash);
    const std::size_t colon = host_port.find(':');
    if (colon == std::string::npos) {
        out.host = host_port;
        out.port = 80;
    } else {
        out.host = host_port.substr(0, colon);
        out.port = static_cast<std::uint16_t>(std::stoul(host_port.substr(colon + 1)));
    }
    if (out.host.empty()) {
        throw std::runtime_error("BadOllamaUrl");
    }
    return out;
}

inline std::string httpRequest(
    const ParsedUrl& url,
    const std::string& method,
    const std::string& body,
    const char* content_type) {
    addrinfo hints{};
    hints.ai_family = AF_UNSPEC;
    hints.ai_socktype = SOCK_STREAM;
    addrinfo* res = nullptr;
    const std::string port_str = std::to_string(url.port);
    if (getaddrinfo(url.host.c_str(), port_str.c_str(), &hints, &res) != 0) {
        throw std::runtime_error("OllamaResolveFailed");
    }

    int fd = -1;
    for (addrinfo* p = res; p != nullptr; p = p->ai_next) {
        fd = static_cast<int>(socket(p->ai_family, p->ai_socktype, p->ai_protocol));
        if (fd < 0) {
            continue;
        }
        if (connect(fd, p->ai_addr, p->ai_addrlen) == 0) {
            break;
        }
        close(fd);
        fd = -1;
    }
    freeaddrinfo(res);
    if (fd < 0) {
        throw std::runtime_error("OllamaConnectFailed");
    }

    std::string req;
    req += method;
    req += ' ';
    req += url.path;
    req += " HTTP/1.1\r\n";
    req += "Host: ";
    req += url.host;
    req += "\r\n";
    req += "Connection: close\r\n";
    if (!body.empty()) {
        req += "Content-Type: ";
        req += content_type;
        req += "\r\n";
        req += "Content-Length: ";
        req += std::to_string(body.size());
        req += "\r\n";
    }
    req += "\r\n";
    req += body;

    std::size_t sent = 0;
    while (sent < req.size()) {
        const ssize_t n = send(fd, req.data() + sent, req.size() - sent, 0);
        if (n <= 0) {
            close(fd);
            throw std::runtime_error("OllamaSendFailed");
        }
        sent += static_cast<std::size_t>(n);
    }

    std::string response;
    char buf[4096];
    while (true) {
        const ssize_t n = recv(fd, buf, sizeof(buf), 0);
        if (n <= 0) {
            break;
        }
        response.append(buf, static_cast<std::size_t>(n));
    }
    close(fd);

    const std::size_t header_end = response.find("\r\n\r\n");
    if (header_end == std::string::npos) {
        throw std::runtime_error("OllamaBadResponse");
    }
    const std::string headers = response.substr(0, header_end);
    if (headers.find("HTTP/1.") == std::string::npos) {
        throw std::runtime_error("OllamaBadResponse");
    }
    const std::size_t status_start = headers.find(' ') + 1;
    const int status = std::stoi(headers.substr(status_start, 3));
    if (status < 200 || status >= 300) {
        throw std::runtime_error("OllamaHttpError");
    }
    return response.substr(header_end + 4);
}

inline std::string jsonEscapePrompt(std::string_view s) {
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

inline std::vector<float> parseEmbeddingArray(const std::string& body) {
    const std::string key = "\"embedding\"";
    const std::size_t pos = body.find(key);
    if (pos == std::string::npos) {
        throw std::runtime_error("OllamaNoEmbedding");
    }
    std::size_t i = pos + key.size();
    while (i < body.size() && body[i] != '[') {
        ++i;
    }
    if (i >= body.size()) {
        throw std::runtime_error("OllamaNoEmbedding");
    }
    ++i;
    std::vector<float> out;
    while (i < body.size() && body[i] != ']') {
        while (i < body.size() && (body[i] == ' ' || body[i] == ',' || body[i] == '\n' || body[i] == '\r')) {
            ++i;
        }
        if (i >= body.size() || body[i] == ']') {
            break;
        }
        char* end = nullptr;
        const float v = std::strtof(body.c_str() + i, &end);
        if (end == body.c_str() + i) {
            throw std::runtime_error("OllamaBadEmbedding");
        }
        out.push_back(v);
        i = static_cast<std::size_t>(end - body.c_str());
    }
    if (out.empty()) {
        throw std::runtime_error("OllamaEmptyEmbedding");
    }
    return out;
}

inline std::vector<float> ollamaEmbed(
    const std::string& base_url,
    const std::string& model,
    const std::string& prompt) {
    ParsedUrl url = parseHttpUrl(base_url);
    url.path = "/api/embeddings";
    const std::string body =
        std::string("{\"model\":\"") + jsonEscapePrompt(model) + "\",\"prompt\":\"" + jsonEscapePrompt(prompt) + "\"}";
    const std::string resp = httpRequest(url, "POST", body, "application/json");
    return parseEmbeddingArray(resp);
}

inline bool ollamaPing(const std::string& base_url) {
    try {
        ParsedUrl url = parseHttpUrl(base_url);
        url.path = "/api/tags";
        (void)httpRequest(url, "GET", "", "application/json");
        return true;
    } catch (...) {
        return false;
    }
}

inline bool ollamaHasModel(const std::string& base_url, const std::string& model) {
    ParsedUrl url = parseHttpUrl(base_url);
    url.path = "/api/tags";
    const std::string body = httpRequest(url, "GET", "", "application/json");
    const std::string needle = std::string("\"name\":\"") + jsonEscapePrompt(model) + "\"";
    if (body.find(needle) != std::string::npos) {
        return true;
    }
    const std::string prefix_needle = std::string("\"name\":\"") + jsonEscapePrompt(model) + ":";
    return body.find(prefix_needle) != std::string::npos;
}

}  // namespace ragbox
