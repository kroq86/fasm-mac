#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_DIR="$(mktemp -d "${TMPDIR:-/tmp}/httpmini-check.XXXXXX")"
trap 'if [[ -n "${SERVER_PID:-}" ]]; then kill "$SERVER_PID" 2>/dev/null || true; wait "$SERVER_PID" 2>/dev/null || true; fi; rm -rf "$OUT_DIR"' EXIT

BIN="$OUT_DIR/httpmini"
PUBLIC="$OUT_DIR/public"
PRIVATE="$OUT_DIR/private.txt"
PYTHON="${PYTHON:-python3}"

PORT="$("$PYTHON" - <<'PY'
import socket

with socket.socket() as s:
    s.bind(("127.0.0.1", 0))
    print(s.getsockname()[1])
PY
)"

fasm "$ROOT/fasm/apps/httpmini.asm" "$BIN" >/dev/null

mkdir -p "$PUBLIC/dir"
printf '<h1>home</h1>\n' > "$PUBLIC/index.html"
printf 'hello\n' > "$PUBLIC/hello.txt"
printf '{"ok":true}\n' > "$PUBLIC/data.json"
"$PYTHON" - "$PUBLIC/big.bin" <<'PY'
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
path.write_bytes(bytes((i * 17 + 3) % 256 for i in range(20000)))
PY
printf 'secret\n' > "$PRIVATE"
ln -s "$PRIVATE" "$PUBLIC/escape.txt"

arch -x86_64 "$BIN" --root "$PUBLIC" --port "$PORT" --bind 127.0.0.1 >"$OUT_DIR/server.out" 2>"$OUT_DIR/server.err" &
SERVER_PID=$!

"$PYTHON" - "$PORT" <<'PY'
import socket
import sys
import time

port = int(sys.argv[1])

def connect_retry():
    last = None
    for _ in range(80):
        try:
            return socket.create_connection(("127.0.0.1", port), timeout=0.5)
        except OSError as exc:
            last = exc
            time.sleep(0.05)
    raise RuntimeError(f"server did not accept connections: {last}")

def request(raw):
    sock = connect_retry()
    sock.settimeout(2)
    sock.sendall(raw)
    chunks = []
    while True:
        try:
            chunk = sock.recv(4096)
        except socket.timeout:
            raise AssertionError("timed out waiting for response")
        if not chunk:
            break
        chunks.append(chunk)
    sock.close()
    return b"".join(chunks)

def assert_contains(response, part):
    if part not in response:
        raise AssertionError(f"missing {part!r} in {response!r}")

probe = connect_retry()
probe.close()

resp = request(b"GET /hello.txt HTTP/1.1\r\nHost: local\r\n\r\n")
assert_contains(resp, b"HTTP/1.1 200 OK\r\n")
assert_contains(resp, b"Content-Length: 6\r\n")
if not resp.endswith(b"\r\n\r\nhello\n"):
    raise AssertionError(f"bad GET body: {resp!r}")

resp = request(b"GET / HTTP/1.1\r\nHost: local\r\n\r\n")
assert_contains(resp, b"HTTP/1.1 200 OK\r\n")
assert_contains(resp, b"Content-Type: text/html; charset=utf-8\r\n")
if not resp.endswith(b"\r\n\r\n<h1>home</h1>\n"):
    raise AssertionError(f"bad index body: {resp!r}")

big_body = bytes((i * 17 + 3) % 256 for i in range(20000))
resp = request(b"GET /big.bin HTTP/1.1\r\nHost: local\r\n\r\n")
assert_contains(resp, b"HTTP/1.1 200 OK\r\n")
assert_contains(resp, b"Content-Length: 20000\r\n")
if not resp.endswith(b"\r\n\r\n" + big_body):
    raise AssertionError("bad big.bin body")

resp = request(b"HEAD /hello.txt HTTP/1.1\r\nHost: local\r\n\r\n")
assert_contains(resp, b"HTTP/1.1 200 OK\r\n")
assert_contains(resp, b"Content-Length: 6\r\n")
if not resp.endswith(b"\r\n\r\n"):
    raise AssertionError(f"HEAD returned a body: {resp!r}")

assert_contains(request(b"GET /missing.txt HTTP/1.1\r\n\r\n"), b"HTTP/1.1 404 Not Found\r\n")
assert_contains(request(b"GET /dir HTTP/1.1\r\n\r\n"), b"HTTP/1.1 403 Forbidden\r\n")
assert_contains(request(b"GET /../private.txt HTTP/1.1\r\n\r\n"), b"HTTP/1.1 403 Forbidden\r\n")
assert_contains(request(b"GET /%2e%2e/private.txt HTTP/1.1\r\n\r\n"), b"HTTP/1.1 403 Forbidden\r\n")
assert_contains(request(b"GET /escape.txt HTTP/1.1\r\n\r\n"), b"HTTP/1.1 403 Forbidden\r\n")
assert_contains(request(b"POST /hello.txt HTTP/1.1\r\n\r\n"), b"HTTP/1.1 405 Method Not Allowed\r\n")

stalled = connect_retry()
stalled.sendall(b"GET /hello.txt")
resp = request(b"GET /data.json HTTP/1.1\r\nHost: local\r\n\r\n")
assert_contains(resp, b"HTTP/1.1 200 OK\r\n")
assert_contains(resp, b'{"ok":true}\n')
stalled.close()
PY

grep -q 'GET /hello.txt 200' "$OUT_DIR/server.err"
grep -q 'GET / 200' "$OUT_DIR/server.err"
grep -q 'GET /big.bin 200' "$OUT_DIR/server.err"
grep -q 'HEAD /hello.txt 200' "$OUT_DIR/server.err"
grep -q 'GET /missing.txt 404' "$OUT_DIR/server.err"
grep -q -- '- 405' "$OUT_DIR/server.err"

echo 'httpmini checks passed'
