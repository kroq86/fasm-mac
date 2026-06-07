#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_DIR="$(mktemp -d "${TMPDIR:-/tmp}/miniredis-server-check.XXXXXX")"
trap 'if [[ -n "${SERVER_PID:-}" ]]; then kill "$SERVER_PID" 2>/dev/null || true; wait "$SERVER_PID" 2>/dev/null || true; fi; rm -rf "$OUT_DIR"' EXIT

BIN="$OUT_DIR/miniredis_server"
TMP_ELF="$OUT_DIR/miniredis_server.elf"
FASM_HOST="$ROOT/fasm/build/out/macos-x64/fasm-macos-x64"
CONVERTER="$ROOT/fasm/tools/elf64_to_macho64.py"
PYTHON="${PYTHON:-python3}"

PORT="$("$PYTHON" - <<'PY'
import socket

with socket.socket() as s:
    s.bind(("127.0.0.1", 0))
    print(s.getsockname()[1])
PY
)"

"$ROOT/bin/fasm" host -d TARGET_OS=macos -d SERVER_PORT="$PORT" "$ROOT/fasm/apps/miniredis_server.asm" "$TMP_ELF" >/dev/null
"$PYTHON" "$CONVERTER" --type exec "$TMP_ELF" "$BIN"

start_server() {
    (cd "$OUT_DIR" && arch -x86_64 "$BIN" >"$OUT_DIR/server.out" 2>"$OUT_DIR/server.err") &
    SERVER_PID=$!
}

stop_server() {
    if [[ -n "${SERVER_PID:-}" ]]; then
        kill "$SERVER_PID" 2>/dev/null || true
        wait "$SERVER_PID" 2>/dev/null || true
        SERVER_PID=
    fi
}

start_server

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

def request(sock, payload, expected):
    sock.sendall(payload)
    got = b""
    while len(got) < len(expected):
        chunk = sock.recv(4096)
        if not chunk:
            break
        got += chunk
    if got != expected:
        raise AssertionError(f"expected {expected!r}, got {got!r}")

probe = connect_retry()
probe.close()

stalled = connect_retry()
stalled.sendall(b"*2\r\n$3\r\nGET\r\n")

client = connect_retry()
request(client, b"*1\r\n$4\r\nPING\r\n", b"+PONG\r\n")
request(
    client,
    b"*3\r\n$3\r\nSET\r\n$3\r\nfoo\r\n$3\r\nbar\r\n",
    b"+OK\r\n",
)
request(client, b"*2\r\n$3\r\nGET\r\n$3\r\nfoo\r\n", b"$3\r\nbar\r\n")
request(
    client,
    b"*3\r\n$3\r\nSET\r\n$5\r\ncount\r\n$2\r\n42\r\n",
    b"+OK\r\n",
)
request(client, b"*2\r\n$3\r\nGET\r\n$5\r\ncount\r\n", b":42\r\n")
request(
    client,
    b"*3\r\n$3\r\nSET\r\n$3\r\nneg\r\n$2\r\n-7\r\n",
    b"+OK\r\n",
)
request(client, b"*2\r\n$3\r\nGET\r\n$3\r\nneg\r\n", b":-7\r\n")

stalled.close()
request(client, b"*1\r\n$4\r\nPING\r\n", b"+PONG\r\n")
client.close()
PY

stop_server
start_server

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
    raise RuntimeError(f"server did not accept connections after restart: {last}")

def request(sock, payload, expected):
    sock.sendall(payload)
    got = b""
    while len(got) < len(expected):
        chunk = sock.recv(4096)
        if not chunk:
            break
        got += chunk
    if got != expected:
        raise AssertionError(f"expected {expected!r}, got {got!r}")

client = connect_retry()
request(client, b"*2\r\n$3\r\nGET\r\n$3\r\nfoo\r\n", b"$3\r\nbar\r\n")
request(client, b"*2\r\n$3\r\nGET\r\n$5\r\ncount\r\n", b":42\r\n")
request(client, b"*2\r\n$3\r\nGET\r\n$3\r\nneg\r\n", b":-7\r\n")
request(client, b"*1\r\n$4\r\nPING\r\n", b"+PONG\r\n")
client.close()
PY

echo 'miniredis_server checks passed'
