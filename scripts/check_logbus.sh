#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_DIR="$(mktemp -d "${TMPDIR:-/tmp}/logbus-check.XXXXXX")"
trap 'if [[ -n "${SERVER_PID:-}" ]]; then kill "$SERVER_PID" 2>/dev/null || true; wait "$SERVER_PID" 2>/dev/null || true; fi; rm -rf "$OUT_DIR"' EXIT

BIN="$OUT_DIR/logbus"
DATA="$OUT_DIR/data"
PYTHON="${PYTHON:-python3}"

PORT="$("$PYTHON" - <<'PY'
import socket

with socket.socket() as s:
    s.bind(("127.0.0.1", 0))
    print(s.getsockname()[1])
PY
)"

fasm "$ROOT/fasm/apps/logbus.asm" "$BIN" >/dev/null

start_server() {
    arch -x86_64 "$BIN" --dir "$DATA" --port "$PORT" --bind 127.0.0.1 --segment-bytes 20 >"$OUT_DIR/server.out" 2>"$OUT_DIR/server.err" &
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
import struct
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

def encode(*args):
    out = f"*{len(args)}\r\n".encode()
    for arg in args:
        if isinstance(arg, str):
            arg = arg.encode()
        out += b"$%d\r\n" % len(arg) + arg + b"\r\n"
    return out

def read_line(sock):
    data = b""
    while not data.endswith(b"\r\n"):
        chunk = sock.recv(1)
        if not chunk:
            raise EOFError("connection closed")
        data += chunk
    return data

def read_resp(sock):
    line = read_line(sock)
    typ = line[:1]
    if typ in b"+-:":
        return line
    if typ == b"$":
        size = int(line[1:-2])
        data = b""
        while len(data) < size + 2:
            data += sock.recv(size + 2 - len(data))
        return line + data
    if typ == b"*":
        count = int(line[1:-2])
        return line + b"".join(read_resp(sock) for _ in range(count))
    raise AssertionError(f"unknown RESP line: {line!r}")

def request(sock, expected, *args):
    sock.sendall(encode(*args))
    got = read_resp(sock)
    if got != expected:
        raise AssertionError(f"{args}: expected {expected!r}, got {got!r}")

probe = connect_retry()
probe.close()

stalled = connect_retry()
stalled.sendall(b"*3\r\n$7\r\nPRODUCE\r\n$6\r\nevents\r\n")

client = connect_retry()
request(client, b"+PONG\r\n", "PING")
request(client, b":0\r\n", "PRODUCE", "events", "hello")
request(client, b":1\r\n", "PRODUCE", "events", "world")
request(
    client,
    b"*2\r\n*2\r\n:0\r\n$5\r\nhello\r\n*2\r\n:1\r\n$5\r\nworld\r\n",
    "FETCH",
    "events",
    "0",
    "4096",
)
raw_batch = struct.pack("<I", 5) + b"hello" + struct.pack("<I", 5) + b"world"
request(
    client,
    b"$%d\r\n" % len(raw_batch) + raw_batch + b"\r\n",
    "FETCHBATCH",
    "events",
    "0",
    "4096",
)
request(client, b"$0\r\n\r\n", "FETCHBATCH", "events", "99", "4096")
request(client, b":2\r\n", "PRODUCE", "events", "again")
request(
    client,
    b"*1\r\n*2\r\n:2\r\n$5\r\nagain\r\n",
    "FETCH",
    "events",
    "2",
    "4096",
)
raw_again = struct.pack("<I", 5) + b"again"
request(
    client,
    b"$%d\r\n" % len(raw_again) + raw_again + b"\r\n",
    "FETCHBATCH",
    "events",
    "2",
    "4096",
)
request(client, b"+OK\r\n", "COMMIT", "group1", "events", "3")
request(client, b":3\r\n", "OFFSET", "group1", "events")

client.sendall(encode("PRODUCE", "bad/name", "x"))
if not read_resp(client).startswith(b"-ERR"):
    raise AssertionError("invalid topic did not return ERR")

client.sendall(encode("PRODUCE", "events", b"a" * 5000))
if not read_resp(client).startswith(b"-ERR"):
    raise AssertionError("overlarge payload did not return ERR")

client.sendall(encode("FETCHBATCH", "events", "-1", "4096"))
if not read_resp(client).startswith(b"-ERR"):
    raise AssertionError("negative FETCHBATCH offset did not return ERR")

client.sendall(encode("FETCHBATCH", "events", "0", "nope"))
if not read_resp(client).startswith(b"-ERR"):
    raise AssertionError("bad FETCHBATCH max_bytes did not return ERR")

stalled.close()
client.close()
PY

test -f "$DATA/topics/events/00000000000000000000.log"
test -f "$DATA/topics/events/00000000000000000002.log"
LOG_COUNT="$(find "$DATA/topics/events" -name '*.log' | wc -l | tr -d ' ')"
if [[ "$LOG_COUNT" -lt 2 ]]; then
    echo "expected rotated log segments, found $LOG_COUNT" >&2
    exit 1
fi

stop_server

"$PYTHON" - "$DATA" <<'PY'
import pathlib
import struct
import sys

data = pathlib.Path(sys.argv[1])
segment = data / "topics" / "events" / "00000000000000000002"
log_path = segment.with_suffix(".log")
idx_path = segment.with_suffix(".idx")
offset = log_path.stat().st_size
with log_path.open("ab") as f:
    f.write(struct.pack("<I", 6) + b"orphan")
with idx_path.open("ab") as f:
    f.write(struct.pack("<Q", offset))
PY

start_server

"$PYTHON" - "$PORT" <<'PY'
import socket
import struct
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

def encode(*args):
    out = f"*{len(args)}\r\n".encode()
    for arg in args:
        if isinstance(arg, str):
            arg = arg.encode()
        out += b"$%d\r\n" % len(arg) + arg + b"\r\n"
    return out

def read_line(sock):
    data = b""
    while not data.endswith(b"\r\n"):
        chunk = sock.recv(1)
        if not chunk:
            raise EOFError("connection closed")
        data += chunk
    return data

def read_resp(sock):
    line = read_line(sock)
    typ = line[:1]
    if typ in b"+-:":
        return line
    if typ == b"$":
        size = int(line[1:-2])
        data = b""
        while len(data) < size + 2:
            data += sock.recv(size + 2 - len(data))
        return line + data
    if typ == b"*":
        count = int(line[1:-2])
        return line + b"".join(read_resp(sock) for _ in range(count))
    raise AssertionError(f"unknown RESP line: {line!r}")

def request(sock, expected, *args):
    sock.sendall(encode(*args))
    got = read_resp(sock)
    if got != expected:
        raise AssertionError(f"{args}: expected {expected!r}, got {got!r}")

client = connect_retry()
request(client, b":3\r\n", "PRODUCE", "events", "after")
request(
    client,
    b"*4\r\n*2\r\n:0\r\n$5\r\nhello\r\n*2\r\n:1\r\n$5\r\nworld\r\n*2\r\n:2\r\n$5\r\nagain\r\n*2\r\n:3\r\n$5\r\nafter\r\n",
    "FETCH",
    "events",
    "0",
    "4096",
)
raw_batch = struct.pack("<I", 5) + b"hello" + struct.pack("<I", 5) + b"world"
request(
    client,
    b"$%d\r\n" % len(raw_batch) + raw_batch + b"\r\n",
    "FETCHBATCH",
    "events",
    "0",
    "4096",
)
request(
    client,
    b"*2\r\n*2\r\n:2\r\n$5\r\nagain\r\n*2\r\n:3\r\n$5\r\nafter\r\n",
    "FETCH",
    "events",
    "2",
    "4096",
)
raw_again = struct.pack("<I", 5) + b"again" + struct.pack("<I", 5) + b"after"
request(
    client,
    b"$%d\r\n" % len(raw_again) + raw_again + b"\r\n",
    "FETCHBATCH",
    "events",
    "2",
    "4096",
)
request(client, b":3\r\n", "OFFSET", "group1", "events")
client.close()
PY

echo 'logbus checks passed'
