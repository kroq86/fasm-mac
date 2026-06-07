#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_DIR="$(mktemp -d "${TMPDIR:-/tmp}/logvec-check.XXXXXX")"
SERVER_PID=
trap 'if [[ -n "${SERVER_PID:-}" ]]; then kill "$SERVER_PID" 2>/dev/null || true; wait "$SERVER_PID" 2>/dev/null || true; fi; rm -rf "$OUT_DIR"' EXIT

PYTHON="${PYTHON:-python3}"

"$ROOT/scripts/check_logvec_search.sh"

if ! command -v zig >/dev/null 2>&1; then
    echo 'FAIL zig is required for logvec check' >&2
    exit 1
fi

CORE_OBJ="$OUT_DIR/logvec_core.o"
LOGVEC="$OUT_DIR/logvec"
LOGBUS="$OUT_DIR/logbus"
DATA="$OUT_DIR/data"
PAYLOAD_DIR="$OUT_DIR/payloads"
BAD_PAYLOAD_DIR="$OUT_DIR/bad-payload"
DIM_MISMATCH_DIR="$OUT_DIR/dim-mismatch"
ZERO_NORM_DIR="$OUT_DIR/zero-norm"
QUERY="$OUT_DIR/query.bin"
ZERO_QUERY="$OUT_DIR/zero-query.bin"
EXPECTED="$OUT_DIR/expected.txt"
PAYLOAD_INDEX="$OUT_DIR/payload.lv"
HOST_INDEX="$OUT_DIR/host.lv"
DIR_INDEX="$OUT_DIR/dir.lv"

PORT="$("$PYTHON" - <<'PY'
import socket

with socket.socket() as s:
    s.bind(("127.0.0.1", 0))
    print(s.getsockname()[1])
PY
)"

fasm --emit=macho-obj "$ROOT/fasm/apps/logvec_core.asm" "$CORE_OBJ" >/dev/null
zig build-exe \
    "$ROOT/fasm/apps/logvec.zig" \
    "$CORE_OBJ" \
    -target x86_64-macos \
    -mcpu=baseline \
    -O ReleaseSafe \
    -femit-bin="$LOGVEC"
fasm "$ROOT/fasm/apps/logbus.asm" "$LOGBUS" >/dev/null

"$PYTHON" - "$OUT_DIR" <<'PY'
import pathlib
import struct
import sys

root = pathlib.Path(sys.argv[1])
payload_dir = root / "payloads"
bad_payload_dir = root / "bad-payload"
dim_mismatch_dir = root / "dim-mismatch"
zero_norm_dir = root / "zero-norm"
for path in (payload_dir, bad_payload_dir, dim_mismatch_dir, zero_norm_dir):
    path.mkdir(parents=True, exist_ok=True)

def payload(vec, doc_id=None):
    out = struct.pack("<I", len(vec))
    out += struct.pack("<" + "f" * len(vec), *vec)
    if doc_id is not None:
        out += struct.pack("<Q", doc_id)
    return out

(payload_dir / "00-x-doc.bin").write_bytes(payload([1.0, 0.0, 0.0, 0.0], 100))
(payload_dir / "01-y-auto.bin").write_bytes(payload([0.0, 1.0, 0.0, 0.0]))
(payload_dir / "02-near-auto.bin").write_bytes(payload([0.9, 0.1, 0.0, 0.0]))
(root / "query.bin").write_bytes(struct.pack("<ffff", 1.0, 0.0, 0.0, 0.0))
(root / "zero-query.bin").write_bytes(struct.pack("<ffff", 0.0, 0.0, 0.0, 0.0))
(root / "expected.txt").write_text("100 1.000000\n2 0.993884\n", encoding="utf-8")

(bad_payload_dir / "bad.bin").write_bytes(struct.pack("<I", 4) + b"x")
(dim_mismatch_dir / "00.bin").write_bytes(payload([1.0, 0.0, 0.0, 0.0]))
(dim_mismatch_dir / "01.bin").write_bytes(payload([1.0, 0.0, 0.0]))
(zero_norm_dir / "00.bin").write_bytes(payload([0.0, 0.0, 0.0, 0.0]))
PY

"$LOGVEC" build-index --payload-dir "$PAYLOAD_DIR" --out "$PAYLOAD_INDEX"
"$LOGVEC" search --index "$PAYLOAD_INDEX" --query "$QUERY" --top 2 >"$OUT_DIR/payload.out"
diff -u "$EXPECTED" "$OUT_DIR/payload.out"

arch -x86_64 "$LOGBUS" --dir "$DATA" --port "$PORT" --bind 127.0.0.1 --segment-bytes 64 >"$OUT_DIR/logbus.out" 2>"$OUT_DIR/logbus.err" &
SERVER_PID=$!

"$PYTHON" - "$PORT" "$PAYLOAD_DIR" <<'PY'
import pathlib
import socket
import sys
import time

port = int(sys.argv[1])
payload_dir = pathlib.Path(sys.argv[2])

def connect_retry():
    last = None
    for _ in range(80):
        try:
            return socket.create_connection(("127.0.0.1", port), timeout=0.5)
        except OSError as exc:
            last = exc
            time.sleep(0.05)
    raise RuntimeError(f"logbus did not accept connections: {last}")

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

client = connect_retry()
for i, path in enumerate(sorted(payload_dir.iterdir())):
    client.sendall(encode("PRODUCE", "embeddings", path.read_bytes()))
    got = read_line(client)
    want = f":{i}\r\n".encode()
    if got != want:
        raise AssertionError(f"expected {want!r}, got {got!r}")
client.close()
PY

"$LOGVEC" build-index --host 127.0.0.1 --port "$PORT" --topic embeddings --out "$HOST_INDEX"
"$LOGVEC" search --index "$HOST_INDEX" --query "$QUERY" --top 2 >"$OUT_DIR/host.out"
diff -u "$EXPECTED" "$OUT_DIR/host.out"

"$LOGVEC" build-index --dir "$DATA" --topic embeddings --out "$DIR_INDEX"
"$LOGVEC" search --index "$DIR_INDEX" --query "$QUERY" --top 2 >"$OUT_DIR/dir.out"
diff -u "$OUT_DIR/host.out" "$OUT_DIR/dir.out"

if "$LOGVEC" build-index --payload-dir "$BAD_PAYLOAD_DIR" --out "$OUT_DIR/bad.lv" >/dev/null 2>&1; then
    echo 'FAIL malformed payload unexpectedly built an index' >&2
    exit 1
fi
if "$LOGVEC" build-index --payload-dir "$DIM_MISMATCH_DIR" --out "$OUT_DIR/mismatch.lv" >/dev/null 2>&1; then
    echo 'FAIL dim mismatch unexpectedly built an index' >&2
    exit 1
fi
if "$LOGVEC" build-index --payload-dir "$ZERO_NORM_DIR" --out "$OUT_DIR/zero.lv" >/dev/null 2>&1; then
    echo 'FAIL zero-norm vector unexpectedly built an index' >&2
    exit 1
fi
if "$LOGVEC" search --index "$HOST_INDEX" --query "$ZERO_QUERY" --top 2 >/dev/null 2>&1; then
    echo 'FAIL zero-norm query unexpectedly searched' >&2
    exit 1
fi

"$PYTHON" - "$HOST_INDEX" "$OUT_DIR/bad-norm.lv" <<'PY'
import pathlib
import sys

src = pathlib.Path(sys.argv[1])
dst = pathlib.Path(sys.argv[2])
body = bytearray(src.read_bytes())
body[48:52] = b"\x00\x00\x00\x00"
dst.write_bytes(body)
PY

if "$LOGVEC" search --index "$OUT_DIR/bad-norm.lv" --query "$QUERY" --top 2 >/dev/null 2>&1; then
    echo 'FAIL bad index norm unexpectedly searched' >&2
    exit 1
fi

"$PYTHON" - "$DATA" <<'PY'
import pathlib
import sys

data = pathlib.Path(sys.argv[1])
log_path = sorted((data / "topics" / "embeddings").glob("*.log"))[0]
body = bytearray(log_path.read_bytes())
body[8] ^= 0xFF
log_path.write_bytes(body)
PY

if "$LOGVEC" build-index --dir "$DATA" --topic embeddings --out "$OUT_DIR/corrupt.lv" >/dev/null 2>&1; then
    echo 'FAIL corrupt segment CRC unexpectedly built an index' >&2
    exit 1
fi

echo 'logvec checks passed'
