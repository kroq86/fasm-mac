const std = @import("std");

const magic: [8]u8 = .{ 'L', 'O', 'G', 'V', 'E', 'C', '1', 0 };
const version: u32 = 1;
const dim_min: u32 = 1;
const dim_max: u32 = 4096;

extern fn lb_vec_dot_f32(a: [*]const f32, b: [*]const f32, len: u64) callconv(.c) f32;
extern fn lb_vec_norm_f32(v: [*]const f32, len: u64) callconv(.c) f32;
extern fn lb_vec_topk_cosine_exact(
    query: [*]const f32,
    vectors: [*]const f32,
    norms: [*]const f32,
    count: u64,
    dim: u64,
    k: u64,
    out_index: [*]u32,
    out_score: [*]f32,
) callconv(.c) i32;
extern fn lb_logvec_payload_validate(
    payload: [*]const u8,
    len: u64,
    out_dim: ?*u32,
    out_doc_id: ?*u64,
    out_vector: ?*[*]const f32,
) callconv(.c) i32;

fn f32Align4(slice: anytype) [*]const f32 {
    return @alignCast(@ptrCast(slice.ptr));
}

fn readWholeFile(io: std.Io, allocator: std.mem.Allocator, path: []const u8, limit: usize) ![]u8 {
    return std.Io.Dir.readFileAlloc(.cwd(), io, path, allocator, .limited(limit));
}

fn writeWholeFile(io: std.Io, path: []const u8, data: []const u8) !void {
    try std.Io.Dir.writeFile(.cwd(), io, .{ .sub_path = path, .data = data });
}

const IndexItem = struct {
    doc_id: u64,
    vector: []const f32,
};

const Entry = struct {
    doc_id: u64,
    norm: f32,
    vector_off: u64,
};

const Index = struct {
    data: []align(1) const u8,
    dim: u32,
    count: u64,

    // magic[8] version u32 dim u32 count u64 flags u64 reserved u64
    fn headerSize() comptime_int {
        return magic.len + 4 + 4 + 8 + 8 + 8;
    }

    // doc_id u64 norm f32 reserved u32 vector f32[dim]
    fn recordSize(dim: u32) u64 {
        return 16 + @as(u64, dim) * 4;
    }

    fn parse(data: []align(1) const u8) !Index {
        if (data.len < headerSize()) return error.BadIndex;
        if (!std.mem.eql(u8, data[0..magic.len], &magic)) return error.BadMagic;
        const ver = std.mem.readInt(u32, data[magic.len..][0..4], .little);
        if (ver != version) return error.BadVersion;
        const dim = std.mem.readInt(u32, data[magic.len + 4 ..][0..4], .little);
        if (dim < dim_min or dim > dim_max) return error.BadDim;
        const count = std.mem.readInt(u64, data[magic.len + 8 ..][0..8], .little);
        const need = headerSize() + count * recordSize(dim);
        if (data.len < need) return error.TruncatedIndex;
        return .{ .data = data, .dim = dim, .count = count };
    }

    fn entry(self: Index, i: u64) Entry {
        const off = headerSize() + i * recordSize(self.dim);
        const doc_id = std.mem.readInt(u64, self.data[off..][0..8], .little);
        const norm = @as(f32, @bitCast(std.mem.readInt(u32, self.data[off + 8 ..][0..4], .little)));
        return .{
            .doc_id = doc_id,
            .norm = norm,
            .vector_off = off + 16,
        };
    }

    fn vectorSlice(self: Index, ent: Entry) []const f32 {
        const ptr: [*]const f32 = @alignCast(@ptrCast(self.data.ptr + ent.vector_off));
        return ptr[0..self.dim];
    }

    fn buildBytes(allocator: std.mem.Allocator, dim: u32, items: []const IndexItem) ![]u8 {
        var out = try std.array_list.Managed(u8).initCapacity(allocator, 4096);
        errdefer out.deinit();
        try out.appendSlice(&magic);
        var buf: [8]u8 = undefined;
        std.mem.writeInt(u32, buf[0..4], version, .little);
        try out.appendSlice(buf[0..4]);
        std.mem.writeInt(u32, buf[0..4], dim, .little);
        try out.appendSlice(buf[0..4]);
        std.mem.writeInt(u64, buf[0..8], @intCast(items.len), .little);
        try out.appendSlice(buf[0..8]);
        std.mem.writeInt(u64, buf[0..8], 0, .little);
        try out.appendSlice(buf[0..8]);
        try out.appendSlice(buf[0..8]);
        for (items) |item| {
            std.mem.writeInt(u64, buf[0..8], item.doc_id, .little);
            try out.appendSlice(buf[0..8]);
            const norm = lb_vec_norm_f32(f32Align4(item.vector), dim);
            if (norm == 0) return error.ZeroNorm;
            std.mem.writeInt(u32, buf[0..4], @bitCast(norm), .little);
            try out.appendSlice(buf[0..4]);
            std.mem.writeInt(u32, buf[0..4], 0, .little);
            try out.appendSlice(buf[0..4]);
            try out.appendSlice(std.mem.sliceAsBytes(item.vector));
        }
        return try out.toOwnedSlice();
    }

    fn write(io: std.Io, path: []const u8, allocator: std.mem.Allocator, dim: u32, items: []const IndexItem) !void {
        const bytes = try buildBytes(allocator, dim, items);
        defer allocator.free(bytes);
        try writeWholeFile(io, path, bytes);
    }
};

const IngestRecord = struct {
    topic_record_offset: u64,
    payload: []const u8,
};

const ParsedPayload = struct {
    dim: u32,
    doc_id: u64,
    vector: []const f32,
};

const doc_id_auto: u64 = std.math.maxInt(u64);

fn parsePayload(payload: []const u8) !ParsedPayload {
    var dim: u32 = undefined;
    var doc_id: u64 = undefined;
    var vector_ptr: [*]const f32 = undefined;
    if (lb_logvec_payload_validate(payload.ptr, payload.len, &dim, &doc_id, &vector_ptr) != 0) {
        return error.BadPayload;
    }
    const vector = vector_ptr[0..dim];
    const norm = lb_vec_norm_f32(f32Align4(vector), dim);
    if (norm == 0) return error.ZeroNorm;
    return .{ .dim = dim, .doc_id = doc_id, .vector = vector };
}

fn crc32c(data: []const u8) u32 {
    var crc: u32 = 0xFFFF_FFFF;
    for (data) |b| {
        crc ^= b;
        var bit: u5 = 0;
        while (bit < 8) : (bit += 1) {
            if (crc & 1 != 0) crc = (crc >> 1) ^ 0x82F63B78 else crc >>= 1;
        }
    }
    return crc ^ 0xFFFF_FFFF;
}

const net = std.Io.net;

fn readLogbusRecord(reader: *std.Io.Reader, payload_out: *std.array_list.Managed(u8)) !enum { ok, eof, bad_crc } {
    var hdr: [8]u8 = undefined;
    const hn = try reader.readSliceShort(&hdr);
    if (hn == 0) return .eof;
    if (hn != 8) return error.TruncatedRecord;
    const plen = std.mem.readInt(u32, hdr[0..4], .little);
    const want_crc = std.mem.readInt(u32, hdr[4..8], .little);
    payload_out.clearRetainingCapacity();
    try payload_out.ensureTotalCapacityPrecise(plen);
    payload_out.items.len = plen;
    if (plen > 0) try reader.readSliceAll(payload_out.items);
    if (crc32c(payload_out.items) != want_crc) return .bad_crc;
    return .ok;
}

fn appendLogbusBatchRecords(
    allocator: std.mem.Allocator,
    batch: []const u8,
    base_offset: u64,
    out: *std.array_list.Managed(IngestRecord),
) !u64 {
    var pos: usize = 0;
    var offset = base_offset;
    while (pos < batch.len) : (offset += 1) {
        if (batch.len - pos < 8) return error.TruncatedRecord;
        const plen = std.mem.readInt(u32, batch[pos..][0..4], .little);
        const want_crc = std.mem.readInt(u32, batch[pos + 4 ..][0..4], .little);
        pos += 8;
        if (batch.len - pos < plen) return error.TruncatedRecord;
        const payload = batch[pos .. pos + plen];
        pos += plen;
        if (crc32c(payload) != want_crc) return error.BadCrc;
        try out.append(.{
            .topic_record_offset = offset,
            .payload = try allocator.dupe(u8, payload),
        });
    }
    return offset - base_offset;
}

const FetchClient = struct {
    stream: net.Stream,
    io: std.Io,
    allocator: std.mem.Allocator,
    read_buf: []u8,
    write_buf: [4096]u8,
    reader_state: net.Stream.Reader,

    fn connect(allocator: std.mem.Allocator, io: std.Io, host: []const u8, port: u16) !FetchClient {
        const addr = try net.IpAddress.parse(host, port);
        const stream = try net.IpAddress.connect(&addr, io, .{ .mode = .stream });
        errdefer stream.close(io);
        const read_buf = try allocator.alloc(u8, 4096);
        errdefer allocator.free(read_buf);
        return .{
            .stream = stream,
            .io = io,
            .allocator = allocator,
            .read_buf = read_buf,
            .write_buf = undefined,
            .reader_state = stream.reader(io, read_buf),
        };
    }

    fn deinit(self: *FetchClient) void {
        self.stream.close(self.io);
        self.allocator.free(self.read_buf);
    }

    fn reader(self: *FetchClient) *net.Stream.Reader {
        return &self.reader_state;
    }

    fn writeAll(self: *FetchClient, bytes: []const u8) !void {
        var w = self.stream.writer(self.io, &self.write_buf);
        try w.interface.writeAll(bytes);
        try w.interface.flush();
    }

    fn readLine(self: *FetchClient, buf: *std.array_list.Managed(u8)) !?[]const u8 {
        const r = self.reader();
        buf.clearRetainingCapacity();
        while (true) {
            var byte: [1]u8 = undefined;
            const n = r.interface.readSliceShort(&byte) catch |err| switch (err) {
                error.ReadFailed => return r.err.?,
            };
            if (n == 0) return null;
            try buf.append(byte[0]);
            if (buf.items.len >= 2 and buf.items[buf.items.len - 2] == '\r' and buf.items[buf.items.len - 1] == '\n') {
                return buf.items[0 .. buf.items.len - 2];
            }
        }
    }

    fn encode(self: *FetchClient, allocator: std.mem.Allocator, args: []const []const u8) !void {
        var msg = std.array_list.Managed(u8).init(allocator);
        defer msg.deinit();
        var scratch: [64]u8 = undefined;
        const head = try std.fmt.bufPrint(&scratch, "*{d}\r\n", .{args.len});
        try msg.appendSlice(head);
        for (args) |arg| {
            const bulk = try std.fmt.bufPrint(&scratch, "${d}\r\n", .{arg.len});
            try msg.appendSlice(bulk);
            try msg.appendSlice(arg);
            try msg.appendSlice("\r\n");
        }
        try self.writeAll(msg.items);
    }

    fn readExact(self: *FetchClient, buf: []u8) !void {
        const r = self.reader();
        try r.interface.readSliceAll(buf);
    }

    fn readBulk(self: *FetchClient, allocator: std.mem.Allocator, buf: *std.array_list.Managed(u8)) ![]u8 {
        const line = (try self.readLine(buf)) orelse return error.ProtocolError;
        if (line.len == 0) return error.ProtocolError;
        if (line[0] == '-') return error.ProtocolError;
        if (line[0] != '$') return error.ProtocolError;
        const size = try std.fmt.parseInt(usize, line[1..], 10);
        const payload = try allocator.alloc(u8, size);
        errdefer allocator.free(payload);
        if (size > 0) try self.readExact(payload);
        var crlf: [2]u8 = undefined;
        try self.readExact(&crlf);
        if (crlf[0] != '\r' or crlf[1] != '\n') return error.ProtocolError;
        return payload;
    }

    fn fetchPayloads(self: *FetchClient, allocator: std.mem.Allocator, topic: []const u8) !std.array_list.Managed(IngestRecord) {
        var out = std.array_list.Managed(IngestRecord).init(allocator);
        errdefer {
            for (out.items) |rec| allocator.free(rec.payload);
            out.deinit();
        }
        var next_offset: u64 = 0;
        var buf = std.array_list.Managed(u8).init(allocator);
        defer buf.deinit();
        var off_print: [32]u8 = undefined;
        while (true) {
            const off_str = try std.fmt.bufPrint(&off_print, "{d}", .{next_offset});
            try self.encode(allocator, &.{ "FETCHBATCH", topic, off_str, "1048576" });
            const batch = try self.readBulk(allocator, &buf);
            defer allocator.free(batch);
            if (batch.len == 0) break;
            const got = try appendLogbusBatchRecords(allocator, batch, next_offset, &out);
            if (got == 0) break;
            next_offset += got;
        }
        return out;
    }
};

fn resolveDocId(explicit: u64, topic_record_offset: u64) u64 {
    if (explicit == doc_id_auto) return topic_record_offset;
    return explicit;
}

fn buildIndexFromRecords(io: std.Io, allocator: std.mem.Allocator, records: []const IngestRecord, out_path: []const u8) !void {
    if (records.len == 0) return error.EmptyInput;
    var parsed = try allocator.alloc(ParsedPayload, records.len);
    defer allocator.free(parsed);
    var i: usize = 0;
    while (i < records.len) : (i += 1) {
        parsed[i] = try parsePayload(records[i].payload);
        if (i > 0 and parsed[i].dim != parsed[0].dim) return error.DimMismatch;
    }
    const dim = parsed[0].dim;
    var items = try allocator.alloc(IndexItem, records.len);
    defer allocator.free(items);
    i = 0;
    while (i < records.len) : (i += 1) {
        items[i] = .{
            .doc_id = resolveDocId(parsed[i].doc_id, records[i].topic_record_offset),
            .vector = parsed[i].vector,
        };
    }
    try Index.write(io, out_path, allocator, dim, items);
}

fn collectDirLogPayloads(io: std.Io, allocator: std.mem.Allocator, root: []const u8, topic: []const u8) !std.array_list.Managed(IngestRecord) {
    const dir_path = try std.fs.path.join(allocator, &.{ root, "topics", topic });
    defer allocator.free(dir_path);
    var dir = try std.Io.Dir.openDirAbsolute(io, dir_path, .{ .iterate = true });
    defer dir.close(io);
    var logs: std.array_list.Managed([]const u8) = .init(allocator);
    defer {
        for (logs.items) |n| allocator.free(n);
        logs.deinit();
    }
    var it = dir.iterate();
    while (try it.next(io)) |entry| {
        if (entry.kind != .file) continue;
        if (!std.mem.endsWith(u8, entry.name, ".log")) continue;
        try logs.append(try allocator.dupe(u8, entry.name));
    }
    std.mem.sort([]const u8, logs.items, {}, struct {
        fn less(_: void, a: []const u8, b: []const u8) bool {
            return std.mem.order(u8, a, b) == .lt;
        }
    }.less);
    var out = std.array_list.Managed(IngestRecord).init(allocator);
    errdefer {
        for (out.items) |rec| allocator.free(rec.payload);
        out.deinit();
    }
    var payload_buf = std.array_list.Managed(u8).init(allocator);
    defer payload_buf.deinit();
    var topic_record_offset: u64 = 0;
    for (logs.items) |name| {
        const log_path = try std.fs.path.join(allocator, &.{ dir_path, name });
        defer allocator.free(log_path);
        var file = try std.Io.Dir.openFileAbsolute(io, log_path, .{});
        defer file.close(io);
        var read_buf: [65536]u8 = undefined;
        var reader = file.reader(io, &read_buf);
        while (true) {
            switch (try readLogbusRecord(&reader.interface, &payload_buf)) {
                .ok => {
                    try out.append(.{
                        .topic_record_offset = topic_record_offset,
                        .payload = try allocator.dupe(u8, payload_buf.items),
                    });
                    topic_record_offset += 1;
                },
                .eof => break,
                .bad_crc => return error.BadCrc,
            }
        }
    }
    return out;
}

fn collectPayloadDir(io: std.Io, allocator: std.mem.Allocator, dir_path: []const u8) !std.array_list.Managed(IngestRecord) {
    var dir = try std.Io.Dir.openDirAbsolute(io, dir_path, .{ .iterate = true });
    defer dir.close(io);
    var names: std.array_list.Managed([]const u8) = .init(allocator);
    defer {
        for (names.items) |n| allocator.free(n);
        names.deinit();
    }
    var it = dir.iterate();
    while (try it.next(io)) |entry| {
        if (entry.kind != .file) continue;
        try names.append(try allocator.dupe(u8, entry.name));
    }
    std.mem.sort([]const u8, names.items, {}, struct {
        fn less(_: void, a: []const u8, b: []const u8) bool {
            return std.mem.order(u8, a, b) == .lt;
        }
    }.less);
    var out = std.array_list.Managed(IngestRecord).init(allocator);
    errdefer {
        for (out.items) |rec| allocator.free(rec.payload);
        out.deinit();
    }
    var i: u64 = 0;
    for (names.items) |name| {
        const path = try std.fs.path.join(allocator, &.{ dir_path, name });
        defer allocator.free(path);
        try out.append(.{
            .topic_record_offset = i,
            .payload = try readWholeFile(io, allocator, path, 16 * 1024 * 1024),
        });
        i += 1;
    }
    return out;
}

const SearchHit = struct {
    doc_id: u64,
    score: f32,
};

fn searchIndex(io: std.Io, allocator: std.mem.Allocator, index_path: []const u8, query_path: []const u8, top_k: u32) ![]SearchHit {
    const data = try readWholeFile(io, allocator, index_path, 256 * 1024 * 1024);
    defer allocator.free(data);
    const index = try Index.parse(data);
    const qbytes = try readWholeFile(io, allocator, query_path, 64 * 1024);
    defer allocator.free(qbytes);
    if (qbytes.len != index.dim * 4) return error.QueryDimMismatch;
    const query = std.mem.bytesAsSlice(f32, qbytes);
    const qnorm = lb_vec_norm_f32(f32Align4(query), index.dim);
    std.mem.doNotOptimizeAway(lb_vec_dot_f32(f32Align4(query), f32Align4(query), index.dim));
    if (qnorm == 0) return error.ZeroNorm;
    if (index.count == 0) return try allocator.alloc(SearchHit, 0);
    const k = @min(@as(u64, top_k), index.count);
    const idx_out = try allocator.alloc(u32, k);
    defer allocator.free(idx_out);
    const score_out = try allocator.alloc(f32, k);
    defer allocator.free(score_out);
    const norms = try allocator.alloc(f32, index.count);
    defer allocator.free(norms);
    const vectors = try allocator.alloc(f32, index.count * index.dim);
    defer allocator.free(vectors);
    var i: u64 = 0;
    while (i < index.count) : (i += 1) {
        const ent = index.entry(i);
        if (!std.math.isFinite(ent.norm) or ent.norm <= 0) return error.BadIndexNorm;
        norms[i] = ent.norm;
        const need = ent.vector_off + @as(u64, index.dim) * 4;
        if (need > index.data.len) return error.TruncatedIndex;
        const src = index.data[ent.vector_off..need];
        var j: u32 = 0;
        while (j < index.dim) : (j += 1) {
            const off = @as(usize, @intCast(j * 4));
            vectors[i * index.dim + j] = @as(f32, @bitCast(std.mem.readInt(u32, src[off..][0..4], .little)));
        }
    }
    if (lb_vec_topk_cosine_exact(
        f32Align4(query),
        vectors.ptr,
        norms.ptr,
        index.count,
        index.dim,
        k,
        idx_out.ptr,
        score_out.ptr,
    ) != 0) return error.TopkFailed;
    var hits = try allocator.alloc(SearchHit, k);
    i = 0;
    while (i < k) : (i += 1) {
        const row = idx_out[i];
        if (row >= index.count) return error.TopkFailed;
        hits[i] = .{
            .doc_id = index.entry(row).doc_id,
            .score = score_out[i],
        };
    }
    std.mem.sort(SearchHit, hits, {}, struct {
        fn less(_: void, a: SearchHit, b: SearchHit) bool {
            if (a.score > b.score) return true;
            if (a.score < b.score) return false;
            return a.doc_id < b.doc_id;
        }
    }.less);
    return hits;
}

fn usage() void {
    std.debug.print(
        \\usage:
        \\  logvec search --index PATH --query PATH --top K
        \\  logvec build-index --payload-dir DIR --out PATH
        \\  logvec build-index --host H --port P --topic TOPIC --out PATH
        \\  logvec build-index --dir DATA --topic TOPIC --out PATH
        \\
    , .{});
}

pub fn main(init: std.process.Init) !void {
    const allocator = init.gpa;
    const io = init.io;
    var args = try std.process.Args.Iterator.initAllocator(init.minimal.args, allocator);
    defer args.deinit();
    _ = args.skip();
    const cmd = args.next() orelse {
        usage();
        return error.Usage;
    };
    if (std.mem.eql(u8, cmd, "search")) {
        var index_path: ?[]const u8 = null;
        var query_path: ?[]const u8 = null;
        var top_k: u32 = 5;
        while (args.next()) |arg| {
            if (std.mem.eql(u8, arg, "--index")) {
                index_path = args.next();
            } else if (std.mem.eql(u8, arg, "--query")) {
                query_path = args.next();
            } else if (std.mem.eql(u8, arg, "--top")) {
                top_k = try std.fmt.parseInt(u32, args.next() orelse return error.Usage, 10);
            } else return error.Usage;
        }
        const hits = try searchIndex(io, allocator, index_path orelse return error.Usage, query_path orelse return error.Usage, top_k);
        defer allocator.free(hits);
        var line_buf: [64]u8 = undefined;
        for (hits) |hit| {
            const line = try std.fmt.bufPrint(&line_buf, "{d} {d:.6}\n", .{ hit.doc_id, hit.score });
            try std.Io.File.stdout().writeStreamingAll(io, line);
        }
        return;
    }
    if (std.mem.eql(u8, cmd, "build-index")) {
        var out_path: ?[]const u8 = null;
        var payload_dir: ?[]const u8 = null;
        var host: ?[]const u8 = null;
        var port: u16 = 9092;
        var data_dir: ?[]const u8 = null;
        var topic: ?[]const u8 = null;
        while (args.next()) |arg| {
            if (std.mem.eql(u8, arg, "--out")) {
                out_path = args.next();
            } else if (std.mem.eql(u8, arg, "--payload-dir")) {
                payload_dir = args.next();
            } else if (std.mem.eql(u8, arg, "--host")) {
                host = args.next();
            } else if (std.mem.eql(u8, arg, "--port")) {
                port = try std.fmt.parseInt(u16, args.next() orelse return error.Usage, 10);
            } else if (std.mem.eql(u8, arg, "--dir")) {
                data_dir = args.next();
            } else if (std.mem.eql(u8, arg, "--topic")) {
                topic = args.next();
            } else return error.Usage;
        }
        const out = out_path orelse return error.Usage;
        var records: std.array_list.Managed(IngestRecord) = .init(allocator);
        defer {
            for (records.items) |rec| allocator.free(rec.payload);
            records.deinit();
        }
        if (payload_dir) |pd| {
            records = try collectPayloadDir(io, allocator, pd);
        } else if (host != null and topic != null) {
            var client = try FetchClient.connect(allocator, io, host.?, port);
            defer client.deinit();
            records = try client.fetchPayloads(allocator, topic.?);
        } else if (data_dir != null and topic != null) {
            records = try collectDirLogPayloads(io, allocator, data_dir.?, topic.?);
        } else return error.Usage;
        try buildIndexFromRecords(io, allocator, records.items, out);
        return;
    }
    usage();
    return error.Usage;
}
