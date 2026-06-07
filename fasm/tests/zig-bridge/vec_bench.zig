const std = @import("std");

extern fn lb_dot_product(a: [*]const f64, b: [*]const f64, len: u64) callconv(.c) f64;
extern fn lb_vector_norm(v: [*]const f64, len: u64) callconv(.c) f64;

const dim: usize = 128;
const db_len: usize = 256;
const dot_iters: usize = 500_000;
const search_iters: usize = 400;

const min_bad_over_good: f64 = 2.0;
const max_good_dot_ratio: f64 = 2.5;

fn rdtsc() u64 {
    var lo: u32 = undefined;
    var hi: u32 = undefined;
    asm volatile ("rdtsc"
        : [_] "={eax}" (lo),
          [_] "={edx}" (hi),
    );
    return (@as(u64, hi) << 32) | lo;
}

fn dotProductZig(a: []const f64, b: []const f64) f64 {
    var sum: f64 = 0;
    for (a, b) |x, y| sum += x * y;
    return sum;
}

fn expectApprox(actual: f64, expected: f64, tol: f64) !void {
    const delta = @abs(actual - expected);
    if (delta > tol) {
        std.debug.print("approx mismatch: expected {d}, got {d}\n", .{ expected, actual });
        return error.BadApprox;
    }
}

fn fillVectors(query: *[dim]f64, database: *[db_len][dim]f64) void {
    var i: usize = 0;
    while (i < dim) : (i += 1) {
        const x = @as(f64, @floatFromInt(i + 1));
        query[i] = x * 0.01;
    }
    var row: usize = 0;
    while (row < db_len) : (row += 1) {
        var col: usize = 0;
        while (col < dim) : (col += 1) {
            const base = @as(f64, @floatFromInt((row + 1) * (col + 3)));
            database[row][col] = @sin(base * 0.001);
        }
    }
}

fn precomputeNorms(database: *const [db_len][dim]f64, out: *[db_len]f64) void {
    var i: usize = 0;
    while (i < db_len) : (i += 1) {
        out[i] = lb_vector_norm(database[i][0..].ptr, dim);
    }
}

fn benchFasmDot(a: []const f64, b: []const f64, iters: usize) u64 {
    const start = rdtsc();
    var sink: f64 = 0;
    var i: usize = 0;
    while (i < iters) : (i += 1) {
        sink += lb_dot_product(a.ptr, b.ptr, dim);
    }
    std.mem.doNotOptimizeAway(sink);
    return rdtsc() - start;
}

fn benchZigNativeDot(a: []const f64, b: []const f64, iters: usize) u64 {
    const start = rdtsc();
    var sink: f64 = 0;
    var i: usize = 0;
    while (i < iters) : (i += 1) {
        sink += dotProductZig(a, b);
    }
    std.mem.doNotOptimizeAway(sink);
    return rdtsc() - start;
}

fn benchZigGoodSearch(
    query: *const [dim]f64,
    database: *const [db_len][dim]f64,
    norms: *const [db_len]f64,
    q_norm: f64,
    iters: usize,
) u64 {
    const start = rdtsc();
    var sink: f64 = 0;
    var i: usize = 0;
    while (i < iters) : (i += 1) {
        var row: usize = 0;
        while (row < db_len) : (row += 1) {
            const dot = lb_dot_product(query[0..].ptr, database[row][0..].ptr, dim);
            const denom = q_norm * norms[row];
            if (denom != 0) sink += dot / denom;
        }
    }
    std.mem.doNotOptimizeAway(sink);
    return rdtsc() - start;
}

fn benchPythonStyleSearch(
    query: *const [dim]f64,
    database: *const [db_len][dim]f64,
    iters: usize,
) u64 {
    const start = rdtsc();
    var sink: f64 = 0;
    var i: usize = 0;
    while (i < iters) : (i += 1) {
        var row: usize = 0;
        while (row < db_len) : (row += 1) {
            var query_copy: [dim]f64 = undefined;
            var vector_copy: [dim]f64 = undefined;
            @memcpy(query_copy[0..], query[0..]);
            @memcpy(vector_copy[0..], database[row][0..]);

            const dot = lb_dot_product(query_copy[0..].ptr, vector_copy[0..].ptr, dim);
            const norm_q = lb_vector_norm(query_copy[0..].ptr, dim);
            const norm_v = lb_vector_norm(vector_copy[0..].ptr, dim);
            if (norm_q != 0 and norm_v != 0) sink += dot / (norm_q * norm_v);
        }
    }
    std.mem.doNotOptimizeAway(sink);
    return rdtsc() - start;
}

pub fn main() !void {
    var query: [dim]f64 = undefined;
    var database: [db_len][dim]f64 = undefined;
    var norms: [db_len]f64 = undefined;
    fillVectors(&query, &database);
    precomputeNorms(&database, &norms);

    const q_norm = lb_vector_norm(query[0..].ptr, dim);
    const expected_dot = dotProductZig(query[0..], database[0][0..]);
    const got_dot = lb_dot_product(query[0..].ptr, database[0][0..].ptr, dim);
    try expectApprox(got_dot, expected_dot, 1e-9);
    try expectApprox(q_norm, lb_vector_norm(query[0..].ptr, dim), 1e-9);

    _ = benchFasmDot(query[0..], database[0][0..], 1024);
    _ = benchZigNativeDot(query[0..], database[0][0..], 1024);
    _ = benchZigGoodSearch(&query, &database, &norms, q_norm, 8);
    _ = benchPythonStyleSearch(&query, &database, 8);

    const ns_fasm_dot = benchFasmDot(query[0..], database[0][0..], dot_iters);
    const ns_zig_dot = benchZigNativeDot(query[0..], database[0][0..], dot_iters);
    const ns_good = benchZigGoodSearch(&query, &database, &norms, q_norm, search_iters);
    const ns_bad = benchPythonStyleSearch(&query, &database, search_iters);

    const fasm_vs_zig_native = @as(f64, @floatFromInt(ns_fasm_dot)) / @as(f64, @floatFromInt(ns_zig_dot));
    const bad_over_good = @as(f64, @floatFromInt(ns_bad)) / @as(f64, @floatFromInt(ns_good));
    const per_dot = @as(f64, @floatFromInt(ns_fasm_dot)) / @as(f64, @floatFromInt(dot_iters));
    const per_good = @as(f64, @floatFromInt(ns_good)) /
        (@as(f64, @floatFromInt(search_iters)) * @as(f64, @floatFromInt(db_len)));
    const good_dot_ratio = per_good / per_dot;

    std.debug.print(
        "zig-bridge vec correctness ok\n",
        .{},
    );
    std.debug.print(
        "zig-bridge perf fasm_vs_zig_native={d:.2} good_dot_ratio={d:.2} bad_over_good={d:.2}\n",
        .{ fasm_vs_zig_native, good_dot_ratio, bad_over_good },
    );

    if (good_dot_ratio > max_good_dot_ratio) {
        std.debug.print(
            "FAIL zig good search too heavy vs one dot: {d:.2} > {d:.2}\n",
            .{ good_dot_ratio, max_good_dot_ratio },
        );
        return error.GoodPatternTooHeavy;
    }
    if (bad_over_good < min_bad_over_good) {
        std.debug.print(
            "FAIL python-style pattern not slower enough: {d:.2} < {d:.2}\n",
            .{ bad_over_good, min_bad_over_good },
        );
        return error.PythonStyleTooFast;
    }
}
