const std = @import("std");

extern fn lb_crc32c(ptr: [*]const u8, len: u64) callconv(.c) u32;

const empty_storage = [_]u8{0};
const hello_storage = [_]u8{ 'h', 'e', 'l', 'l', 'o' };
const digits_storage = [_]u8{ '1', '2', '3', '4', '5', '6', '7', '8', '9' };

fn expectCrc(bytes: []const u8, expected: u32) !void {
    const got = lb_crc32c(bytes.ptr, bytes.len);
    if (got != expected) {
        std.debug.print("crc32c mismatch: expected 0x{x}, got 0x{x}\n", .{ expected, got });
        return error.BadCrc;
    }
}

pub fn main() !void {
    try expectCrc(empty_storage[0..0], 0x00000000);
    try expectCrc(hello_storage[0..], 0x9a71bb4c);
    try expectCrc(digits_storage[0..], 0xe3069283);
}
