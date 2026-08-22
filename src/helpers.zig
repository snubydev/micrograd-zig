const std = @import("std");

pub fn testHeader(src: std.builtin.SourceLocation) void {
    std.debug.print("\n[{s}:{d}] {s}\n", .{ src.file, src.line, src.fn_name });
}
