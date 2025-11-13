const std = @import("std");

const engine = @import("engine.zig");
const value = @import("engine.zig").value;

// const micrograd_zig = @import("micrograd_zig");

pub fn main() !void {
    std.debug.print("hello\n", .{});

    // const allocator = std.heap.page_allocator;

    var x1 = value(1.5, "x1");
    var x2 = value(-5.0, "x2");

    x1.printL();
    x2.printL();

    var y1 = x1.add(&x2, "y1");
    var y2 = x2.add(&x2, "y2");

    const y3 = y1.add(&y2, "y3");

    y3.printL();

    engine.GenerateGraph(&y3);
}
