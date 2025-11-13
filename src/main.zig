const std = @import("std");

const engine = @import("engine.zig");
const value = @import("engine.zig").value;

// const micrograd_zig = @import("micrograd_zig");

pub fn main() !void {
    // const allocator = std.heap.page_allocator;

    var x1 = value(2.0, "x1");
    var x2 = value(-5.0, "x2");
    var y1 = x1.add(&x2, "y1");
    var y2 = x2.mul(&x2, "y2");
    var Y = y1.add(&y2, "y3");
    Y.backward();

    std.debug.print("Y = {d:.5}\n", .{Y.data});
    engine.GenerateGraph(&Y);

    const h: f32 = 0.001;

    var x11 = value(2.0, "x1");
    var x12 = value(-5.0 + h, "x2");
    var y11 = x11.add(&x12, "y1");
    var y12 = x12.mul(&x12, "y2");
    var Y2 = y11.add(&y12, "y3");
    Y2.backward();

    std.debug.print("Y2 = {d:.5}\n", .{Y2.data});
    engine.GenerateGraph(&Y2);
    std.debug.print("(Y2 - Y) / h = {d:.5}\n", .{(Y2.data - Y.data) / h});
}
