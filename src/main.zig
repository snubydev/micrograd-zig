const std = @import("std");

const engine = @import("engine.zig");
const value = @import("engine.zig").value;

// const micrograd_zig = @import("micrograd_zig");

pub fn main() !void {
    // const allocator = std.heap.page_allocator;

    var x1 = value(2.0, "x1");
    var x2 = value(3.0, "x2");
    var y1 = x1.add(&x2, "y1");
    var y2 = x2.mul(&x2, "y2");
    var Y = y1.add(&y2, "y3");
    Y.backward();

    std.debug.print("Y = {d:.5}\n", .{Y.data});
    engine.GenerateGraph(&Y);

    const h: f32 = 0.001;
    const Y2 = Y.data;
    x2.data = x2.data + h;

    Y.forward(); // update data vaules bottom to top
    Y.backward(); // update grad values top to bottom

    std.debug.print("Y = {d:.5}\n", .{Y.data});
    engine.GenerateGraph(&Y);
    std.debug.print("(Y2 - Y) / h = {d:.5}\n", .{(Y2 - Y.data) / h});
}
