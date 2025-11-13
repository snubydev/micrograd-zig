const std = @import("std");

const engine = @import("engine.zig");
const value = @import("engine.zig").value;

// const micrograd_zig = @import("micrograd_zig");

pub fn main() !void {
    // const allocator = std.heap.page_allocator;

    // works for cases:
    //   yk = xi + xj
    //   yk = xi + xi
    //   yk = xi * xj
    //
    // does't work for cases:
    //   yj = xi * xi - squre relation

    var x1 = value(2, "x1");
    var x2 = value(-4, "x2");

    var y1 = try x1.mul(&x2, "y1");
    var y2 = x2.add(&x2, "y2");

    var y3 = y1.add(&y2, "y3");

    y3.backward();

    engine.GenerateGraph(&y3);
}
