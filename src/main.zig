const std = @import("std");

const engine = @import("engine.zig");
const value = @import("engine.zig").value;
const Value = @import("engine.zig").Value;
const Neuron = @import("nn.zig").Neuron;
const Layer = @import("nn.zig").Layer;

// const micrograd_zig = @import("micrograd_zig");

fn makeNeurone(x1: f32, x2: f32, w1: f32, w2: f32, b: f32) void {
    var _x1 = value(x1, "x1");
    var _x2 = value(x2, "x2");
    var _w1 = value(w1, "w1");
    var _w2 = value(w2, "w2");
    var _b = value(b, "b");
    var _x1_w1 = _x1.mul(&_w1, "x1w1");
    var _x2_w2 = _x2.mul(&_w2, "x2w2");
    var _xw = _x1_w1.add(&_x2_w2, "xw");
    var n = _xw.add(&_b, "n");
    var o = n.tanh();
    o.printMore();
    o.backward();
    engine.GenerateGraph(&o);
}

pub fn main() !void {
    // const allocator = std.heap.page_allocator;

    // makeNeurone(2.0, 0.0, -3.0, 1.0, 6.88137358701954);

    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer std.debug.assert(gpa.deinit() == .ok); // assert no leaks
    const allocator = gpa.allocator();

    //const allocator = std.heap.page_allocator;

    const nin = 4;
    var x_array = [_][nin]f32{
        .{ 2.0, 3.0, -1.0, 1.0 },
        .{ 2.2, 0.5, 1.0, -5.0 },
        .{ -1.7, -1.1, -2.3, 0.4 },
        .{ 0.0, 1.0, 0.0, 0.1 },
    };

    var n1 = try Neuron.init(allocator, nin);
    defer n1.deinit();
    //n1.print();

    var xs = try allocator.alloc([]Value, x_array.len);
    defer {
        for (xs) |x| {
            allocator.free(x);
        }
        allocator.free(xs);
    }

    for (0..x_array.len) |i| {
        xs[i] = engine.vec(allocator, x_array[i][0..]);
    }

    for (xs, 0..) |xs_i, i| {
        std.debug.print("xs[{d}]=[", .{i});
        for (xs_i, 0..) |x, j| {
            if (x.data >= 0) std.debug.print(" ", .{});
            std.debug.print("{d:.1}", .{x.data});
            if (j < xs_i.len - 1) std.debug.print(",\t", .{});
        }
        std.debug.print("]\n", .{});
    }

    for (xs) |x| {
        const out = try n1.call(x);
        out.print();
    }

    var l1 = try Layer.init(allocator, 4, 4);
    defer l1.deinit();

    for (xs, 0..) |x, i| {
        const outs = try l1.call(x);
        std.debug.print("layer outputs [{d}] ------\n", .{i + 1});
        for (outs) |o| {
            o.print();
        }
    }

    //const out = try n1.call(xs[0]);
    //out.print();
    // engine.GenerateGraph(&out);
}

// zig test --summary all

test "value_op" {
    var x1 = value(2.0, "x1");
    var x2 = value(3.0, "x2");
    var y1 = x1.add(&x2, "y1");
    var y2 = x2.mul(&x2, "y2");
    var Y = y1.add(&y2, "y3");
    Y.backward();

    try std.testing.expectEqual(14.0, Y.data);

    // std.debug.print("Y1 = {d:.5}\n", .{Y.data});
    // engine.GenerateGraph(&Y);

    const h: f32 = 0.0001;
    const Y1 = Y.data;
    x2.data = x2.data + h;

    Y.forward(); // update data vaules bottom to top
    Y.backward(); // update grad values top to bottom

    // std.debug.print("Y2 = {d:.5}\n", .{Y.data});
    // engine.GenerateGraph(&Y);
    // std.debug.print("(Y2 - Y1) / h = {d:.5}\n", .{(Y.data - Y1) / h});

    const dYdx = (Y.data - Y1) / h;
    const E = 0.001;
    const margin = @abs(dYdx - x2.grad);
    // std.debug.print("dYdx: {d}, x2.grad: {d}\n", .{ dYdx, x2.grad });
    // std.debug.print("margin: {d}\n", .{margin});
    try std.testing.expect(margin < E);
}
