const std = @import("std");

const engine = @import("engine.zig");
const value = @import("engine.zig").value;
const Value = @import("engine.zig").Value;
const Neuron = @import("nn.zig").Neuron;

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

    const allocator = std.heap.page_allocator;
    var n1 = try Neuron.init(allocator, 4);
    defer n1.deinit();

    n1.print();

    var x = [_]Value{
        value(2.0, "x1"),
        value(3.0, "x2"),
        value(-1.0, "x3"),
        value(1.0, "x4"),
    };

    const out = try n1.call(&x);

    // out.printMore();
    engine.GenerateGraph(&out);
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
