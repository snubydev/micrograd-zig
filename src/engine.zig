const std = @import("std");
const t = std.testing;
const fmt = std.fmt;
const Allocator = std.mem.Allocator;
const testHeader = @import("helpers.zig").testHeader;

const Operation = enum { add, sub, mul, tanh, leaf };

pub fn vec(a: Allocator, array: []const f32) []Value {
    var v_list = a.alloc(Value, array.len) catch unreachable;
    for (array, 0..) |x, i| {
        v_list[i] = Value.init(x, fmt.allocPrint(a, "x{d}", .{i + 1}) catch unreachable);
    }
    return v_list;
}

pub const Value = struct {
    data: f32,
    grad: f32 = 0.0,
    label: []u8,
    prev: []*Value,
    op: Operation = .leaf,

    pub fn init(a: Allocator, data: f32, label: []const u8) !Value {
        return Value{ .data = data, .op = .leaf, .label = try a.dupe(u8, label), .prev = &.{} };
    }

    pub fn deinit(self: *Value, a: Allocator) void {
        a.free(self.label);
        a.free(self.prev);
    }

    pub fn print(self: *Value) void {
        std.debug.print("type=Value data={d:.4} grad={d} label={s} op={s})\n", .{ self.data, self.grad, self.label, @tagName(self.op) });
    }

    pub fn printGraph(self: *Value) void {
        self.print();
        for (self.prev) |child| {
            child.print();
        }
    }
};

//     fn _backward_add(self: *const Value) void {
//         self.prev[0].?.grad += 1.0 * self.grad;
//         self.prev[1].?.grad += 1.0 * self.grad;
//     }
//
//     fn _backward_sub(self: *const Value) void {
//         self.prev[0].?.grad += 1.0 * self.grad;
//         self.prev[1].?.grad -= 1.0 * self.grad;
//     }
//
//     fn _backward_mul(self: *const Value) void {
//         self.prev[0].?.grad += self.prev[1].?.data * self.grad;
//         self.prev[1].?.grad += self.prev[0].?.data * self.grad;
//     }
//
//     fn _backward_tanh(self: *const Value) void {
//         self.prev[0].?.grad += (1 - std.math.pow(f32, self.data, 2)) * self.grad;
//     }
//
//     fn _forward_op(self: *Value) void {
//         // std.debug.print("label: {s}, op: {s}\n", .{ self.label.slice(), @tagName(self.op) });
//         switch (self.op) {
//             .add => {
//                 self.data = self.prev[0].?.data + self.prev[1].?.data;
//             },
//             .mul => {
//                 self.data = self.prev[0].?.data * self.prev[1].?.data;
//             },
//             else => {},
//         }
//     }
//
//     pub fn add(self: *Value, other: *Value, label: []const u8) Value {
//         return Value{
//             .data = self.data + other.data,
//             .prev = .{ self, other },
//             .op = .add,
//             .label = Label.init(label),
//             ._backward = _backward_add,
//         };
//     }
//
//     pub fn sub(self: *Value, other: *Value, label: []const u8) Value {
//         return Value{
//             .data = self.data - other.data,
//             .prev = .{ self, other },
//             .op = .sub,
//             .label = Label.init(label),
//             ._backward = _backward_sub,
//         };
//     }
//
//     pub fn mul(self: *Value, other: *Value, label: []const u8) Value {
//         // return if (self == other) error.IncorrectArguments else Value{
//         return Value{
//             .data = self.data * other.data,
//             .prev = .{ self, other },
//             .op = .mul,
//             .label = Label.init(label),
//             ._backward = _backward_mul,
//         };
//     }
//
//     pub fn tanh(self: *Value, layer_id: u8, neuron_id: u8) Value {
//         var buf: [12]u8 = undefined;
//         const label = std.fmt.bufPrint(&buf, "L{d}N{d}_tanh", .{ layer_id, neuron_id }) catch "n__tanh";
//
//         return Value{
//             .data = std.math.tanh(self.data),
//             .prev = .{ self, null },
//             .op = .tanh,
//             .label = Label.init(label),
//             ._backward = _backward_tanh,
//         };
//     }
//
//     pub fn backward(self: *Value) void {
//         var topo = Topo.init(self);
//         const sorted = topo.sorted();
//         for (sorted) |s| {
//             s.grad = 0;
//         }
//         self.grad = 1.0;
//
//         for (0..sorted.len) |i| {
//             const v = sorted[sorted.len - i - 1];
//             //std.debug.print("{d}: {s}\n", .{ i, v.label.slice() });
//             if (v._backward) |f| {
//                 f(v);
//             }
//         }
//     }
//
//     pub fn forward(self: *Value) void {
//         var topo = Topo.init(self);
//         const sorted = topo.sorted();
//         for (0..sorted.len) |i| {
//             const v = sorted[i];
//             // std.debug.print("{d}: {s}\n", .{ i, v.label.slice() });
//             v._forward_op();
//         }
//     }
// };

fn printNode(v: *const Value) void {
    std.debug.print("\t{s} [label=\"{{{s} | data: {d} | grad: {d}}}\"];\n", .{ v.label.slice(), v.label.slice(), v.data, v.grad });
    for (v.prev) |child| {
        if (child) |c| {
            std.debug.print("\t{s} -> {s};\n", .{ c.label.slice(), v.label.slice() });
            //printNode(c);
        }
    }
}

pub fn GenerateGraph(root: *Value) void {
    std.debug.print("digraph G {s}\n", .{"{"});
    std.debug.print("\tnode [shape = record];\n", .{});
    var topo = Topo.init(root);
    const sorted = topo.sorted();
    for (sorted) |v| {
        printNode(v);
    }
    std.debug.print("{s}\n", .{"}"});
}

const Topo = struct {
    values: [4096]*Value = undefined,
    count: u32 = 0,
    visited: [4096]*Value = undefined,
    visited_count: u32 = 0,

    pub fn isVisited(self: *Topo, v: *Value) bool {
        var already_visited = false;

        for (self.visited[0..self.visited_count]) |s| {
            if (v == s) {
                already_visited = true;
                break;
            }
        }

        if (!already_visited) {
            self.visited[self.visited_count] = v;
            self.visited_count += 1;
        }
        return already_visited;
    }

    pub fn build(self: *Topo, v: *Value) void {
        if (!isVisited(self, v)) {
            for (v.prev) |child| {
                if (child) |c| {
                    build(self, c);
                }
            }
            self.values[self.count] = v;
            self.count += 1;
        }
    }

    pub fn sorted(self: *Topo) []*Value {
        return self.values[0..self.count];
    }

    pub fn init(v: *Value) Topo {
        var topo = Topo{};
        topo.build(v);
        return topo;
    }
};

test "value" {
    testHeader(@src());
    const a = t.allocator;
    var x1 = try Value.init(a, 2.0, "x1");
    var x2 = try Value.init(a, 3.0, "x2");

    defer {
        x1.deinit(a);
        x2.deinit(a);
    }

    try t.expectEqual(2.0, x1.data);
    try t.expectEqual(3.0, x2.data);

    x1.print();
    x2.print();
}
