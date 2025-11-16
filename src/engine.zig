const std = @import("std");

const Operation = enum { add, mul, tanh, nope };

pub fn FixedString(comptime N: usize) type {
    return struct {
        buf: [N]u8 = undefined,
        len_: usize = 0,

        pub fn init(text: []const u8) @This() {
            var s = @This(){};
            const n = @min(text.len, N);
            std.mem.copyForwards(u8, s.buf[0..n], text[0..n]);
            s.len_ = n;
            // if (n < N) std.mem.set(u8, s.buf[n..], 0);  // reset tail
            return s;
        }

        pub fn slice(self: *const @This()) []const u8 {
            return self.buf[0..self.len_];
        }
    };
}

pub const Label = FixedString(12);

// pub const FString = struct {
//     buf: [12]u8,
//     len_: usize = 0,
//
//     pub fn len(self: *const FString) usize {
//         return self.buf.len;
//     }
// };

pub fn value(data: f32, label: []const u8) Value {
    return Value{ .data = data, .label = Label.init(label) };
}

pub fn vec(allocator: std.mem.Allocator, array: []const f32) []Value {
    var v_list = allocator.alloc(Value, array.len) catch unreachable;
    var buf: [12]u8 = undefined;
    for (array, 0..) |x, i| {
        v_list[i] = value(x, std.fmt.bufPrint(&buf, "x{d}", .{i}) catch unreachable);
    }
    return v_list;
}

pub const Value = struct {
    // allocator: std.mem.Allocator,
    data: f32,
    grad: f32 = 0.0,
    label: Label = Label.init("undefined"), //FixedString, // [12]u8 = [_]u8{0} ** 12,

    _backward: ?*const fn (self: *const Value) void = null,

    prev: [2]?*Value = [_]?*Value{ null, null },

    op: Operation = .nope,

    pub fn print(self: Value) void {
        std.debug.print("Value(data={d})\n", .{self.data});
    }

    pub fn printL(self: Value) void {
        std.debug.print("Value({s}: data={d}, grad={d})\n", .{ self.label.slice(), self.data, self.grad });
    }

    pub fn printMore(self: Value) void {
        std.debug.print("Value({s}({d}): data={d}, op={s})\n", .{ self.label.buf, self.label.len_, self.data, @tagName(self.op) });
        if (self.op != .nope and self.prev[0] != null) {
            for (self.prev) |child| {
                if (child) |c| {
                    //std.debug.print(", child: {s}", .{c.label.slice()});
                    c.printMore();
                }
            }
        }
    }

    fn _backward_add(self: *const Value) void {
        self.prev[0].?.grad += 1.0 * self.grad;
        self.prev[1].?.grad += 1.0 * self.grad;
    }

    fn _backward_mul(self: *const Value) void {
        self.prev[0].?.grad += self.prev[1].?.data * self.grad;
        self.prev[1].?.grad += self.prev[0].?.data * self.grad;
    }

    fn _backward_tanh(self: *const Value) void {
        self.prev[0].?.grad += (1 - std.math.pow(f32, self.data, 2)) * self.grad;
    }

    fn _forward_op(self: *Value) void {
        // std.debug.print("label: {s}, op: {s}\n", .{ self.label.slice(), @tagName(self.op) });
        switch (self.op) {
            .add => {
                self.data = self.prev[0].?.data + self.prev[1].?.data;
            },
            .mul => {
                self.data = self.prev[0].?.data * self.prev[1].?.data;
            },
            else => {},
        }
    }

    pub fn add(self: *Value, other: *Value, label: []const u8) Value {
        return Value{
            .data = self.data + other.data,
            .prev = .{ self, other },
            .op = .add,
            .label = Label.init(label),
            ._backward = _backward_add,
        };
    }

    pub fn mul(self: *Value, other: *Value, label: []const u8) Value {
        // return if (self == other) error.IncorrectArguments else Value{
        return Value{
            .data = self.data * other.data,
            .prev = .{ self, other },
            .op = .mul,
            .label = Label.init(label),
            ._backward = _backward_mul,
        };
    }

    pub fn tanh(self: *Value) Value {
        return Value{
            .data = std.math.tanh(self.data),
            .prev = .{ self, null },
            .op = .tanh,
            .label = Label.init("tanh"),
            ._backward = _backward_tanh,
        };
    }

    pub fn backward(self: *Value) void {
        var topo = Topo.init(self);
        const sorted = topo.sorted();
        for (sorted) |s| {
            s.grad = 0;
        }
        self.grad = 1.0;

        for (0..sorted.len) |i| {
            const v = sorted[sorted.len - i - 1];
            //std.debug.print("{d}: {s}\n", .{ i, v.label.slice() });
            if (v._backward) |f| {
                f(v);
            }
        }
    }

    pub fn forward(self: *Value) void {
        var topo = Topo.init(self);
        const sorted = topo.sorted();
        for (0..sorted.len) |i| {
            const v = sorted[i];
            // std.debug.print("{d}: {s}\n", .{ i, v.label.slice() });
            v._forward_op();
        }
    }
};

fn printNode(v: *const Value) void {
    std.debug.print("\t{s} [label=\"{{{s} | data: {d} | grad: {d}}}\"];\n", .{ v.label.slice(), v.label.slice(), v.data, v.grad });

    for (v.prev) |child| {
        if (child) |c| {
            std.debug.print("\t{s} -> {s};\n", .{ c.label.slice(), v.label.slice() });
            printNode(c);
        }
    }
}

pub fn GenerateGraph(root: *const Value) void {
    std.debug.print("digraph G {s}\n", .{"{"});
    std.debug.print("\tnode [shape = record];\n", .{});

    printNode(root);
    std.debug.print("{s}\n", .{"}"});
}

const Topo = struct {
    values: [1024]*Value = undefined,
    count: u32 = 0,
    visited: [1024]*Value = undefined,
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
            //std.debug.print("visited: \t{s}\n", .{v.label.slice()});
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
        var t = Topo{};
        t.build(v);
        return t;
    }
};
