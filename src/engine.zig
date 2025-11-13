const std = @import("std");

const Operation = enum { add, mul, nope };

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

pub const Value = struct {
    // allocator: std.mem.Allocator,
    data: f32,
    grad: f32 = 0.0,
    label: Label = Label.init("undefined"), //FixedString, // [12]u8 = [_]u8{0} ** 12,

    backward: ?*fn () void = null,

    prev: [2]?*Value = [_]?*Value{ null, null },

    op: Operation = .nope,

    pub fn print(self: Value) void {
        std.debug.print("Value(data={d})\n", .{self.data});
    }

    pub fn printL(self: Value) void {
        std.debug.print("Value({s}: data={d})\n", .{ self.label.slice(), self.data });
    }

    pub fn printMore(self: Value) void {
        std.debug.print("Value({s}: data={d}", .{ self.label.slice(), self.data });
        if (self.op != .nope and self.prev[0] != null) {
            std.debug.print(", {s}", .{@tagName(self.op)});
            for (self.prev) |child| {
                std.debug.print(", {s}", .{child.?.label.slice()});
            }
        }
        std.debug.print(")\n", .{});
    }

    pub fn add(self: *Value, other: *Value, label: []const u8) Value {
        return Value{
            .data = self.data + other.data,
            .prev = .{ self, other },
            .op = .add,
            .label = Label.init(label),
        };
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
