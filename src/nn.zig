const std = @import("std");
const Value = @import("engine.zig").Value;
const value = @import("engine.zig").value;

pub const Neuron = struct {
    allocator: std.mem.Allocator,
    w: []Value = undefined,
    b: Value = value(0.0, "b"),
    oper_buf: []Value = undefined,

    pub fn init(allocator: std.mem.Allocator, nin: u32) !Neuron {
        var buf: [8]u8 = undefined;
        const weights = try allocator.alloc(Value, nin);
        for (weights, 0..) |*w, i| {
            const label = try std.fmt.bufPrint(&buf, "w{d}", .{i + 1});
            w.* = value(1.0, buf[0..label.len]);
        }

        return Neuron{
            .w = weights,
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *const Neuron) void {
        self.allocator.free(self.w);
        if (self.oper_buf.len > 0) {
            self.allocator.free(self.oper_buf);
        }
    }

    pub fn print(self: *const Neuron) void {
        for (self.w) |w| {
            //std.debug.print("weight {s} data: {d}, grad: {d}\n", .{ w.label.slice(), w.data, w.grad });
            w.printL();
        }
    }

    pub fn call(self: *Neuron, inputs: []Value) !Value {
        if (inputs.len != self.w.len) return error.IncorrectInputCount;
        const oper_count = inputs.len + (inputs.len - 1) + 1; // sum(wi * xi) + b

        // std.debug.print("oper_count {d}\n", .{oper_count});
        self.oper_buf = try self.allocator.alloc(Value, oper_count);

        var i: u32 = 0;

        var buf: [12]u8 = undefined;
        for (0..inputs.len) |k| {
            // std.debug.print("k={d}, xw\n", .{k});
            const x = &inputs[k];
            const label = try std.fmt.bufPrint(&buf, "xw{d}", .{k + 1});
            self.oper_buf[i] = self.w[i].mul(x, label);
            i += 1;
        }

        const j = i;
        var acc = &self.oper_buf[0];
        for (1..j) |k| {
            // std.debug.print("from k={d}, into i={d} xw_sum\n", .{ k, i });
            const label = try std.fmt.bufPrint(&buf, "xw_sum{d}", .{k});
            self.oper_buf[i] = acc.add(&self.oper_buf[k], label);
            acc = &self.oper_buf[i];
            i += 1;
        }

        std.debug.print("i={d}, b_sum\n", .{i});
        self.oper_buf[i] = self.oper_buf[i - 1].add(&self.b, "b_sum");

        return self.oper_buf[i].tanh();
    }
};
