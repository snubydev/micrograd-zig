const std = @import("std");
const Value = @import("engine.zig").Value;
const value = @import("engine.zig").value;

var rand_impl = std.Random.DefaultPrng.init(512);

pub const Neuron = struct {
    allocator: std.mem.Allocator,
    w: []Value = undefined,
    b: Value = value(0.0, "b"),
    oper_buf: []Value = undefined,

    pub fn init(allocator: std.mem.Allocator, nin: u32) !Neuron {
        const oper_count = nin + (nin - 1) + 1 + 1; // sum(wi * xi) + b + out
        const oper_buf = try allocator.alloc(Value, oper_count);

        var buf: [8]u8 = undefined;
        const weights = try allocator.alloc(Value, nin);
        for (weights, 0..) |*w, i| {
            const label = try std.fmt.bufPrint(&buf, "w{d}", .{i + 1});

            const f32_value = rand_impl.random().float(f32);

            w.* = value(f32_value, buf[0..label.len]);
        }

        return Neuron{
            .w = weights,
            .oper_buf = oper_buf,
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
            w.printL();
        }
    }

    pub fn out(self: *const Neuron) *Value {
        return &self.oper_buf[self.oper_buf.len - 1];
    }

    pub fn call(self: *Neuron, inputs: []Value) !*Value {
        if (inputs.len != self.w.len) return error.IncorrectInputCount;
        var i: u32 = 0;

        var buf: [12]u8 = undefined;
        for (0..inputs.len) |k| {
            const x = &inputs[k];
            const label = try std.fmt.bufPrint(&buf, "xw{d}", .{k + 1});
            self.oper_buf[i] = self.w[i].mul(x, label);
            i += 1;
        }

        const j = i;
        var acc = &self.oper_buf[0];
        for (1..j) |k| {
            const label = try std.fmt.bufPrint(&buf, "xw_sum{d}", .{k});
            self.oper_buf[i] = acc.add(&self.oper_buf[k], label);
            acc = &self.oper_buf[i];
            i += 1;
        }

        self.oper_buf[i] = self.oper_buf[i - 1].add(&self.b, "b_sum");

        i += 1;
        self.oper_buf[i] = self.oper_buf[i - 1].tanh();
        return self.out();
    }
};

pub const Layer = struct {
    allocator: std.mem.Allocator,
    neurons: []Neuron = undefined,
    outs: []*Value = undefined,

    pub fn init(allocator: std.mem.Allocator, nin: u8, nout: u8) !Layer {
        const neurons = try allocator.alloc(Neuron, nout);
        const outs = try allocator.alloc(*Value, nout);
        for (0..nout) |i| {
            neurons[i] = try Neuron.init(allocator, nin);
            outs[i] = neurons[i].out();
        }

        return Layer{
            .allocator = allocator,
            .neurons = neurons,
            .outs = outs,
        };
    }

    pub fn deinit(self: *Layer) void {
        self.allocator.free(self.outs);
        for (self.neurons) |n| {
            n.deinit();
        }
        self.allocator.free(self.neurons);
    }

    pub fn call(self: *Layer, inputs: []Value) ![]*Value {
        for (0..self.neurons.len) |i| {
            _ = try self.neurons[i].call(inputs);
        }
        return self.outs;
    }
};
