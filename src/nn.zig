const std = @import("std");
const Value = @import("engine.zig").Value;
const value = @import("engine.zig").value;

var rand_impl = std.Random.DefaultPrng.init(512);
var n_id: u8 = 1;
var l_id: u8 = 1;

pub const Neuron = struct {
    allocator: std.mem.Allocator,
    w: []Value = undefined,
    b: Value,
    oper_buf: []Value = undefined,
    id: u8 = 0,
    layer_id: u8 = 0,

    pub fn init(allocator: std.mem.Allocator, nin: u32, layer_id: u8) !Neuron {
        const oper_count = nin + (nin - 1) + 1; // sum(wi * xi) + b
        const oper_buf = try allocator.alloc(Value, oper_count);

        const id = n_id;
        n_id += 1;
        var buf: [8]u8 = undefined;
        const weights = try allocator.alloc(Value, nin);
        for (weights, 0..) |*w, i| {
            const label = try std.fmt.bufPrint(&buf, "L{d}N{d}_w{d}", .{ layer_id, id, i + 1 });
            const f32_value = rand_impl.random().float(f32);

            w.* = value(f32_value, buf[0..label.len]);
        }

        const b_label = try std.fmt.bufPrint(&buf, "L{d}N{d}_b", .{ layer_id, id });

        return Neuron{
            .id = id,
            .layer_id = layer_id,
            .w = weights,
            .b = value(0.2, b_label),
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
        self.b.printL();
    }

    pub fn call(self: *Neuron, inputs: []Value) !Value {
        if (inputs.len != self.w.len) return error.IncorrectInputCount;
        var i: u32 = 0;

        var buf: [12]u8 = undefined;
        for (0..inputs.len) |k| {
            const x = &inputs[k];
            const label = try std.fmt.bufPrint(&buf, "L{d}N{d}_xw{d}", .{ self.layer_id, self.id, k + 1 });
            self.oper_buf[i] = self.w[i].mul(x, label);
            i += 1;
        }

        const j = i;
        var acc = &self.oper_buf[0];
        for (1..j) |k| {
            const label = try std.fmt.bufPrint(&buf, "L{d}N{d}_xw_sum{d}", .{ self.layer_id, self.id, k });
            self.oper_buf[i] = acc.add(&self.oper_buf[k], label);
            acc = &self.oper_buf[i];
            i += 1;
        }

        const label = try std.fmt.bufPrint(&buf, "L{d}N{d}_b_sum", .{ self.layer_id, self.id });
        self.oper_buf[i] = self.oper_buf[i - 1].add(&self.b, label);

        return self.oper_buf[i].tanh(self.layer_id, self.id);
    }
};

pub const Layer = struct {
    id: u8 = 0,
    allocator: std.mem.Allocator,
    neurons: []Neuron = undefined,
    outs: []Value = undefined,

    pub fn init(allocator: std.mem.Allocator, nin: u8, nout: u8) !Layer {
        const id = l_id;
        l_id += 1;

        const neurons = try allocator.alloc(Neuron, nout);
        const outs = try allocator.alloc(Value, nout);
        for (0..nout) |i| {
            neurons[i] = try Neuron.init(allocator, nin, id);
            neurons[i].print();
        }

        return Layer{
            .id = l_id,
            .allocator = allocator,
            .neurons = neurons,
            .outs = outs,
        };
    }

    pub fn deinit(self: *const Layer) void {
        self.allocator.free(self.outs);
        for (self.neurons) |n| {
            n.deinit();
        }
        self.allocator.free(self.neurons);
    }

    pub fn call(self: *const Layer, inputs: []Value) ![]Value {
        for (0..self.neurons.len) |i| {
            self.outs[i] = try self.neurons[i].call(inputs);
        }
        return self.outs;
    }
};

pub const MLP = struct {
    allocator: std.mem.Allocator,
    layers: []Layer = undefined,

    pub fn init(allocator: std.mem.Allocator, nin: u8, nouts: []u8) !MLP {
        const layers = try allocator.alloc(Layer, nouts.len);
        layers[0] = try Layer.init(allocator, nin, nouts[0]);
        for (0..nouts.len - 1) |i| {
            layers[i + 1] = try Layer.init(allocator, nouts[i], nouts[i + 1]);
        }
        return MLP{
            .allocator = allocator,
            .layers = layers,
        };
    }

    pub fn deinit(self: *const MLP) void {
        for (self.layers) |layer| {
            layer.deinit();
        }
        self.allocator.free(self.layers);
    }

    pub fn call(self: *const MLP, inputs: []Value) ![]Value {
        var outs: []Value = inputs;
        for (self.layers) |layer| {
            outs = try layer.call(outs);
        }
        return self.layers[self.layers.len - 1].outs;
    }
};
