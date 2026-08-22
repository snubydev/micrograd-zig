const std = @import("std");
const Allocator = std.mem.Allocator;
const assert = std.debug.assert;
const fmt = std.fmt;
const t = std.testing;
const testHeader = @import("helpers.zig").testHeader;
const Value = @import("engine.zig").Value;
const value = @import("engine.zig").value;

var rand_impl = std.Random.DefaultPrng.init(512);
var n_id: u64 = 1;
var l_id: u64 = 1;

pub const Neuron = struct {
    weights: []Value,
    bias: Value,
    id: u64,

    pub fn init(a: Allocator, nin: u32) !Neuron {
        const id = getNeuronId();
        const weights = try a.alloc(Value, nin);

        var buf: [64]u8 = undefined;

        for (weights, 0..) |*w, i| {
            const w_label = try fmt.bufPrint(&buf, "n{d}_w{d}", .{ id, i + 1 });
            const f32_value = rand_impl.random().float(f32);
            w.* = try Value.init(a, f32_value, w_label);
        }

        const b_label = try fmt.bufPrint(&buf, "n{d}_b", .{id});

        return Neuron{
            .weights = weights,
            .bias = try Value.init(a, 0.2, b_label),
            .id = id,
        };
    }

    pub fn deinit(self: *Neuron, a: Allocator) void {
        for (self.weights) |*w| {
            w.deinit(a);
        }
        a.free(self.weights);
        self.bias.deinit(a);
    }

    pub fn print(self: *Neuron) void {
        std.debug.print("type=Neuron id={d}\n", .{self.id});
        for (self.weights) |w| {
            w.print();
        }
        self.bias.print();
    }
};

pub const NeuronActivation = struct {
    x: []Value,
    prod: []Value,
    sum: []Value,
    out: Value,
    topo: []*Value,
    id: u64,

    pub fn init(a: Allocator, nin: u32, neuron: *Neuron, inputs: []Value) !NeuronActivation {
        var self: NeuronActivation = undefined;
        self.id = neuron.id;

        assert(inputs.len == nin);
        self.x = inputs;
        // } else {
        //     self.x = try a.alloc(Value, nin);
        //     for (self.x, 0..) |*x, i| {
        //         const label = try fmt.allocPrint(a, "n{d}_x{d}", .{ neuron.id, i });
        //         x.* = .{ .data = 0, .grad = 0, .op = .leaf, .prev = &.{}, .label = label };
        //     }
        // }

        var buf: [64]u8 = undefined;

        self.prod = try a.alloc(Value, nin);
        for (self.prod, neuron.weights, self.x, 0..) |*p, *w, *x, j| {
            const label = try fmt.bufPrint(&buf, "n{d}_w{d}x{d}", .{ neuron.id, j + 1, j + 1 });
            p.* = .{
                .data = w.data * x.data,
                .grad = 0,
                .op = .mul,
                .prev = try a.dupe(*Value, &.{ w, x }),
                .label = try a.dupe(u8, label),
            };
        }

        self.sum = try a.alloc(Value, nin); // sum ([xi*wi, i=0..nin-1]) + b
        const b_label = try fmt.bufPrint(&buf, "n{d}_sum_b", .{neuron.id});
        self.sum[0] = .{
            .data = neuron.bias.data + self.prod[0].data,
            .grad = 0,
            .op = .add,
            .prev = try a.dupe(*Value, &.{ &neuron.bias, &self.prod[0] }),
            .label = try a.dupe(u8, b_label),
        };

        for (self.sum[1..], self.prod[1..], 0..) |*s, *p, i| {
            const label = try fmt.bufPrint(&buf, "n{d}_sum{d}", .{ neuron.id, i + 1 });
            s.* = .{
                .data = self.sum[i].data + p.data,
                .grad = 0,
                .op = .add,
                .prev = try a.dupe(*Value, &.{ &self.sum[i], p }),
                .label = try a.dupe(u8, label),
            };
        }

        const out_label = try fmt.bufPrint(&buf, "n{d}_out", .{neuron.id});
        self.out = .{
            .data = 0,
            .grad = 0,
            .op = .tanh,
            .prev = try a.dupe(*Value, &.{&self.sum[self.sum.len - 1]}),
            .label = try a.dupe(u8, out_label),
        };
        return self;
    }

    pub fn deinit(self: *NeuronActivation, a: Allocator) void {
        self.out.deinit(a);
        for (self.sum) |*s| {
            s.deinit(a);
        }
        a.free(self.sum);
        for (self.prod) |*p| {
            p.deinit(a);
        }
        a.free(self.prod);
        for (self.x) |*x| {
            x.deinit(a);
        }
    }

    pub fn print(self: *NeuronActivation) void {
        std.debug.print("type=NeuronActivation id={d}\n", .{self.id});
        for (self.x) |x| {
            x.print();
        }
        for (self.prod) |p| {
            p.print();
        }
        for (self.sum) |s| {
            s.print();
        }
        self.out.print();
    }
};

// pub const Neuron = struct {
//     allocator: std.mem.Allocator,
//     w: []Value = undefined,
//     b: Value,
//     oper_buf: []Value = undefined,
//     id: u8 = 0,
//     layer_id: u8 = 0,
//
//     pub fn init(allocator: std.mem.Allocator, nin: u32, layer_id: u8) !Neuron {
//         const oper_count = nin + (nin - 1) + 1; // sum(wi * xi) + b
//         const oper_buf = try allocator.alloc(Value, oper_count);
//
//         const id = n_id;
//         n_id += 1;
//         var buf: [8]u8 = undefined;
//         const weights = try allocator.alloc(Value, nin);
//         for (weights, 0..) |*w, i| {
//             const label = try std.fmt.bufPrint(&buf, "L{d}N{d}_w{d}", .{ layer_id, id, i + 1 });
//             const f32_value = rand_impl.random().float(f32);
//
//             w.* = value(f32_value, buf[0..label.len]);
//         }
//
//         const b_label = try std.fmt.bufPrint(&buf, "L{d}N{d}_b", .{ layer_id, id });
//
//         return Neuron{
//             .id = id,
//             .layer_id = layer_id,
//             .w = weights,
//             .b = value(0.2, b_label),
//             .oper_buf = oper_buf,
//             .allocator = allocator,
//         };
//     }
//
//     pub fn deinit(self: *const Neuron) void {
//         self.allocator.free(self.w);
//         if (self.oper_buf.len > 0) {
//             self.allocator.free(self.oper_buf);
//         }
//     }
//
//     pub fn print(self: *const Neuron) void {
//         for (self.w) |w| {
//             w.printL();
//         }
//         self.b.printL();
//     }
//
//     // call - bulds a linked structure of Values for tanh( sum(xi*wi[i=0..n]) + b )
//     //      - calculate data values for created operation nodes, grad=0
//     // returns head Value as result of last (tanh) operation
//     pub fn call(self: *Neuron, inputs: []Value) !Value {
//         if (inputs.len != self.w.len) return error.IncorrectInputCount;
//         var i: u32 = 0;
//
//         const buf: [12]u8 = undefined;
//         for (inputs, 0..) |x, k| {
//             self.oper_buf[i] = self.w[i].mul(x, self._label(buf, "xw", k));
//             i += 1;
//         }
//
//         const j = i;
//         var acc = &self.oper_buf[0];
//         for (1..j) |k| {
//             self.oper_buf[i] = acc.add(&self.oper_buf[k], self._label(buf, "xw_sum", k));
//             acc = &self.oper_buf[i];
//             i += 1;
//         }
//
//         self.oper_buf[i] = self.oper_buf[i - 1].add(&self.b, self._label(buf, "b_sum", 0)); // SUM(xi*wi[i=0..n]) + b
//
//         return self.oper_buf[i].tanh(self.layer_id, self.id);
//     }
//
//     fn _label(self: *Neuron, buf: [12]u8, tag: []const u8, k: usize) std.fmt.BufPrintError![]u8 {
//         return try std.fmt.bufPrint(&buf, "L{d}N{d}_{s}{d}", .{ self.layer_id, self.id, tag, k + 1 });
//     }
// };

fn getNeuronId() u64 {
    const id = n_id;
    n_id += 1;
    return id;
}

fn getLayerId() u64 {
    const id = l_id;
    l_id += 1;
    return id;
}

// pub const Layer = struct {
//     id: u8,
//     allocator: std.mem.Allocator,
//     neurons: []Neuron,
//     outs: []Value,
//
//     pub fn init(allocator: std.mem.Allocator, nin: u8, nout: u8) !Layer {
//         const id = getLayerId();
//         const neurons = try allocator.alloc(Neuron, nout);
//         const outs = try allocator.alloc(Value, nout);
//         for (0..nout) |i| {
//             neurons[i] = try Neuron.init(allocator, nin, id);
//             //std.debug.print("Layer.init: ", .{});
//             //neurons[i].print();
//         }
//         return Layer{
//             .id = l_id,
//             .allocator = allocator,
//             .neurons = neurons,
//             .outs = outs,
//         };
//     }
//
//     pub fn deinit(self: *const Layer) void {
//         self.allocator.free(self.outs);
//         for (self.neurons) |n| {
//             n.deinit();
//         }
//         self.allocator.free(self.neurons);
//     }
//
//     // call - builds neurons structures, calculates output activation value (result) and stores into self.outs
//     pub fn call(self: *const Layer, inputs: []Value) ![]Value {
//         for (self.neurons, 0..) |neuron, i| {
//             self.outs[i] = try neuron.call(inputs);
//         }
//         return self.outs;
//     }
// };
//
// pub const MLP = struct {
//     allocator: std.mem.Allocator,
//     layers: []Layer = undefined,
//
//     pub fn init(allocator: std.mem.Allocator, nin: u8, nouts: []u8) !MLP {
//         const layers = try allocator.alloc(Layer, nouts.len);
//         layers[0] = try Layer.init(allocator, nin, nouts[0]);
//         for (1..nouts.len) |i| {
//             layers[i] = try Layer.init(allocator, nouts[i - 1], nouts[i]);
//         }
//         return MLP{
//             .allocator = allocator,
//             .layers = layers,
//         };
//     }
//
//     pub fn deinit(self: *const MLP) void {
//         for (self.layers) |layer| {
//             layer.deinit();
//         }
//         self.allocator.free(self.layers);
//     }
//
//     // call - builds neurons structures, calculates output activation value (result) and returns as outs
//     pub fn call(self: *const MLP, inputs: []Value) ![]Value {
//         var outs: []Value = inputs;
//         for (self.layers) |layer| {
//             outs = try layer.call(outs);
//         }
//         return outs;
//     }
//
//     pub fn adjust(self: *const MLP, h: f32) void {
//         for (self.layers) |layer| {
//             for (layer.neurons) |neuron| {
//                 neuron.b.data += h * neuron.b.grad;
//                 for (neuron.w) |w| {
//                     w.data += h * w.data;
//                 }
//             }
//         }
//     }
// };

test "neuron_init" {
    testHeader(@src());
    const a = t.allocator;
    var n1 = try Neuron.init(a, 4);
    var n2 = try Neuron.init(a, 3);

    defer {
        n1.deinit(a);
        n2.deinit(a);
    }

    try t.expectEqual(0.2, n1.bias.data);
    try t.expectEqual(4, n1.weights.len);
    try t.expectEqual(3, n2.weights.len);

    n1.print();
    n2.print();
}

test "neuron_activation_init" {
    testHeader(@src());
    const a = t.allocator;
    var n1 = try Neuron.init(a, 4);
    var x = [_]Value{
        try Value.init(a, 1.1, "x1"),
        try Value.init(a, 2.2, "x2"),
        try Value.init(a, 3.3, "x3"),
        try Value.init(a, 4.4, "x4"),
    };
    var na1 = try NeuronActivation.init(a, 4, &n1, &x);
    n1.print();
    na1.print();

    try t.expectEqual(n1.weights[0].data * x[0].data, na1.prod[0].data);
    try t.expectEqual(4, n1.weights.len);
    try t.expectEqual(4, na1.x.len);
    try t.expectEqual(4, na1.prod.len);
    try t.expectEqual(4, na1.sum.len);

    defer {
        na1.deinit(a);
        n1.deinit(a);
    }
}
