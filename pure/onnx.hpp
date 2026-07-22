// Minimal, self-contained ONNX I/O for the pure engine — a hand-rolled protobuf
// codec (no external deps) plus a small in-memory graph IR. Supports the op subset
// yolov8 needs at opset 13: Conv, Sigmoid, Mul, MaxPool, Resize, Concat, Add, Slice.
// This header holds the IR + writer + reader; the interpreter is in onnx_run.hpp.
#pragma once
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>
#include <fstream>
#include <cstdio>

namespace onx {

// ---- in-memory graph IR ----
struct Tensor64 { std::string name; std::vector<int64_t> dims; std::vector<float> data; };  // float initializer
struct IntsTensor { std::string name; std::vector<int64_t> dims; std::vector<int64_t> data; }; // int64 initializer
enum AType { A_FLOAT = 1, A_INT = 2, A_STRING = 3, A_FLOATS = 6, A_INTS = 7 };
struct Attr { std::string name; int type; int64_t i = 0; float f = 0; std::string s; std::vector<int64_t> ints; std::vector<float> floats; };
struct Node { std::string op_type, name; std::vector<std::string> input, output; std::vector<Attr> attr; };
struct ValueInfo { std::string name; std::vector<int64_t> dims; };  // dim -1 = dynamic
struct Graph {
  std::vector<Node> nodes;
  std::vector<Tensor64> init_f;      // float initializers (weights)
  std::vector<IntsTensor> init_i;    // int64 initializers (slice starts/ends/axes)
  std::vector<ValueInfo> inputs, outputs;
  int opset = 13;
};

// ============================ protobuf writer ============================
struct PB {
  std::string b;
  void varint(uint64_t v) { while (v >= 0x80) { b.push_back((char)(v | 0x80)); v >>= 7; } b.push_back((char)v); }
  void key(int f, int wt) { varint(((uint64_t)f << 3) | wt); }
  void vint(int f, uint64_t v) { key(f, 0); varint(v); }
  void f32(int f, float x) { key(f, 5); uint32_t u; std::memcpy(&u, &x, 4); for (int i = 0; i < 4; ++i) b.push_back((char)((u >> (8 * i)) & 0xff)); }
  void bytes(int f, const std::string& s) { key(f, 2); varint(s.size()); b += s; }
  void msg(int f, const PB& m) { bytes(f, m.b); }
};

inline std::string raw_of(const std::vector<float>& v) { return std::string((const char*)v.data(), v.size() * 4); }

inline PB build_ftensor(const Tensor64& t) {
  PB p;
  for (int64_t d : t.dims) p.vint(1, (uint64_t)d);   // dims
  p.vint(2, 1);                                       // data_type FLOAT
  p.bytes(8, t.name);                                 // name
  p.bytes(9, raw_of(t.data));                         // raw_data
  return p;
}
inline PB build_itensor(const IntsTensor& t) {
  PB p;
  for (int64_t d : t.dims) p.vint(1, (uint64_t)d);
  p.vint(2, 7);                                       // data_type INT64
  p.bytes(8, t.name);
  std::string raw((const char*)t.data.data(), t.data.size() * 8);
  p.bytes(9, raw);
  return p;
}
inline PB build_attr(const Attr& a) {
  PB p; p.bytes(1, a.name); p.vint(20, a.type);
  switch (a.type) {
    case A_FLOAT:  p.f32(2, a.f); break;
    case A_INT:    p.vint(3, (uint64_t)a.i); break;
    case A_STRING: p.bytes(4, a.s); break;
    case A_INTS:   for (int64_t v : a.ints) p.vint(8, (uint64_t)v); break;
    case A_FLOATS: for (float v : a.floats) p.f32(7, v); break;
  }
  return p;
}
inline PB build_node(const Node& n) {
  PB p;
  for (auto& s : n.input) p.bytes(1, s);
  for (auto& s : n.output) p.bytes(2, s);
  p.bytes(3, n.name); p.bytes(4, n.op_type);
  for (auto& a : n.attr) p.msg(5, build_attr(a));
  return p;
}
inline PB build_valueinfo(const ValueInfo& v) {
  PB shape; for (int64_t d : v.dims) { PB dd; if (d >= 0) dd.vint(1, (uint64_t)d); else dd.bytes(2, "N"); shape.msg(1, dd); }
  PB tt; tt.vint(1, 1); tt.msg(2, shape);            // Tensor: elem_type FLOAT, shape
  PB tp; tp.msg(1, tt);                              // TypeProto.tensor_type
  PB vi; vi.bytes(1, v.name); vi.msg(2, tp);
  return vi;
}

inline void save_onnx(const Graph& g, const std::string& path) {
  PB gp;
  for (auto& n : g.nodes) gp.msg(1, build_node(n));
  gp.bytes(2, "yolov8_cpp");
  for (auto& t : g.init_f) gp.msg(5, build_ftensor(t));
  for (auto& t : g.init_i) gp.msg(5, build_itensor(t));
  for (auto& v : g.inputs) gp.msg(11, build_valueinfo(v));
  for (auto& v : g.outputs) gp.msg(12, build_valueinfo(v));

  PB mp;
  mp.vint(1, 7);                                     // ir_version 7
  mp.bytes(2, "yolov8_cpp");                         // producer_name
  mp.msg(7, gp);                                     // graph
  PB ops; ops.bytes(1, ""); ops.vint(2, (uint64_t)g.opset); mp.msg(8, ops);  // opset_import

  std::ofstream f(path, std::ios::binary); f.write(mp.b.data(), mp.b.size());
}

}  // namespace onx
