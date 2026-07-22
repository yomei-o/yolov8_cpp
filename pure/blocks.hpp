// yolov8 building blocks on the pure autograd engine (fused Conv, C2f, SPPF).
#pragma once
#include "autograd.hpp"
#include <fstream>

struct Conv { Tensor w, b; };   // BN already folded in

inline std::vector<float> read_bin(const std::string& p) {
  std::ifstream f(p, std::ios::binary | std::ios::ate);
  if (!f) { printf("cannot open %s\n", p.c_str()); std::exit(1); }
  auto n = f.tellg(); f.seekg(0);
  std::vector<float> v(n / sizeof(float));
  f.read(reinterpret_cast<char*>(v.data()), n);
  return v;
}

// fused Conv: stride given, padding = k/2 ('same' for odd k as yolov8 uses).
inline Tensor conv_fused(const Tensor& x, const Conv& c, int64_t stride = 1) {
  int64_t k = c.w->shape[2];
  return silu(conv2d(x, c.w, c.b, stride, k / 2));
}

// C2f: cv1 -> split(2) -> n bottlenecks (shortcut add) -> concat -> cv2.
// convs = [cv1, (b0.cv1,b0.cv2), (b1.cv1,b1.cv2)..., cv2]
inline Tensor c2f(const Tensor& x, const std::vector<Conv>& convs, int64_t n_bott,
                  bool shortcut = true) {
  auto y0 = conv_fused(x, convs.front());           // 2c channels
  int64_t twoc = y0->shape[1], c = twoc / 2;
  std::vector<Tensor> outs = {slice_ch(y0, 0, c), slice_ch(y0, c, twoc)};
  Tensor last = outs[1];
  for (int64_t i = 0; i < n_bott; ++i) {
    auto h = conv_fused(last, convs[1 + 2 * i]);
    h = conv_fused(h, convs[2 + 2 * i]);
    last = shortcut ? add(h, last) : h;
    outs.push_back(last);
  }
  return conv_fused(concat_ch(outs), convs.back());
}

// SPPF: cv1 -> 3x maxpool(k5,s1,p2) -> concat(4) -> cv2.
inline Tensor sppf(const Tensor& x, const std::vector<Conv>& convs) {
  auto x1 = conv_fused(x, convs[0]);
  auto p1 = maxpool2d(x1, 5, 1, 2);
  auto p2 = maxpool2d(p1, 5, 1, 2);
  auto p3 = maxpool2d(p2, 5, 1, 2);
  return conv_fused(concat_ch({x1, p1, p2, p3}), convs[1]);
}

// load a block's convs + input from a data dir; returns {input, convs, n_bott, ref}.
struct Block { Tensor x; std::vector<Conv> convs; int64_t n_bott; std::vector<float> ref; };
inline Block load_block(const std::string& D) {
  std::ifstream f(D + "meta.txt");
  int64_t n_bott, n_conv, Cin, H, W; f >> n_bott >> n_conv >> Cin >> H >> W;
  Block blk; blk.n_bott = n_bott;
  blk.x = from_data({1, Cin, H, W}, read_bin(D + "x.bin"));
  for (int64_t i = 0; i < n_conv; ++i) {
    int64_t Co, Ci, k; f >> Co >> Ci >> k;
    Conv c;
    c.w = from_data({Co, Ci, k, k}, read_bin(D + "w" + std::to_string(i) + ".bin"));
    c.b = from_data({Co}, read_bin(D + "b" + std::to_string(i) + ".bin"));
    blk.convs.push_back(c);
  }
  blk.ref = read_bin(D + "ref.bin");
  return blk;
}
inline double maxdiff(const Tensor& y, const std::vector<float>& ref) {
  double m = 0; for (int64_t i = 0; i < y->numel(); ++i)
    m = std::max(m, (double)std::abs(y->data[i] - ref[i]));
  return m;
}
