// Full yolov8n forward on the pure autograd engine. Consumes convs in the exact
// order m3c_export.py emits them (a running provider), so weights line up 1:1.
#pragma once
#include "autograd.hpp"
#include <fstream>
#include <string>
#include <cstdio>
#include <cstdlib>

struct ConvW { Tensor w, b; int64_t stride, act; };

struct Provider {
  std::vector<ConvW> convs;
  size_t i = 0;
  ConvW& next() { return convs[i++]; }
};

inline std::vector<float> rd(const std::string& p) {
  std::ifstream f(p, std::ios::binary | std::ios::ate);
  if (!f) { printf("cannot open %s\n", p.c_str()); std::exit(1); }
  auto n = f.tellg(); f.seekg(0);
  std::vector<float> v(n / sizeof(float)); f.read((char*)v.data(), n); return v;
}
inline Provider load_net(const std::string& D) {
  std::ifstream f(D + "manifest.txt"); int n; f >> n;
  Provider p;
  for (int i = 0; i < n; ++i) {
    int64_t Co, Ci, k, s, act; f >> Co >> Ci >> k >> s >> act;
    ConvW c; c.stride = s; c.act = act;
    c.w = from_data({Co, Ci, k, k}, rd(D + "w" + std::to_string(i) + ".bin"));
    c.b = from_data({Co}, rd(D + "b" + std::to_string(i) + ".bin"));
    p.convs.push_back(c);
  }
  return p;
}

// pack per-level (B,C,h,w) feature maps into (B*Atot, C), anchor-major per batch.
inline Tensor pack_levels(const std::vector<Tensor>& lv, int64_t B, int64_t Atot, int64_t C) {
  auto o = make_tensor({B * Atot, C}, true);
  int64_t off = 0;
  for (auto& t : lv) {
    int64_t hw = t->shape[2] * t->shape[3];
    for (int64_t b = 0; b < B; ++b) for (int64_t c = 0; c < C; ++c) for (int64_t a = 0; a < hw; ++a)
      o->data[(b * Atot + off + a) * C + c] = t->data[((b * C + c) * hw) + a];
    off += hw;
  }
  o->parents = lv; Node* op = o.get();
  o->backward_fn = [lv, op, B, Atot, C] {
    int64_t off = 0;
    for (auto& t : lv) {
      int64_t hw = t->shape[2] * t->shape[3];
      for (int64_t b = 0; b < B; ++b) for (int64_t c = 0; c < C; ++c) for (int64_t a = 0; a < hw; ++a)
        t->grad[((b * C + c) * hw) + a] += op->grad[(b * Atot + off + a) * C + c];
      off += hw;
    }
  };
  return o;
}

inline Tensor conv_apply(const Tensor& x, ConvW& c) {
  auto y = conv2d(x, c.w, c.b, c.stride, c.w->shape[2] / 2);
  return c.act ? silu(y) : y;
}
inline Tensor cL(const Tensor& x, Provider& p) { return conv_apply(x, p.next()); }

inline Tensor c2f_p(const Tensor& x, Provider& p, int64_t n_bott, bool shortcut) {
  auto y0 = conv_apply(x, p.next());
  int64_t twoc = y0->shape[1], c = twoc / 2;
  std::vector<Tensor> outs = {slice_ch(y0, 0, c), slice_ch(y0, c, twoc)};
  Tensor last = outs[1];
  for (int64_t i = 0; i < n_bott; ++i) {
    auto h = conv_apply(last, p.next());
    h = conv_apply(h, p.next());
    last = shortcut ? add(h, last) : h;
    outs.push_back(last);
  }
  return conv_apply(concat_ch(outs), p.next());
}
inline Tensor sppf_p(const Tensor& x, Provider& p) {
  auto x1 = conv_apply(x, p.next());
  auto q1 = maxpool2d(x1, 5, 1, 2), q2 = maxpool2d(q1, 5, 1, 2), q3 = maxpool2d(q2, 5, 1, 2);
  return conv_apply(concat_ch({x1, q1, q2, q3}), p.next());
}

// Detect head output for one level: box (b,4*rm,h,w), cls (b,nc,h,w).
inline std::pair<Tensor, Tensor> detect_level(const Tensor& x, Provider& p) {
  auto hb = cL(x, p); hb = cL(hb, p); auto box = conv_apply(hb, p.next());   // last plain
  auto hc = cL(x, p); hc = cL(hc, p); auto cls = conv_apply(hc, p.next());
  return {box, cls};
}

// Full yolov8n. Returns the 3 (box,cls) level pairs.
inline std::vector<std::pair<Tensor, Tensor>> yolov8n_forward(const Tensor& x, Provider& p) {
  auto x0 = cL(x, p);
  auto x1 = cL(x0, p);
  auto x2 = c2f_p(x1, p, 1, true);
  auto x3 = cL(x2, p);
  auto x4 = c2f_p(x3, p, 2, true);              // P3 save
  auto x5 = cL(x4, p);
  auto x6 = c2f_p(x5, p, 2, true);              // P4 save
  auto x7 = cL(x6, p);
  auto x8 = c2f_p(x7, p, 1, true);
  auto x9 = sppf_p(x8, p);                       // P5 save
  // neck
  auto u10 = upsample_nearest(x9, 2);
  auto x12 = c2f_p(concat_ch({u10, x6}), p, 1, false);
  auto u13 = upsample_nearest(x12, 2);
  auto x15 = c2f_p(concat_ch({u13, x4}), p, 1, false);   // detect P3
  auto x16 = cL(x15, p);
  auto x18 = c2f_p(concat_ch({x16, x12}), p, 1, false);  // detect P4
  auto x19 = cL(x18, p);
  auto x21 = c2f_p(concat_ch({x19, x9}), p, 1, false);   // detect P5
  // detect
  std::vector<std::pair<Tensor, Tensor>> out;
  for (auto& xi : {x15, x18, x21}) out.push_back(detect_level(xi, p));
  return out;
}
