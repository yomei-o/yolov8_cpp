// Full yolov8n forward with conv + BatchNorm2d + SiLU kept SEPARATE (BN not folded),
// so the same weights can be trained and written back to a .pt. Same topology as
// net.hpp; consumes layers in the order pure/ref/export_unfused.py emits them.
#pragma once
#include "autograd.hpp"
#include "bn.hpp"
#include "net.hpp"          // reuse rd(), pack_levels()
#include <fstream>
#include <string>

struct LayerU {
  int kind;                 // 1 = conv+bn+silu, 0 = plain conv (bias, no act)
  int64_t stride; float eps;
  Tensor w, b, gamma, beta; // b only for plain
  std::vector<float> rm, rv;
};

struct ProviderU { std::vector<LayerU> layers; size_t i = 0; LayerU& next() { return layers[i++]; } };

inline ProviderU load_net_unfused(const std::string& D) {
  std::ifstream f(D + "manifest_unfused.txt"); if (!f) { printf("run: python pure/ref/export_unfused.py\n"); std::exit(1); }
  int n; f >> n; ProviderU p;
  for (int i = 0; i < n; ++i) {
    int kind; int64_t Co, Ci, k, s; float eps; f >> kind >> Co >> Ci >> k >> s >> eps;
    LayerU L; L.kind = kind; L.stride = s; L.eps = eps;
    L.w = from_data({Co, Ci, k, k}, rd(D + "cw" + std::to_string(i) + ".bin"), true);
    if (kind == 1) {
      L.gamma = from_data({Co}, rd(D + "bg" + std::to_string(i) + ".bin"), true);
      L.beta  = from_data({Co}, rd(D + "bb" + std::to_string(i) + ".bin"), true);
      L.rm = rd(D + "rm" + std::to_string(i) + ".bin");
      L.rv = rd(D + "rv" + std::to_string(i) + ".bin");
    } else {
      L.b = from_data({Co}, rd(D + "cb" + std::to_string(i) + ".bin"), true);
    }
    p.layers.push_back(std::move(L));
  }
  return p;
}

// Same as load_net_unfused but weights come from a state_dict .pt (read in pure C++ by
// ptio) instead of per-layer .bin files. Arch is from the tiny committable manifest; each
// tensor is looked up by its state_dict key from names.txt (engine order). No Python.
#include "ptio.hpp"
#include <map>
inline ProviderU load_net_unfused_pt(const std::string& D, const std::string& pt_path) {
  std::ifstream f(D + "manifest_unfused.txt"); if (!f) { printf("missing %smanifest_unfused.txt\n", D.c_str()); std::exit(1); }
  std::vector<std::string> names; { std::ifstream nf(D + "names.txt"); std::string s; while (nf >> s) names.push_back(s); }
  std::map<std::string, std::vector<float>> W; for (auto& t : pt::load_pt(pt_path)) W[t.name] = t.data;
  auto get = [&](size_t idx) -> std::vector<float>& {
    auto it = W.find(names[idx]); if (it == W.end()) { printf("key not in %s: %s\n", pt_path.c_str(), names[idx].c_str()); std::exit(1); } return it->second; };
  int n; f >> n; ProviderU p; size_t ni = 0;
  for (int i = 0; i < n; ++i) {
    int kind; int64_t Co, Ci, k, s; float eps; f >> kind >> Co >> Ci >> k >> s >> eps;
    LayerU L; L.kind = kind; L.stride = s; L.eps = eps;
    L.w = from_data({Co, Ci, k, k}, get(ni++), true);
    if (kind == 1) {
      L.gamma = from_data({Co}, get(ni++), true);
      L.beta  = from_data({Co}, get(ni++), true);
      L.rm = get(ni++); L.rv = get(ni++);
    } else { L.b = from_data({Co}, get(ni++), true); }
    p.layers.push_back(std::move(L));
  }
  return p;
}

inline Tensor applyU(const Tensor& x, LayerU& L, bool training) {
  int64_t pad = L.w->shape[2] / 2;
  if (L.kind == 1) {
    auto y = conv2d(x, L.w, nullptr, L.stride, pad);
    y = batchnorm2d(y, L.gamma, L.beta, L.rm, L.rv, L.eps, training, 0.03f);
    return silu(y);
  }
  return conv2d(x, L.w, L.b, L.stride, pad);
}

inline Tensor cLU(const Tensor& x, ProviderU& p, bool tr) { return applyU(x, p.next(), tr); }

inline Tensor c2f_u(const Tensor& x, ProviderU& p, int64_t n_bott, bool shortcut, bool tr) {
  auto y0 = applyU(x, p.next(), tr);
  int64_t twoc = y0->shape[1], c = twoc / 2;
  std::vector<Tensor> outs = {slice_ch(y0, 0, c), slice_ch(y0, c, twoc)};
  Tensor last = outs[1];
  for (int64_t i = 0; i < n_bott; ++i) {
    auto h = applyU(last, p.next(), tr); h = applyU(h, p.next(), tr);
    last = shortcut ? add(h, last) : h; outs.push_back(last);
  }
  return applyU(concat_ch(outs), p.next(), tr);
}
inline Tensor sppf_u(const Tensor& x, ProviderU& p, bool tr) {
  auto x1 = applyU(x, p.next(), tr);
  auto q1 = maxpool2d(x1, 5, 1, 2), q2 = maxpool2d(q1, 5, 1, 2), q3 = maxpool2d(q2, 5, 1, 2);
  return applyU(concat_ch({x1, q1, q2, q3}), p.next(), tr);
}
inline std::pair<Tensor, Tensor> detect_level_u(const Tensor& x, ProviderU& p, bool tr) {
  auto hb = cLU(x, p, tr); hb = cLU(hb, p, tr); auto box = applyU(hb, p.next(), tr);
  auto hc = cLU(x, p, tr); hc = cLU(hc, p, tr); auto cls = applyU(hc, p.next(), tr);
  return {box, cls};
}

inline std::vector<std::pair<Tensor, Tensor>> yolov8n_forward_u(const Tensor& x, ProviderU& p, bool tr, const std::vector<int64_t>& d = {1,2,2,1,1,1,1,1}) {
  auto x0 = cLU(x, p, tr);
  auto x1 = cLU(x0, p, tr);
  auto x2 = c2f_u(x1, p, d[0], true, tr);
  auto x3 = cLU(x2, p, tr);
  auto x4 = c2f_u(x3, p, d[1], true, tr);
  auto x5 = cLU(x4, p, tr);
  auto x6 = c2f_u(x5, p, d[2], true, tr);
  auto x7 = cLU(x6, p, tr);
  auto x8 = c2f_u(x7, p, d[3], true, tr);
  auto x9 = sppf_u(x8, p, tr);
  auto u10 = upsample_nearest(x9, 2);
  auto x12 = c2f_u(concat_ch({u10, x6}), p, d[4], false, tr);
  auto u13 = upsample_nearest(x12, 2);
  auto x15 = c2f_u(concat_ch({u13, x4}), p, d[5], false, tr);
  auto x16 = cLU(x15, p, tr);
  auto x18 = c2f_u(concat_ch({x16, x12}), p, d[6], false, tr);
  auto x19 = cLU(x18, p, tr);
  auto x21 = c2f_u(concat_ch({x19, x9}), p, d[7], false, tr);
  std::vector<std::pair<Tensor, Tensor>> out;
  for (auto& xi : {x15, x18, x21}) out.push_back(detect_level_u(xi, p, tr));
  return out;
}
