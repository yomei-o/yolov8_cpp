// Data-driven yolov8 forward: builds the graph from an architecture manifest
// (type / from / params) instead of hardcoded topology, so n/s/m/l/x all run through
// the same code — only the weights and the per-layer C2f depth differ. Reuses the
// block ops from net.hpp; consumes convs from the same Provider in canonical order.
#pragma once
#include "net.hpp"
#include <fstream>
#include <string>
#include <vector>

struct ArchLayer { int idx, type; std::vector<int> from; int nbott = 0, scale = 1; bool shortcut = false; };

struct Arch { int nlayers, nc, reg_max; std::vector<ArchLayer> layers; };

inline Arch load_arch(const std::string& path) {
  std::ifstream f(path); if (!f) { printf("cannot open %s\n", path.c_str()); std::exit(1); }
  Arch a; f >> a.nlayers >> a.nc >> a.reg_max;
  for (int i = 0; i < a.nlayers; ++i) {
    ArchLayer L; int nf; f >> L.idx >> L.type >> nf;
    L.from.resize(nf); for (int j = 0; j < nf; ++j) f >> L.from[j];
    if (L.type == 1) { int sc; f >> L.nbott >> sc; L.shortcut = sc != 0; }   // C2f: n, shortcut
    else if (L.type == 3) f >> L.scale;                                       // Upsample: scale
    a.layers.push_back(std::move(L));
  }
  return a;
}

// Returns the 3 (box, cls) level pairs, exactly like yolov8n_forward.
inline std::vector<std::pair<Tensor, Tensor>> yolov8_forward_dyn(const Tensor& x, Provider& p, const Arch& a) {
  std::vector<Tensor> outs(a.nlayers);
  std::vector<std::pair<Tensor, Tensor>> detect;
  for (const auto& L : a.layers) {
    std::vector<Tensor> in;
    for (int fv : L.from) {
      if (fv == -1) in.push_back(L.idx == 0 ? x : outs[L.idx - 1]);   // -1 = prev layer (or input for layer 0)
      else in.push_back(outs[fv]);
    }
    switch (L.type) {
      case 0: outs[L.idx] = conv_apply(in[0], p.next()); break;          // Conv (stride from weight manifest)
      case 1: outs[L.idx] = c2f_p(in[0], p, L.nbott, L.shortcut); break; // C2f
      case 2: outs[L.idx] = sppf_p(in[0], p); break;                     // SPPF
      case 3: outs[L.idx] = upsample_nearest(in[0], L.scale); break;     // Upsample
      case 4: outs[L.idx] = concat_ch(in); break;                        // Concat
      case 5: for (auto& t : in) detect.push_back(detect_level(t, p)); break;  // Detect
      default: printf("bad layer type %d\n", L.type); std::exit(1);
    }
  }
  return detect;
}
