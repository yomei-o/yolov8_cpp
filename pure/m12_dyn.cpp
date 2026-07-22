// B: data-driven yolov8 forward. Same builder runs any exported size; here it is
// checked against the real net for whatever model dir is given (default yolov8n).
//   python pure/ref/export_arch.py yolov8m 64   &&   m12_dyn yolov8m
#include "net_dyn.hpp"
#include <cstdio>
#include <string>

int main(int argc, char** argv) {
  std::string model = argc > 1 ? argv[1] : "yolov8n";
  const std::string D = "pure/ref/data_arch_" + model + "/";

  auto arch = load_arch(D + "arch.txt");
  auto prov = load_net(D);
  std::ifstream io(D + "io.txt"); int64_t IMG; io >> IMG;
  auto x = from_data({1, 3, IMG, IMG}, rd(D + "x.bin"));

  auto levels = yolov8_forward_dyn(x, prov, arch);
  printf("%s: %d layers, consumed %zu/%zu convs\n", model.c_str(), arch.nlayers, prov.i, prov.convs.size());

  int64_t Atot = 0; for (auto& lv : levels) Atot += lv.first->shape[2] * lv.first->shape[3];
  int64_t NB = levels[0].first->shape[1], NS = levels[0].second->shape[1];
  std::vector<float> boxes(NB * Atot), scores(NS * Atot);
  auto pack = [&](std::vector<float>& dst, bool box) {
    int64_t aoff = 0;
    for (auto& lv : levels) {
      const Tensor& t = box ? lv.first : lv.second;
      int64_t Cc = t->shape[1], hw = t->shape[2] * t->shape[3];
      for (int64_t c = 0; c < Cc; ++c) for (int64_t k = 0; k < hw; ++k) dst[c * Atot + aoff + k] = t->data[c * hw + k];
      aoff += hw;
    }
  };
  pack(boxes, true); pack(scores, false);

  auto rb = rd(D + "ref_boxes.bin"), rs = rd(D + "ref_scores.bin");
  double db = 0, ds = 0;
  for (size_t i = 0; i < boxes.size(); ++i) db = std::max(db, (double)std::abs(boxes[i] - rb[i]));
  for (size_t i = 0; i < scores.size(); ++i) ds = std::max(ds, (double)std::abs(scores[i] - rs[i]));
  printf("boxes  max|diff| = %.3e  %s\n", db, db < 1e-3 ? "OK" : "FAIL");
  printf("scores max|diff| = %.3e  %s\n", ds, ds < 1e-3 ? "OK" : "FAIL");
  bool ok = db < 1e-3 && ds < 1e-3;
  printf("\n%s\n", ok ? ("B: DATA-DRIVEN forward == " + model).c_str() : "MISMATCH");
  return ok ? 0 : 1;
}
