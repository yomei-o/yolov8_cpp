// M3c: full yolov8n forward in the pure engine, boxes/scores matched to the real net.
#include "net.hpp"
#include <cstdio>

int main() {
  const std::string D = "pure/ref/data_net/";
  auto prov = load_net(D);
  auto x = from_data({1, 3, 64, 64}, rd(D + "x.bin"));

  auto levels = yolov8n_forward(x, prov);
  printf("consumed %zu/%zu convs\n", prov.i, prov.convs.size());

  // assemble boxes (1,64,Atot) and scores (1,80,Atot) in Python's anchor order
  int64_t NB = levels[0].first->shape[1], NS = levels[0].second->shape[1];
  int64_t Atot = 0; for (auto& lv : levels) Atot += lv.first->shape[2] * lv.first->shape[3];
  std::vector<float> boxes(NB * Atot), scores(NS * Atot);
  auto pack = [&](std::vector<float>& dst, bool box) {
    int64_t aoff = 0;
    for (auto& lv : levels) {
      const Tensor& t = box ? lv.first : lv.second;
      int64_t C = t->shape[1], hw = t->shape[2] * t->shape[3];
      for (int64_t c = 0; c < C; ++c)
        for (int64_t a = 0; a < hw; ++a)
          dst[c * Atot + aoff + a] = t->data[c * hw + a];
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
  printf("\n%s\n", ok ? "M3c: PURE ENGINE == yolov8n forward" : "MISMATCH");
  return ok ? 0 : 1;
}
