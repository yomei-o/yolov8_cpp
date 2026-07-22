// A-2 (read): load a .onnx written by onnx_export and run it purely graph-driven
// (parse -> interpret with pure ops), with no architecture manifest or weight Provider.
// Verify the result matches the reference forward.  run: m13_onnx_run [model]
#include "onnx_run.hpp"
#include "net.hpp"        // rd()
#include <cstdio>
#include <string>

int main(int argc, char** argv) {
  std::string model = argc > 1 ? argv[1] : "yolov8n";
  const std::string D = "pure/ref/data_arch_" + model + "/";

  auto g = onx::load_onnx(model + ".onnx");
  std::ifstream io(D + "io.txt"); int64_t IMG, NB, NS, RM, Atot; io >> IMG >> NB >> NS >> RM >> Atot;
  auto x = from_data({1, 3, IMG, IMG}, rd(D + "x.bin"));

  printf("%s.onnx: %zu nodes, %zu float inits, %zu int inits\n",
         model.c_str(), g.nodes.size(), g.init_f.size(), g.init_i.size());
  auto vals = onx::run_onnx(g, x);

  // pack box0..2 / cls0..2 into (NB,Atot)/(NS,Atot) like the reference
  std::vector<float> boxes(NB * Atot), scores(NS * Atot);
  auto pack = [&](std::vector<float>& dst, const char* prefix, int64_t C) {
    int64_t off = 0;
    for (int l = 0; l < 3; ++l) {
      Tensor t = vals.at(prefix + std::to_string(l));
      int64_t hw = t->shape[2] * t->shape[3];
      for (int64_t c = 0; c < C; ++c) for (int64_t a = 0; a < hw; ++a) dst[c * Atot + off + a] = t->data[c * hw + a];
      off += hw;
    }
  };
  pack(boxes, "box", NB); pack(scores, "cls", NS);

  auto rb = rd(D + "ref_boxes.bin"), rs = rd(D + "ref_scores.bin");
  double db = 0, ds = 0;
  for (size_t i = 0; i < boxes.size(); ++i) db = std::max(db, (double)std::abs(boxes[i] - rb[i]));
  for (size_t i = 0; i < scores.size(); ++i) ds = std::max(ds, (double)std::abs(scores[i] - rs[i]));
  printf("boxes  max|diff| = %.3e  %s\n", db, db < 1e-3 ? "OK" : "FAIL");
  printf("scores max|diff| = %.3e  %s\n", ds, ds < 1e-3 ? "OK" : "FAIL");
  bool ok = db < 1e-3 && ds < 1e-3;
  printf("\n%s\n", ok ? ("A-2(read): pure engine runs " + model + ".onnx == yolov8 forward").c_str() : "MISMATCH");
  return ok ? 0 : 1;
}
