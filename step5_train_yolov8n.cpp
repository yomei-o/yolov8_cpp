// Step 4A: load the REAL yolov8n (TorchScript, training-mode head) in C++ and train it
// end-to-end with our v8loss.h. Overfits a synthetic 64x64 batch -> loss must go down.
#include "v8loss.h"
#include <torch/script.h>

// Multi-level make_anchors (grid units, +0.5 offset), concatenated over levels.
static std::pair<torch::Tensor, torch::Tensor>
make_anchors(std::vector<std::tuple<int64_t, int64_t, double>> levels) {
  std::vector<torch::Tensor> ap, st;
  for (auto& lv : levels) {
    int64_t H = std::get<0>(lv), W = std::get<1>(lv); double s = std::get<2>(lv);
    auto sx = torch::arange(W, torch::kFloat32) + 0.5;
    auto sy = torch::arange(H, torch::kFloat32) + 0.5;
    auto g = torch::meshgrid({sy, sx}, "ij");
    ap.push_back(torch::stack({g[1].flatten(), g[0].flatten()}, 1));   // (H*W,2)
    st.push_back(torch::full({H * W, 1}, s, torch::kFloat32));
  }
  return {torch::cat(ap, 0), torch::cat(st, 0)};
}

int main() {
 try {
  torch::manual_seed(0);
  Meta m;
  m.B = 2; m.NC = 80; m.M = 3; m.REG_MAX = 16; m.TOPK = 10;
  m.ALPHA = 0.5; m.BETA = 6.0; m.GBOX = 7.5; m.GCLS = 0.5; m.GDFL = 1.5;
  const int64_t IMG = 64;

  auto mod = torch::jit::load("yolov8n_train.torchscript");
  mod.train();
  std::vector<torch::Tensor> params;
  for (auto p : mod.parameters()) { p.set_requires_grad(true); params.push_back(p); }
  std::cout << "loaded yolov8n: " << params.size() << " parameter tensors\n";

  // anchors for strides 8/16/32 at imgsz 64 -> grids 8,4,2 ; A = 84
  torch::Tensor anc, stride;
  std::tie(anc, stride) = make_anchors({{8, 8, 8.0}, {4, 4, 16.0}, {2, 2, 32.0}});
  m.A = anc.size(0);

  // fixed synthetic batch (image + gt in image units, labels in [0,79])
  auto img = torch::randn({m.B, 3, IMG, IMG});
  std::vector<float> gtb = {8, 8, 40, 44,  30, 10, 60, 40,  10, 30, 34, 60,
                           4, 4, 44, 52,  28, 24, 60, 60,  6, 34, 38, 60};
  auto gt_bboxes = torch::from_blob(gtb.data(), {m.B, m.M, 4}, torch::kFloat32).clone();
  std::vector<int64_t> gtl = {12, 40, 7, 63, 3, 25};
  auto gt_labels = torch::from_blob(gtl.data(), {m.B, m.M, 1}, torch::kLong).clone();
  auto mask_gt = torch::ones({m.B, m.M, 1});

  torch::optim::Adam opt(params, torch::optim::AdamOptions(1e-3));

  std::cout << "iter |   total     box      cls      dfl\n";
  for (int it = 0; it <= 60; ++it) {
    std::vector<torch::jit::IValue> in{img};
    auto out = mod.forward(in).toTuple();
    auto boxes  = out->elements()[0].toTensor();     // (B, 4*reg_max, A)
    auto scores = out->elements()[1].toTensor();     // (B, nc, A)
    auto pred_distri = boxes.permute({0, 2, 1}).contiguous();   // (B,A,64)
    auto pred_scores = scores.permute({0, 2, 1}).contiguous();  // (B,A,80)

    torch::Tensor total, box, cls, dfl;
    std::tie(total, box, cls, dfl) =
        v8_loss(pred_distri, pred_scores, anc, stride, gt_labels, gt_bboxes, mask_gt, m);

    opt.zero_grad();
    total.backward();
    opt.step();

    if (it % 10 == 0)
      printf("%4d | %8.4f %8.4f %8.4f %8.4f\n", it, total.item<float>(),
             box.item<float>(), cls.item<float>(), dfl.item<float>());
  }
  std::cout << "done.\n";
  return 0;
 } catch (const std::exception& e) {
  std::cerr << "EXCEPTION: " << e.what() << std::endl; return 2;
 }
}
