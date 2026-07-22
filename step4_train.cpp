// Step 4 (option B): a tiny CNN + YOLOv8 detection head, trained end-to-end in C++
// using v8loss.h. Overfits a fixed synthetic batch -> loss must go down.
#include "v8loss.h"
#include <torch/torch.h>

// Tiny backbone: 40x40 -> 5x5, then 1x1 head -> (4*reg_max + nc) channels.
struct TinyYOLO : torch::nn::Module {
  torch::nn::Conv2d c1{nullptr}, c2{nullptr}, c3{nullptr}, head{nullptr};
  TinyYOLO(int64_t no) {
    auto opt = [](int64_t i, int64_t o) {
      return torch::nn::Conv2dOptions(i, o, 3).stride(2).padding(1);
    };
    c1 = register_module("c1", torch::nn::Conv2d(opt(3, 16)));
    c2 = register_module("c2", torch::nn::Conv2d(opt(16, 32)));
    c3 = register_module("c3", torch::nn::Conv2d(opt(32, 64)));
    head = register_module("head", torch::nn::Conv2d(
        torch::nn::Conv2dOptions(64, no, 1)));
  }
  torch::Tensor forward(torch::Tensor x) {
    x = torch::relu(c1->forward(x));
    x = torch::relu(c2->forward(x));
    x = torch::relu(c3->forward(x));
    return head->forward(x);                    // (B, no, 5, 5)
  }
};

// make_anchors for a single feature level (grid units), Ultralytics-style (+0.5 offset).
static std::pair<torch::Tensor, torch::Tensor> make_anchors(int64_t H, int64_t W, double stride) {
  auto sx = torch::arange(W, torch::kFloat32) + 0.5;
  auto sy = torch::arange(H, torch::kFloat32) + 0.5;
  auto g = torch::meshgrid({sy, sx}, "ij");
  auto anc = torch::stack({g[1].flatten(), g[0].flatten()}, 1);   // (A,2) [x,y]
  auto st = torch::full({H * W, 1}, stride, torch::kFloat32);
  return {anc, st};
}

int main() {
 try {
  torch::manual_seed(0);
  Meta m;
  m.NC = 4; m.M = 3; m.REG_MAX = 16; m.TOPK = 10;
  m.ALPHA = 0.5; m.BETA = 6.0; m.GBOX = 7.5; m.GCLS = 0.5; m.GDFL = 1.5;
  const int64_t IMG = 40, GRID = 5, no = 4 * m.REG_MAX + m.NC;
  const double STRIDE = 8.0;
  m.B = 2; m.A = GRID * GRID;

  // fixed synthetic batch
  auto img = torch::randn({m.B, 3, IMG, IMG});
  std::vector<float> gtb = {8, 8, 32, 32,  20, 6, 39, 30,  6, 18, 30, 39,
                           4, 4, 28, 36,  18, 16, 39, 39,  2, 20, 24, 39};
  auto gt_bboxes = torch::from_blob(gtb.data(), {m.B, m.M, 4}, torch::kFloat32).clone();
  std::vector<int64_t> gtl = {1, 2, 0, 3, 0, 2};
  auto gt_labels = torch::from_blob(gtl.data(), {m.B, m.M, 1}, torch::kLong).clone();
  auto mask_gt = torch::ones({m.B, m.M, 1});
  torch::Tensor anc, stride;
  std::tie(anc, stride) = make_anchors(GRID, GRID, STRIDE);

  TinyYOLO net(no);
  torch::optim::Adam opt(net.parameters(), torch::optim::AdamOptions(1e-2));

  std::cout << "iter |   total     box      cls      dfl\n";
  for (int it = 0; it <= 200; ++it) {
    auto out = net.forward(img);                          // (B,no,5,5)
    auto flat = out.view({m.B, no, m.A}).permute({0, 2, 1}).contiguous();  // (B,A,no)
    auto pred_distri = flat.index({Slice(), Slice(), Slice(0, 4 * m.REG_MAX)});
    auto pred_scores = flat.index({Slice(), Slice(), Slice(4 * m.REG_MAX, no)});

    torch::Tensor total, box, cls, dfl;
    std::tie(total, box, cls, dfl) =
        v8_loss(pred_distri, pred_scores, anc, stride, gt_labels, gt_bboxes, mask_gt, m);

    opt.zero_grad();
    total.backward();
    opt.step();

    if (it % 20 == 0)
      printf("%4d | %8.4f %8.4f %8.4f %8.4f\n", it, total.item<float>(),
             box.item<float>(), cls.item<float>(), dfl.item<float>());
  }
  std::cout << "done.\n";
  return 0;
 } catch (const std::exception& e) {
  std::cerr << "EXCEPTION: " << e.what() << std::endl;
  return 2;
 }
}
