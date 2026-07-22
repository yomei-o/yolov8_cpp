// Step 3a: DFL (Distribution Focal Loss) forward, parity with loss_dfl_ref.py.
#include <torch/torch.h>
#include <fstream>
#include <vector>
#include <string>
#include <iostream>
using torch::indexing::Slice;

static torch::Tensor load(const std::string& p, std::vector<int64_t> shape) {
  std::ifstream f(p, std::ios::binary | std::ios::ate);
  if (!f) { std::cerr << "cannot open " << p << "\n"; std::exit(1); }
  std::streamsize n = f.tellg(); f.seekg(0);
  std::vector<float> v(n / sizeof(float));
  f.read(reinterpret_cast<char*>(v.data()), n);
  return torch::from_blob(v.data(), shape, torch::kFloat32).clone();
}
static void write_bin(const std::string& p, const torch::Tensor& t) {
  auto c = t.contiguous().to(torch::kFloat32).cpu();
  std::ofstream f(p, std::ios::binary);
  f.write(reinterpret_cast<const char*>(c.data_ptr<float>()), c.numel() * sizeof(float));
}

// DFLoss: pred_dist (N,4,reg_max) logits, target (N,4) continuous -> (N,1)
static torch::Tensor dfl_loss(torch::Tensor pred_dist, torch::Tensor target, int64_t reg_max) {
  target = target.clamp(0, reg_max - 1 - 0.01);
  auto tl = target.to(torch::kLong);          // floor (target >= 0)
  auto tr = tl + 1;
  auto wl = tr.to(torch::kFloat) - target;
  auto wr = 1 - wl;
  auto logp = pred_dist.log_softmax(-1);      // (N,4,reg_max)  CE = -logsoftmax[target]
  auto ce_l = -logp.gather(-1, tl.unsqueeze(-1)).squeeze(-1);   // (N,4)
  auto ce_r = -logp.gather(-1, tr.unsqueeze(-1)).squeeze(-1);
  return (ce_l * wl + ce_r * wr).mean(-1, /*keepdim=*/true);    // (N,1)
}

int main() {
  int64_t N, REG_MAX;
  { std::ifstream f("data/dfl_meta.txt"); f >> N >> REG_MAX; }
  auto pred_dist = load("data/d_pred_dist.bin", {N, 4, REG_MAX});
  auto target = load("data/d_target_ltrb.bin", {N, 4});

  auto loss = dfl_loss(pred_dist, target, REG_MAX);
  write_bin("out/cpp_dfl.bin", loss);
  std::cout << "DFL done. loss[:3] = " << loss.view({-1}).index({Slice(0,3)})
            << "total mean = " << loss.mean().item<float>() << "\n";
  return 0;
}
