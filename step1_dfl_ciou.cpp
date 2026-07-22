// Step 1: DFL decode + CIoU in C++ (LibTorch), for numeric parity with Ultralytics.
// Reads raw float32 inputs from ./data, writes raw float32 outputs to ./out.
#include <torch/torch.h>
#include <fstream>
#include <vector>
#include <string>
#include <iostream>

using torch::indexing::Slice;

static std::vector<float> read_bin(const std::string& path) {
  std::ifstream f(path, std::ios::binary | std::ios::ate);
  if (!f) { std::cerr << "cannot open " << path << "\n"; std::exit(1); }
  std::streamsize n = f.tellg();
  f.seekg(0);
  std::vector<float> v(n / sizeof(float));
  f.read(reinterpret_cast<char*>(v.data()), n);
  return v;
}

static void write_bin(const std::string& path, const torch::Tensor& t) {
  auto c = t.contiguous().to(torch::kFloat32).cpu();
  std::ofstream f(path, std::ios::binary);
  f.write(reinterpret_cast<const char*>(c.data_ptr<float>()),
          c.numel() * sizeof(float));
}

// DFL decode: (N, 4*reg_max) logits -> ltrb distances -> xyxy boxes (grid units) * stride
static torch::Tensor dfl_decode(torch::Tensor pred_dist, torch::Tensor anchors,
                                torch::Tensor strides, int64_t reg_max) {
  int64_t N = pred_dist.size(0);
  auto prob = pred_dist.view({N, 4, reg_max}).softmax(2);           // softmax over bins
  auto proj = torch::arange(reg_max, torch::kFloat32).view({1, 1, reg_max});
  auto dist = (prob * proj).sum(2);                                  // (N,4) ltrb
  auto lt = dist.index({Slice(), Slice(0, 2)});
  auto rb = dist.index({Slice(), Slice(2, 4)});
  auto x1y1 = anchors - lt;
  auto x2y2 = anchors + rb;
  auto boxes = torch::cat({x1y1, x2y2}, 1);                          // xyxy (grid units)
  return boxes * strides;                                            // (N,1) broadcast
}

// CIoU between paired boxes b1,b2 (both (N,4) xyxy). Returns (N,).
static torch::Tensor ciou(torch::Tensor b1, torch::Tensor b2, double eps = 1e-7) {
  auto x1 = b1.index({Slice(), 0}), y1 = b1.index({Slice(), 1});
  auto x2 = b1.index({Slice(), 2}), y2 = b1.index({Slice(), 3});
  auto X1 = b2.index({Slice(), 0}), Y1 = b2.index({Slice(), 1});
  auto X2 = b2.index({Slice(), 2}), Y2 = b2.index({Slice(), 3});

  auto w1 = (x2 - x1).clamp_min(0), h1 = (y2 - y1).clamp_min(0);
  auto w2 = (X2 - X1).clamp_min(0), h2 = (Y2 - Y1).clamp_min(0);

  auto inter = (torch::min(x2, X2) - torch::max(x1, X1)).clamp_min(0) *
               (torch::min(y2, Y2) - torch::max(y1, Y1)).clamp_min(0);
  auto uni = w1 * h1 + w2 * h2 - inter + eps;
  auto iou = inter / uni;

  auto cw = torch::max(x2, X2) - torch::min(x1, X1);
  auto ch = torch::max(y2, Y2) - torch::min(y1, Y1);
  auto c2 = cw * cw + ch * ch + eps;

  auto cx1 = (x1 + x2) / 2, cy1 = (y1 + y2) / 2;
  auto cx2 = (X1 + X2) / 2, cy2 = (Y1 + Y2) / 2;
  auto rho2 = (cx1 - cx2).pow(2) + (cy1 - cy2).pow(2);

  auto pi = std::acos(-1.0);
  auto v = (4 / (pi * pi)) *
           (torch::atan(w2 / (h2 + eps)) - torch::atan(w1 / (h1 + eps))).pow(2);
  torch::Tensor alpha;
  {
    torch::NoGradGuard ng;                 // Ultralytics detaches alpha
    alpha = v / (v - iou + (1 + eps));
  }
  return iou - (rho2 / c2 + v * alpha);
}

int main() {
  const int64_t reg_max = 16;
  auto pd = read_bin("data/pred_dist.bin");
  int64_t N = static_cast<int64_t>(pd.size()) / (4 * reg_max);

  auto pred_dist = torch::from_blob(pd.data(), {N, 4 * reg_max}, torch::kFloat32).clone();
  auto av = read_bin("data/anchors.bin");
  auto anchors = torch::from_blob(av.data(), {N, 2}, torch::kFloat32).clone();
  auto sv = read_bin("data/strides.bin");
  auto strides = torch::from_blob(sv.data(), {N, 1}, torch::kFloat32).clone();
  auto gv = read_bin("data/gt.bin");
  auto gt = torch::from_blob(gv.data(), {N, 4}, torch::kFloat32).clone();

  auto boxes = dfl_decode(pred_dist, anchors, strides, reg_max);
  auto c = ciou(boxes, gt);

  write_bin("out/cpp_boxes.bin", boxes);
  write_bin("out/cpp_ciou.bin", c);
  std::cout << "N=" << N << "  wrote out/cpp_boxes.bin, out/cpp_ciou.bin\n";
  std::cout << "boxes[0]=" << boxes[0] << "ciou[0]=" << c[0].item<float>() << "\n";
  return 0;
}
