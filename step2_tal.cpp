// Step 2: TaskAlignedAssigner (TAL) in C++/LibTorch, parity with tal_ref.py.
// Runs entirely under NoGradGuard (TAL is a non-differentiable matching step).
#include <torch/torch.h>
#include <fstream>
#include <vector>
#include <string>
#include <iostream>

using torch::indexing::Slice;

static std::vector<float> g_keep;  // storage backing from_blob (kept alive)

static torch::Tensor load(const std::string& path, std::vector<int64_t> shape) {
  std::ifstream f(path, std::ios::binary | std::ios::ate);
  if (!f) { std::cerr << "cannot open " << path << "\n"; std::exit(1); }
  std::streamsize n = f.tellg(); f.seekg(0);
  std::vector<float> v(n / sizeof(float));
  f.read(reinterpret_cast<char*>(v.data()), n);
  auto t = torch::from_blob(v.data(), shape, torch::kFloat32).clone();
  return t;
}
static void write_bin(const std::string& path, const torch::Tensor& t) {
  auto c = t.contiguous().to(torch::kFloat32).cpu();
  std::ofstream f(path, std::ios::binary);
  f.write(reinterpret_cast<const char*>(c.data_ptr<float>()), c.numel() * sizeof(float));
}

// CIoU, broadcasting over the last dim (xyxy). b1,b2 broadcastable to (...,4).
static torch::Tensor bbox_ciou(torch::Tensor b1, torch::Tensor b2, double eps = 1e-7) {
  auto u1 = b1.unbind(-1); auto u2 = b2.unbind(-1);
  auto x1 = u1[0], y1 = u1[1], x2 = u1[2], y2 = u1[3];
  auto X1 = u2[0], Y1 = u2[1], X2 = u2[2], Y2 = u2[3];
  auto w1 = x2 - x1, h1 = y2 - y1, w2 = X2 - X1, h2 = Y2 - Y1;
  auto inter = (torch::min(x2, X2) - torch::max(x1, X1)).clamp_min(0) *
               (torch::min(y2, Y2) - torch::max(y1, Y1)).clamp_min(0);
  auto uni = w1 * h1 + w2 * h2 - inter + eps;
  auto iou = inter / uni;
  auto cw = torch::max(x2, X2) - torch::min(x1, X1);
  auto ch = torch::max(y2, Y2) - torch::min(y1, Y1);
  auto c2 = cw * cw + ch * ch + eps;
  auto rho2 = ((x1 + x2 - X1 - X2).pow(2) + (y1 + y2 - Y1 - Y2).pow(2)) / 4;
  auto pi = std::acos(-1.0);
  auto v = (4 / (pi * pi)) * (torch::atan(w2 / (h2 + eps)) - torch::atan(w1 / (h1 + eps))).pow(2);
  auto alpha = v / (v - iou + (1 + eps));
  return iou - (rho2 / c2 + v * alpha);
}

int main() {
  torch::NoGradGuard ng;
  // ---- meta ----
  int64_t B, NC, M, A, TOPK; double ALPHA, BETA; const double EPS = 1e-9;
  { std::ifstream f("data/tal_meta.txt"); f >> B >> NC >> M >> A >> TOPK >> ALPHA >> BETA; }

  auto pd_scores = load("data/t_pd_scores.bin", {B, A, NC});
  auto pd_bboxes = load("data/t_pd_bboxes.bin", {B, A, 4});
  auto anc       = load("data/t_anc.bin", {A, 2});
  auto gt_labels = load("data/t_gt_labels.bin", {B, M, 1}).to(torch::kLong);
  auto gt_bboxes = load("data/t_gt_bboxes.bin", {B, M, 4});
  auto mask_gt   = load("data/t_mask_gt.bin", {B, M, 1});

  // ---- candidates in gts: (B,M,A) ----
  auto lt = gt_bboxes.index({Slice(), Slice(), torch::indexing::None, Slice(0, 2)}); // (B,M,1,2)
  auto rb = gt_bboxes.index({Slice(), Slice(), torch::indexing::None, Slice(2, 4)});
  auto ancB = anc.view({1, 1, A, 2});
  auto deltas = torch::cat({ancB - lt, rb - ancB}, -1);              // (B,M,A,4)
  auto mask_in_gts = std::get<0>(deltas.min(-1)).gt(EPS);            // (B,M,A) bool
  auto mask_comb = mask_in_gts.to(torch::kFloat) * mask_gt;          // (B,M,A) 0/1

  // ---- box metrics ----
  auto gt_e = gt_bboxes.view({B, M, 1, 4});
  auto pd_e = pd_bboxes.view({B, 1, A, 4});
  auto overlaps = bbox_ciou(gt_e, pd_e).clamp_min(0) * mask_comb;    // (B,M,A)
  auto scores_t = pd_scores.permute({0, 2, 1});                      // (B,NC,A)
  auto cls_idx = gt_labels.view({B, M, 1}).expand({B, M, A});        // (B,M,A)
  auto bbox_scores = torch::gather(scores_t, 1, cls_idx) * mask_comb;// (B,M,A)
  auto align_metric = bbox_scores.pow(ALPHA) * overlaps.pow(BETA);   // (B,M,A)

  // ---- topk over anchors ----
  auto tk = torch::topk(align_metric, TOPK, -1);
  auto topk_idxs = std::get<1>(tk);                                  // (B,M,TOPK)
  auto topk_mask = mask_gt.expand({B, M, TOPK}).to(torch::kBool);
  topk_idxs = topk_idxs.masked_fill(~topk_mask, 0);
  auto count = torch::zeros_like(align_metric, torch::kInt);
  auto ones = torch::ones({B, M, 1}, torch::kInt);
  for (int64_t k = 0; k < TOPK; ++k)
    count.scatter_add_(-1, topk_idxs.index({Slice(), Slice(), Slice(k, k + 1)}), ones);
  count = count.masked_fill(count > 1, 0);
  auto mask_topk = count.to(torch::kFloat);

  auto mask_pos = mask_topk * mask_in_gts.to(torch::kFloat) * mask_gt;  // (B,M,A)

  // ---- resolve multi-gt anchors by highest overlap ----
  auto fg_mask = mask_pos.sum(1);                                    // (B,A)
  if (fg_mask.max().item<float>() > 1) {
    auto multi = (fg_mask.unsqueeze(1) > 1).expand({B, M, A});
    auto max_idx = overlaps.argmax(1);                              // (B,A)
    auto is_max = torch::zeros_like(mask_pos);
    is_max.scatter_(1, max_idx.unsqueeze(1), 1);
    mask_pos = torch::where(multi, is_max, mask_pos);
    fg_mask = mask_pos.sum(1);
  }
  auto target_gt_idx = mask_pos.argmax(1);                           // (B,A) long

  // ---- gather targets ----
  auto batch_ind = torch::arange(B, torch::kLong).view({B, 1});
  auto flat_idx = (target_gt_idx + batch_ind * M).reshape({-1});     // (B*A)
  auto labels_flat = gt_labels.reshape({-1});                        // (B*M)
  auto target_labels = labels_flat.index_select(0, flat_idx).view({B, A}).clamp_min(0);
  auto target_bboxes = gt_bboxes.reshape({B * M, 4}).index_select(0, flat_idx).view({B, A, 4});
  auto target_scores = torch::one_hot(target_labels, NC).to(torch::kFloat);  // (B,A,NC)
  auto fg_scores_mask = (fg_mask.unsqueeze(-1) > 0).expand({B, A, NC});
  target_scores = torch::where(fg_scores_mask, target_scores, torch::zeros_like(target_scores));

  // ---- normalize target_scores by alignment metric ----
  align_metric = align_metric * mask_pos;
  auto pos_am = align_metric.amax({-1}, true);                       // (B,M,1)
  auto pos_ov = (overlaps * mask_pos).amax({-1}, true);             // (B,M,1)
  auto norm = (align_metric * pos_ov / (pos_am + EPS)).amax({-2}).unsqueeze(-1);  // (B,A,1)
  target_scores = target_scores * norm;

  write_bin("out/cpp_target_labels.bin", target_labels.to(torch::kFloat));
  write_bin("out/cpp_target_bboxes.bin", target_bboxes);
  write_bin("out/cpp_target_scores.bin", target_scores);
  write_bin("out/cpp_fg_mask.bin", fg_mask);
  write_bin("out/cpp_target_gt_idx.bin", target_gt_idx.to(torch::kFloat));
  std::cout << "TAL done. fg per batch = " << fg_mask.sum(1) << "\n";
  return 0;
}
