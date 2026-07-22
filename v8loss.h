// Shared v8DetectionLoss forward (box=CIoU, cls=BCE-with-logits, dfl=DFL).
// Used by step3b (forward parity) and step3c (backward parity).
#pragma once
#include <torch/torch.h>
#include <fstream>
#include <vector>
#include <string>
#include <iostream>

using torch::indexing::Slice;
using torch::indexing::Ellipsis;
using torch::indexing::None;

struct Meta {
  int64_t B, NC, M, A, REG_MAX, TOPK;
  double ALPHA, BETA, GBOX, GCLS, GDFL;
};
inline Meta load_meta(const std::string& p) {
  Meta m; std::ifstream f(p);
  f >> m.B >> m.NC >> m.M >> m.A >> m.REG_MAX >> m.TOPK
    >> m.ALPHA >> m.BETA >> m.GBOX >> m.GCLS >> m.GDFL;
  return m;
}
inline torch::Tensor load_t(const std::string& p, std::vector<int64_t> shape) {
  std::ifstream f(p, std::ios::binary | std::ios::ate);
  if (!f) { std::cerr << "cannot open " << p << "\n"; std::exit(1); }
  std::streamsize n = f.tellg(); f.seekg(0);
  std::vector<float> v(n / sizeof(float));
  f.read(reinterpret_cast<char*>(v.data()), n);
  return torch::from_blob(v.data(), shape, torch::kFloat32).clone();
}
inline void write_bin(const std::string& p, const torch::Tensor& t) {
  auto c = t.contiguous().to(torch::kFloat32).cpu();
  std::ofstream f(p, std::ios::binary);
  f.write(reinterpret_cast<const char*>(c.data_ptr<float>()), c.numel() * sizeof(float));
}

// CIoU broadcasting over last dim (xyxy). Returns shape = broadcast of inputs minus last dim.
inline torch::Tensor bbox_ciou(torch::Tensor b1, torch::Tensor b2, double eps = 1e-7) {
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
  auto alpha = (v / (v - iou + (1 + eps))).detach();   // Ultralytics computes alpha under no_grad
  return iou - (rho2 / c2 + v * alpha);
}

// DFL decode: (B,A,4*reg_max) logits -> xyxy boxes (grid units)
inline torch::Tensor bbox_decode(torch::Tensor anc, torch::Tensor pred_dist, const Meta& m) {
  auto proj = torch::arange(m.REG_MAX, torch::kFloat32);
  auto d = pred_dist.view({m.B, m.A, 4, m.REG_MAX}).softmax(3).matmul(proj);  // (B,A,4)
  auto lt = d.index({Ellipsis, Slice(0, 2)});
  auto rb = d.index({Ellipsis, Slice(2, 4)});
  return torch::cat({anc - lt, anc + rb}, -1);
}

// TaskAlignedAssigner (no grad). Returns {target_bboxes(img units), target_scores, fg_mask(bool)}.
inline std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
assign(torch::Tensor pd_scores, torch::Tensor pd_bboxes, torch::Tensor anc_pts,
       torch::Tensor gt_labels, torch::Tensor gt_bboxes, torch::Tensor mask_gt, const Meta& m) {
  torch::NoGradGuard ng;
  const double EPS = 1e-9;
  int64_t B = m.B, M = m.M, A = m.A, NC = m.NC, TOPK = m.TOPK;
  auto lt = gt_bboxes.index({Slice(), Slice(), None, Slice(0, 2)});
  auto rb = gt_bboxes.index({Slice(), Slice(), None, Slice(2, 4)});
  auto ancB = anc_pts.view({1, 1, A, 2});
  auto deltas = torch::cat({ancB - lt, rb - ancB}, -1);
  auto mask_in = std::get<0>(deltas.min(-1)).gt(EPS);
  auto mc = mask_in.to(torch::kFloat) * mask_gt;

  auto ov = bbox_ciou(gt_bboxes.view({B, M, 1, 4}), pd_bboxes.view({B, 1, A, 4})).clamp_min(0) * mc;
  auto st = pd_scores.permute({0, 2, 1});
  auto ci = gt_labels.view({B, M, 1}).expand({B, M, A});
  auto bs = torch::gather(st, 1, ci) * mc;
  auto am = bs.pow(m.ALPHA) * ov.pow(m.BETA);

  auto ti = std::get<1>(torch::topk(am, TOPK, -1));
  auto tm = mask_gt.expand({B, M, TOPK}).to(torch::kBool);
  ti = ti.masked_fill(~tm, 0);
  auto cnt = torch::zeros_like(am, torch::kInt);
  auto one = torch::ones({B, M, 1}, torch::kInt);
  for (int64_t k = 0; k < TOPK; ++k)
    cnt.scatter_add_(-1, ti.index({Slice(), Slice(), Slice(k, k + 1)}), one);
  auto mask_topk = cnt.masked_fill(cnt > 1, 0).to(torch::kFloat);
  auto mask_pos = mask_topk * mask_in.to(torch::kFloat) * mask_gt;

  auto fg = mask_pos.sum(1);
  if (fg.max().item<float>() > 1) {
    auto multi = (fg.unsqueeze(1) > 1).expand({B, M, A});
    auto mi = ov.argmax(1);
    auto ismax = torch::zeros_like(mask_pos);
    ismax.scatter_(1, mi.unsqueeze(1), 1);
    mask_pos = torch::where(multi, ismax, mask_pos);
    fg = mask_pos.sum(1);
  }
  auto tgi = mask_pos.argmax(1);
  auto bind = torch::arange(B, torch::kLong).view({B, 1});
  auto fi = (tgi + bind * M).reshape({-1});
  auto tl = gt_labels.reshape({-1}).index_select(0, fi).view({B, A}).clamp_min(0);
  auto tb = gt_bboxes.reshape({B * M, 4}).index_select(0, fi).view({B, A, 4});
  auto ts = torch::one_hot(tl, NC).to(torch::kFloat);
  ts = torch::where((fg.unsqueeze(-1) > 0).expand({B, A, NC}), ts, torch::zeros_like(ts));
  am = am * mask_pos;
  auto pa = am.amax({-1}, true);
  auto po = (ov * mask_pos).amax({-1}, true);
  auto norm = (am * po / (pa + EPS)).amax({-2}).unsqueeze(-1);
  ts = ts * norm;
  return {tb, ts, fg.to(torch::kBool)};
}

// DFL loss: pred_dist (n,4,reg_max) logits, target (n,4) continuous -> (n,)
inline torch::Tensor dfl_loss(torch::Tensor pred_dist, torch::Tensor target, int64_t reg_max) {
  target = target.clamp(0, reg_max - 1 - 0.01);
  auto tl = target.to(torch::kLong);
  auto tr = tl + 1;
  auto wl = tr.to(torch::kFloat) - target;
  auto wr = 1 - wl;
  auto logp = pred_dist.log_softmax(-1);
  auto ce_l = -logp.gather(-1, tl.unsqueeze(-1)).squeeze(-1);
  auto ce_r = -logp.gather(-1, tr.unsqueeze(-1)).squeeze(-1);
  return (ce_l * wl + ce_r * wr).mean(-1);          // (n,)
}

// Full loss. Returns {total, box, cls, dfl}.
inline std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
v8_loss(torch::Tensor pred_distri, torch::Tensor pred_scores,
        torch::Tensor anc, torch::Tensor stride,
        torch::Tensor gt_labels, torch::Tensor gt_bboxes, torch::Tensor mask_gt, const Meta& m) {
  auto pred_bboxes = bbox_decode(anc, pred_distri, m);          // grid units (diff)

  torch::Tensor tb, ts, fg;
  std::tie(tb, ts, fg) = assign(pred_scores.detach().sigmoid(),
                                pred_bboxes.detach() * stride, anc * stride,
                                gt_labels, gt_bboxes, mask_gt, m);
  auto tss = ts.sum().clamp_min(1);

  // cls: BCE-with-logits (stable form), summed
  auto x = pred_scores, z = ts;
  auto bce = x.clamp_min(0) - x * z + torch::log1p(torch::exp(-x.abs()));
  auto loss_cls = bce.sum() / tss;

  // box: CIoU over positives
  tb = tb / stride;                                            // grid units
  auto weight = ts.sum(-1).index({fg});                        // (n,)
  auto iou = bbox_ciou(pred_bboxes.index({fg}), tb.index({fg}));
  auto loss_box = ((1 - iou) * weight).sum() / tss;

  // dfl over positives
  auto lt = tb.index({Ellipsis, Slice(0, 2)});
  auto rb = tb.index({Ellipsis, Slice(2, 4)});
  auto tltrb = torch::cat({anc - lt, rb - anc}, -1).clamp(0, m.REG_MAX - 1 - 0.01);
  auto pd_fg = pred_distri.index({fg}).view({-1, 4, m.REG_MAX});
  auto ldfl = dfl_loss(pd_fg, tltrb.index({fg}), m.REG_MAX) * weight;
  auto loss_dfl = ldfl.sum() / tss;

  auto total = m.GBOX * loss_box + m.GCLS * loss_cls + m.GDFL * loss_dfl;
  return {total, loss_box, loss_cls, loss_dfl};
}
