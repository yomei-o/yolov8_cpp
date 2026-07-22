// v8 loss pieces built from pure-engine ops (differentiable via autograd).
#pragma once
#include "ops2d.hpp"

inline Tensor sq(const Tensor& z) { return mul(z, z); }

// DFL decode: pred_dist (N,4*rm) logits -> xyxy boxes (N,4), scaled to image units.
// anc_sx/anc_sy: (N,1) constants = anchor*stride; stride4: (N,4) constant = stride broadcast.
inline Tensor dfl_decode(const Tensor& pred_dist, const Tensor& anc_sx, const Tensor& anc_sy,
                         const Tensor& stride4, int64_t N, int64_t reg_max) {
  std::vector<float> proj(reg_max); for (int64_t i = 0; i < reg_max; ++i) proj[i] = (float)i;
  auto sm = softmax_rows(reshape(pred_dist, {N * 4, reg_max}));
  auto d = reshape(wsum_rows(sm, proj), {N, 4});      // ltrb distances
  auto ds = mul(d, stride4);
  auto l = narrow_col(ds, 0), t = narrow_col(ds, 1), r = narrow_col(ds, 2), b = narrow_col(ds, 3);
  return concat_cols({sub(anc_sx, l), sub(anc_sy, t), add(anc_sx, r), add(anc_sy, b)});
}

// CIoU between paired boxes (N,4)x(N,4) -> (N,1). alpha detached (as Ultralytics).
inline Tensor ciou_rows(const Tensor& b1, const Tensor& b2, float eps = 1e-7f) {
  auto x1 = narrow_col(b1, 0), y1 = narrow_col(b1, 1), x2 = narrow_col(b1, 2), y2 = narrow_col(b1, 3);
  auto X1 = narrow_col(b2, 0), Y1 = narrow_col(b2, 1), X2 = narrow_col(b2, 2), Y2 = narrow_col(b2, 3);
  auto w1 = sub(x2, x1), h1 = sub(y2, y1), w2 = sub(X2, X1), h2 = sub(Y2, Y1);
  auto inter = mul(clampmin_scalar(sub(min2(x2, X2), max2(x1, X1)), 0.f),
                   clampmin_scalar(sub(min2(y2, Y2), max2(y1, Y1)), 0.f));
  auto uni = add_scalar(sub(add(mul(w1, h1), mul(w2, h2)), inter), eps);
  auto iou = divi(inter, uni);
  auto cw = sub(max2(x2, X2), min2(x1, X1)), ch = sub(max2(y2, Y2), min2(y1, Y1));
  auto c2 = add_scalar(add(mul(cw, cw), mul(ch, ch)), eps);
  auto rho2 = mul_scalar(add(sq(sub(add(x1, x2), add(X1, X2))), sq(sub(add(y1, y2), add(Y1, Y2)))), 0.25f);
  float pi = (float)std::acos(-1.0);
  auto v = mul_scalar(sq(sub(op_atan(divi(w2, add_scalar(h2, eps))),
                            op_atan(divi(w1, add_scalar(h1, eps))))), 4.f / (pi * pi));
  auto alpha = detach(divi(v, add_scalar(sub(v, iou), 1.f + eps)));
  return sub(iou, add(divi(rho2, c2), mul(v, alpha)));
}

// DFL loss per row: pred_distri (R,4*rm), target_ltrb data (R*4) -> (R,1).
inline Tensor dfl_loss_pure(const Tensor& pred_distri, const std::vector<float>& tltrb,
                            int64_t R, int64_t rm) {
  auto logp = log_softmax_rows(reshape(pred_distri, {R * 4, rm}));   // (R*4, rm)
  std::vector<int64_t> tl(R * 4), tr(R * 4);
  std::vector<float> wl(R * 4), wr(R * 4);
  for (int64_t i = 0; i < R * 4; ++i) {
    float t = std::min(std::max(tltrb[i], 0.f), (float)(rm - 1) - 0.01f);
    int64_t li = (int64_t)t; tl[i] = li; tr[i] = li + 1;
    wl[i] = (li + 1) - t; wr[i] = 1.f - wl[i];
  }
  auto cel = gather_row(logp, tl), cer = gather_row(logp, tr);       // (R*4,1) = log p
  auto wlt = from_data({R * 4, 1}, wl), wrt = from_data({R * 4, 1}, wr);
  auto perside = add(mul(mul_scalar(cel, -1.f), wlt), mul(mul_scalar(cer, -1.f), wrt));
  return wsum_rows(reshape(perside, {R, 4}), {0.25f, 0.25f, 0.25f, 0.25f});
}

// BCE-with-logits (stable): max(x,0) - x*z + log1p(exp(-|x|)). z constant.
inline Tensor bce_logits(const Tensor& x, const Tensor& z) {
  auto t3 = op_log1p(op_exp(mul_scalar(op_abs(x), -1.f)));
  return add(sub(clampmin_scalar(x, 0.f), mul(x, z)), t3);
}

// Full v8 loss given precomputed TAL targets. anc in grid units (per row).
struct LossOut { Tensor total, box, cls, dfl; };
inline LossOut pure_v8_loss(const Tensor& pred_distri, const Tensor& pred_scores,
                            const std::vector<float>& ancx, const std::vector<float>& ancy,
                            const std::vector<float>& stride, const std::vector<float>& tb_img,
                            const std::vector<float>& ts, int64_t R, int64_t nc, int64_t rm) {
  auto AX = from_data({R, 1}, ancx), AY = from_data({R, 1}, ancy);
  auto ones4 = from_data({R, 4}, std::vector<float>(R * 4, 1.f));
  auto boxes_grid = dfl_decode(pred_distri, AX, AY, ones4, R, rm);

  std::vector<float> tbg(R * 4), tltrb(R * 4), wdat(R, 0.f);
  double tss_d = 0;
  for (int64_t r = 0; r < R; ++r) {
    for (int j = 0; j < 4; ++j) tbg[r * 4 + j] = tb_img[r * 4 + j] / stride[r];
    // bbox2dist(anc_grid, tb_grid) -> ltrb, clamped
    float l = ancx[r] - tbg[r * 4 + 0], t = ancy[r] - tbg[r * 4 + 1];
    float rr = tbg[r * 4 + 2] - ancx[r], bb = tbg[r * 4 + 3] - ancy[r];
    float hi = (float)(rm - 1) - 0.01f;
    tltrb[r * 4 + 0] = std::min(std::max(l, 0.f), hi);
    tltrb[r * 4 + 1] = std::min(std::max(t, 0.f), hi);
    tltrb[r * 4 + 2] = std::min(std::max(rr, 0.f), hi);
    tltrb[r * 4 + 3] = std::min(std::max(bb, 0.f), hi);
    for (int64_t c = 0; c < nc; ++c) { wdat[r] += ts[r * nc + c]; tss_d += ts[r * nc + c]; }
  }
  float inv_tss = 1.f / (float)std::max(1.0, tss_d);
  auto weight = from_data({R, 1}, wdat);
  auto tbg_t = from_data({R, 4}, tbg);

  auto ciou = ciou_rows(boxes_grid, tbg_t);                       // (R,1)
  auto box = mul_scalar(sum(mul(add_scalar(mul_scalar(ciou, -1.f), 1.f), weight)), inv_tss);

  auto ts_t = from_data({R, nc}, ts);
  auto cls = mul_scalar(sum(bce_logits(pred_scores, ts_t)), inv_tss);

  auto ldfl = dfl_loss_pure(pred_distri, tltrb, R, rm);           // (R,1)
  auto dfl = mul_scalar(sum(mul(ldfl, weight)), inv_tss);

  auto total = add(add(mul_scalar(box, 7.5f), mul_scalar(cls, 0.5f)), mul_scalar(dfl, 1.5f));
  return {total, box, cls, dfl};
}
