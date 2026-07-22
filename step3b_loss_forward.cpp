// Step 3b: full v8 loss forward parity (box/cls/dfl + total).
#include "v8loss.h"

int main() {
  auto m = load_meta("data/loss_meta.txt");
  auto pred_distri = load_t("data/L_pred_distri.bin", {m.B, m.A, 4 * m.REG_MAX});
  auto pred_scores = load_t("data/L_pred_scores.bin", {m.B, m.A, m.NC});
  auto anc         = load_t("data/L_anc.bin", {m.A, 2});
  auto stride      = load_t("data/L_stride.bin", {m.A, 1});
  auto gt_labels   = load_t("data/L_gt_labels.bin", {m.B, m.M, 1}).to(torch::kLong);
  auto gt_bboxes   = load_t("data/L_gt_bboxes.bin", {m.B, m.M, 4});
  auto mask_gt     = load_t("data/L_mask_gt.bin", {m.B, m.M, 1});

  torch::Tensor total, box, cls, dfl;
  std::tie(total, box, cls, dfl) =
      v8_loss(pred_distri, pred_scores, anc, stride, gt_labels, gt_bboxes, mask_gt, m);

  write_bin("out/cpp_losses.bin", torch::stack({box, cls, dfl}));
  write_bin("out/cpp_total.bin", total.view({1}));
  std::cout << "box=" << box.item<float>() << " cls=" << cls.item<float>()
            << " dfl=" << dfl.item<float>() << " total=" << total.item<float>() << "\n";
  return 0;
}
