// Step 3c: backward parity. Compute grads of total loss wrt pred_distri / pred_scores
// via LibTorch autograd, compare against PyTorch reference grads.
#include "v8loss.h"

int main() {
  auto m = load_meta("data/loss_meta.txt");
  auto pred_distri = load_t("data/L_pred_distri.bin", {m.B, m.A, 4 * m.REG_MAX}).set_requires_grad(true);
  auto pred_scores = load_t("data/L_pred_scores.bin", {m.B, m.A, m.NC}).set_requires_grad(true);
  auto anc         = load_t("data/L_anc.bin", {m.A, 2});
  auto stride      = load_t("data/L_stride.bin", {m.A, 1});
  auto gt_labels   = load_t("data/L_gt_labels.bin", {m.B, m.M, 1}).to(torch::kLong);
  auto gt_bboxes   = load_t("data/L_gt_bboxes.bin", {m.B, m.M, 4});
  auto mask_gt     = load_t("data/L_mask_gt.bin", {m.B, m.M, 1});

  torch::Tensor total, box, cls, dfl;
  std::tie(total, box, cls, dfl) =
      v8_loss(pred_distri, pred_scores, anc, stride, gt_labels, gt_bboxes, mask_gt, m);

  total.backward();

  write_bin("out/cpp_grad_distri.bin", pred_distri.grad());
  write_bin("out/cpp_grad_scores.bin", pred_scores.grad());
  std::cout << "total=" << total.item<float>()
            << "  grad_distri |max|=" << pred_distri.grad().abs().max().item<float>()
            << "  grad_scores |max|=" << pred_scores.grad().abs().max().item<float>() << "\n";
  return 0;
}
