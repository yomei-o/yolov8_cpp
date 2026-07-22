// M4b + M4c: full v8 loss forward AND backward in the pure engine, compared to
// the torch reference (same inputs/targets). TAL targets are injected as constants.
#include "v8pure.hpp"
#include "net.hpp"     // rd()
#include <cstdio>

int main() {
  const std::string D = "pure/ref/data_m4/";
  int64_t B, A, NC, RM; { std::ifstream f(D + "meta.txt"); f >> B >> A >> NC >> RM; }
  int64_t R = B * A;

  auto pd = from_data({R, 4 * RM}, rd(D + "pred_distri.bin"), true);
  auto ps = from_data({R, NC}, rd(D + "pred_scores.bin"), true);
  auto anc = rd(D + "anc.bin"); auto stv = rd(D + "stride.bin");
  auto tb = rd(D + "tb.bin"); auto ts = rd(D + "ts.bin");

  std::vector<float> ancx(R), ancy(R), stride(R);
  for (int64_t r = 0; r < R; ++r) {
    int64_t a = r % A;
    ancx[r] = anc[a * 2 + 0]; ancy[r] = anc[a * 2 + 1]; stride[r] = stv[a];
  }

  auto L = pure_v8_loss(pd, ps, ancx, ancy, stride, tb, ts, R, NC, RM);

  // ---- M4b: forward ----
  auto rl = rd(D + "ref_losses.bin"), rt = rd(D + "ref_total.bin");
  float box = L.box->data[0], cls = L.cls->data[0], dfl = L.dfl->data[0], tot = L.total->data[0];
  printf("forward  box=%.6f cls=%.6f dfl=%.6f total=%.6f\n", box, cls, dfl, tot);
  printf("ref      box=%.6f cls=%.6f dfl=%.6f total=%.6f\n", rl[0], rl[1], rl[2], rt[0]);
  double df = std::max({std::abs(box - rl[0]), std::abs(cls - rl[1]),
                        std::abs(dfl - rl[2]), std::abs(tot - rt[0])});
  printf("[M4b forward] max|diff| = %.3e  %s\n\n", df, df < 1e-3 ? "OK" : "FAIL");

  // ---- M4c: backward ----
  backward(L.total);
  auto gd = rd(D + "ref_grad_distri.bin"), gs = rd(D + "ref_grad_scores.bin");
  double dgd = 0, dgs = 0;
  for (int64_t i = 0; i < pd->numel(); ++i) dgd = std::max(dgd, (double)std::abs(pd->grad[i] - gd[i]));
  for (int64_t i = 0; i < ps->numel(); ++i) dgs = std::max(dgs, (double)std::abs(ps->grad[i] - gs[i]));
  printf("[M4c grad_distri] max|diff| = %.3e  (scale %.2e)\n", dgd,
         (double)*std::max_element(gd.begin(), gd.end()));
  printf("[M4c grad_scores] max|diff| = %.3e  (scale %.2e)\n", dgs,
         (double)*std::max_element(gs.begin(), gs.end()));

  bool ok = df < 1e-3 && dgd < 1e-5 && dgs < 1e-5;
  printf("\n%s\n", ok ? "M4: PURE LOSS + AUTOGRAD == torch" : "MISMATCH");
  return ok ? 0 : 1;
}
