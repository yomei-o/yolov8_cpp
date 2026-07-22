// M4a: DFL decode + CIoU in the pure engine, matched to the reference.
#include "v8pure.hpp"
#include "net.hpp"   // for rd()
#include <cstdio>

int main() {
  const std::string D = "pure/ref/data_m4a/";
  int64_t N, RM; { std::ifstream f(D + "meta.txt"); f >> N >> RM; }
  auto pred = from_data({N, 4 * RM}, rd(D + "pred_dist.bin"), true);
  auto anc = rd(D + "anc.bin"); auto stv = rd(D + "stride.bin"); auto gtv = rd(D + "gt.bin");

  // build constants: anc*stride columns and stride broadcast (N,4)
  std::vector<float> axs(N), ays(N), s4(N * 4);
  for (int64_t i = 0; i < N; ++i) {
    axs[i] = anc[i * 2 + 0] * stv[i]; ays[i] = anc[i * 2 + 1] * stv[i];
    for (int j = 0; j < 4; ++j) s4[i * 4 + j] = stv[i];
  }
  auto anc_sx = from_data({N, 1}, axs), anc_sy = from_data({N, 1}, ays);
  auto stride4 = from_data({N, 4}, s4);
  auto gt = from_data({N, 4}, gtv);

  auto boxes = dfl_decode(pred, anc_sx, anc_sy, stride4, N, RM);
  auto c = ciou_rows(boxes, gt);

  auto rb = rd(D + "ref_boxes.bin"), rc = rd(D + "ref_ciou.bin");
  double db = 0, dc = 0;
  for (int64_t i = 0; i < boxes->numel(); ++i) db = std::max(db, (double)std::abs(boxes->data[i] - rb[i]));
  for (int64_t i = 0; i < c->numel(); ++i) dc = std::max(dc, (double)std::abs(c->data[i] - rc[i]));
  printf("boxes max|diff| = %.3e  %s\n", db, db < 1e-4 ? "OK" : "FAIL");
  printf("ciou  max|diff| = %.3e  %s\n", dc, dc < 1e-4 ? "OK" : "FAIL");
  bool ok = db < 1e-4 && dc < 1e-4;
  printf("\n%s\n", ok ? "M4a: DFL decode + CIoU match" : "MISMATCH");
  return ok ? 0 : 1;
}
