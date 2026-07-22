// M5a: verify pure-C++ TAL against the torch reference (ref/tal_ref.py output).
#include "tal.hpp"
#include "net.hpp"   // rd()
#include <cstdio>
#include <fstream>

int main() {
  int64_t B,NC,M,A,TOPK; double ALPHA,BETA;
  { std::ifstream f("data/tal_meta.txt"); f>>B>>NC>>M>>A>>TOPK>>ALPHA>>BETA; }

  auto ps = rd("data/t_pd_scores.bin");
  auto pb = rd("data/t_pd_bboxes.bin");
  auto anc = rd("data/t_anc.bin");
  auto gl_f = rd("data/t_gt_labels.bin");
  auto gb = rd("data/t_gt_bboxes.bin");
  auto mg = rd("data/t_mask_gt.bin");
  std::vector<int64_t> gl(gl_f.size()); for (size_t i=0;i<gl_f.size();++i) gl[i]=(int64_t)gl_f[i];

  auto o = tal_assign(ps, pb, anc, gl, gb, mg, B,A,M,NC,TOPK,(float)ALPHA,(float)BETA);

  auto rtb = rd("out/ref_target_bboxes.bin"), rts = rd("out/ref_target_scores.bin"),
       rfg = rd("out/ref_fg_mask.bin");
  double dfg=0,dtb=0,dts=0;
  for (size_t i=0;i<rfg.size();++i) dfg=std::max(dfg,(double)std::abs(o.fg[i]-rfg[i]));
  for (size_t i=0;i<rtb.size();++i) dtb=std::max(dtb,(double)std::abs(o.tb[i]-rtb[i]));
  for (size_t i=0;i<rts.size();++i) dts=std::max(dts,(double)std::abs(o.ts[i]-rts[i]));
  printf("fg     max|diff| = %.3e\n", dfg);
  printf("tboxes max|diff| = %.3e\n", dtb);
  printf("tscore max|diff| = %.3e\n", dts);
  bool ok = dfg==0 && dtb<1e-4 && dts<1e-4;
  printf("\n%s\n", ok ? "M5a: pure TAL == torch" : "MISMATCH");
  return ok ? 0 : 1;
}
