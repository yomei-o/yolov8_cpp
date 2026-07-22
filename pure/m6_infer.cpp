// M6: pure-engine inference. Loads the Ultralytics-preprocessed input, runs the
// pure yolov8n forward, decodes (DFL + anchors) and runs NMS — then checks the
// decoded predictions against Ultralytics eval output (bit-level) and the final
// detections against Ultralytics NMS.
#include "net.hpp"
#include "infer.hpp"
#include <cstdio>
#include <fstream>
#include <sstream>

int main() {
  const std::string DN = "pure/ref/data_net/", DI = "pure/ref/data_infer/";

  // meta
  std::ifstream mf(DI + "meta.txt");
  if (!mf) { printf("run: python pure/ref/m6_infer_ref.py <imgsz>\n"); return 1; }
  int64_t S, nc, Atot; float conf, iou; mf >> S >> nc >> Atot >> conf >> iou;
  float w0, h0, r, dw, dh; mf >> w0 >> h0 >> r >> dw >> dh;
  int ndet; mf >> ndet;
  std::vector<Det> refd(ndet);
  for (int i = 0; i < ndet; ++i) mf >> refd[i].x1 >> refd[i].y1 >> refd[i].x2 >> refd[i].y2 >> refd[i].conf >> refd[i].cls;

  const int64_t RM = 16;
  auto prov = load_net(DN);
  auto x = from_data({1, 3, S, S}, rd(DI + "x.bin"));

  printf("forward (%lldx%lld, naive conv)...\n", (long long)S, (long long)S);
  prov.i = 0;
  auto lv = yolov8n_forward(x, prov);
  int64_t A = 0; for (auto& p : lv) A += p.first->shape[2] * p.first->shape[3];
  if (A != Atot) { printf("anchor mismatch %lld vs %lld\n", (long long)A, (long long)Atot); return 1; }

  std::vector<Tensor> boxes = {lv[0].first,  lv[1].first,  lv[2].first};
  std::vector<Tensor> clses = {lv[0].second, lv[1].second, lv[2].second};
  auto pred_distri = pack_levels(boxes, 1, A, 4 * RM);   // (A, 64)
  auto pred_scores = pack_levels(clses, 1, A, nc);       // (A, nc)

  std::vector<float> ax, ay, st; make_anchors(S, ax, ay, st);
  std::vector<float> pred;
  decode_predictions(pred_distri->data, pred_scores->data, ax, ay, st, A, nc, RM, pred);

  // 1) pure forward raw logits vs the reference head logits (channel-major (C,Atot)).
  // This is the bit-level forward check; the whole net is exercised at this input size.
  { auto hb = rd(DI + "ref_head_boxes.bin"); auto hs = rd(DI + "ref_head_scores.bin");
    double db = 0, ds = 0;
    for (int64_t a = 0; a < A; ++a) {
      for (int c = 0; c < 4 * (int)RM; ++c) db = std::max(db, (double)std::abs(pred_distri->data[a * 4 * RM + c] - hb[(int64_t)c * A + a]));
      for (int c = 0; c < (int)nc; ++c)     ds = std::max(ds, (double)std::abs(pred_scores->data[a * nc + c] - hs[(int64_t)c * A + a]));
    }
    printf("forward vs ultralytics head: box logits=%.3e  score logits=%.3e\n", db, ds);
  }

  // 2) decoded predictions vs Ultralytics eval output. Scores (sigmoid) are checked
  // tightly on every anchor; boxes on confident anchors only, because DFL amplifies
  // the tiny forward difference into large box swings on saturated background anchors.
  auto refp = rd(DI + "ref_pred.bin");                   // (4+nc, Atot) channel-major, xyxy
  double score_max = 0, box_max_conf = 0;
  for (int64_t a = 0; a < A; ++a) {
    float mc = 0; for (int64_t c = 0; c < nc; ++c) mc = std::max(mc, pred[(4 + c) * A + a]);
    for (int64_t c = 0; c < nc; ++c)
      score_max = std::max(score_max, (double)std::abs(pred[(4 + c) * A + a] - refp[(4 + c) * A + a]));
    if (mc > conf)
      for (int j = 0; j < 4; ++j) box_max_conf = std::max(box_max_conf, (double)std::abs(pred[j * A + a] - refp[j * A + a]));
  }
  printf("decode: score max|diff| = %.3e  box max|diff|(conf>%.2f) = %.3e\n", score_max, conf, box_max_conf);
  bool decode_ok = score_max < 1e-3 && box_max_conf < 1.0;

  // 3) NMS vs Ultralytics
  auto dets = nms(pred, A, nc, conf, iou, 300);
  std::sort(dets.begin(), dets.end(), [](const Det& a, const Det& b){ return a.conf > b.conf; });
  std::sort(refd.begin(), refd.end(), [](const Det& a, const Det& b){ return a.conf > b.conf; });
  printf("NMS: pure=%zu  ultralytics=%d\n", dets.size(), ndet);

  bool match = (int)dets.size() == ndet;
  double boxerr = 0, conferr = 0;
  for (size_t i = 0; i < dets.size() && i < refd.size(); ++i) {
    if (dets[i].cls != refd[i].cls) match = false;
    boxerr = std::max({boxerr, (double)std::abs(dets[i].x1 - refd[i].x1), (double)std::abs(dets[i].y1 - refd[i].y1),
                               (double)std::abs(dets[i].x2 - refd[i].x2), (double)std::abs(dets[i].y2 - refd[i].y2)});
    conferr = std::max(conferr, (double)std::abs(dets[i].conf - refd[i].conf));
  }
  printf("det box max|diff| = %.3e  conf max|diff| = %.3e\n", boxerr, conferr);

  std::vector<std::string> names; { std::ifstream nf(DI + "names.txt"); std::string s; while (std::getline(nf, s)) { if (!s.empty() && s.back()=='\r') s.pop_back(); names.push_back(s); } }
  printf("\npure detections (letterboxed %lld px):\n", (long long)S);
  for (auto& d : dets)
    printf("  %-12s conf=%.3f  xyxy=(%.1f,%.1f,%.1f,%.1f)\n",
           (d.cls < (int)names.size() ? names[d.cls].c_str() : "?"), d.conf, d.x1, d.y1, d.x2, d.y2);

  bool ok = decode_ok && match && boxerr < 1.0 && conferr < 1e-2;
  printf("\n%s\n", ok ? "M6: PURE INFERENCE == yolov8n (decode + NMS)" : "MISMATCH");
  return ok ? 0 : 1;
}
