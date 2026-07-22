// Inference postprocess for yolov8: DFL decode + NMS. Plain float (no autograd),
// a faithful port of Ultralytics' Detect._inference + non_max_suppression.
#pragma once
#include <vector>
#include <string>
#include <algorithm>
#include <cmath>
#include <cstdint>

struct Det { float x1, y1, x2, y2, conf; int cls; };

// Anchor points (cell centre = grid + 0.5) and per-anchor stride for the 3 detect
// levels of an SxS input (strides 8/16/32), in the level-major, y-then-x order that
// pack_levels produces. Fills ax, ay (grid units) and st (stride).
inline void make_anchors(int64_t S, std::vector<float>& ax, std::vector<float>& ay,
                         std::vector<float>& st) {
  const int strides[3] = {8, 16, 32};
  for (int l = 0; l < 3; ++l) {
    int g = (int)(S / strides[l]);
    for (int y = 0; y < g; ++y)
      for (int x = 0; x < g; ++x) {
        ax.push_back(x + 0.5f); ay.push_back(y + 0.5f); st.push_back((float)strides[l]);
      }
  }
}

// DFL decode. pred_distri (A, 4*reg_max) logits, pred_scores (A, nc) logits.
// Writes decoded predictions channel-major into out (size (4+nc)*A): rows
// [x1, y1, x2, y2, cls0..] each of length A, boxes as xyxy in pixel units, scores
// sigmoid'd — exactly the (1, 4+nc, A) tensor Ultralytics returns in eval mode
// (this Ultralytics version's Detect head outputs xyxy, not xywh).
inline void decode_predictions(const std::vector<float>& pred_distri,
                               const std::vector<float>& pred_scores,
                               const std::vector<float>& ax, const std::vector<float>& ay,
                               const std::vector<float>& st,
                               int64_t A, int64_t nc, int64_t reg_max,
                               std::vector<float>& out) {
  out.assign((4 + nc) * A, 0.f);
  const int64_t nb = 4 * reg_max;
  std::vector<float> proj(reg_max);
  for (int64_t i = 0; i < reg_max; ++i) proj[i] = (float)i;
  for (int64_t a = 0; a < A; ++a) {
    float d[4];
    for (int j = 0; j < 4; ++j) {
      const float* p = &pred_distri[a * nb + j * reg_max];
      float m = -1e30f; for (int64_t k = 0; k < reg_max; ++k) m = std::max(m, p[k]);
      double s = 0; double acc = 0;
      for (int64_t k = 0; k < reg_max; ++k) { float e = std::exp(p[k] - m); s += e; acc += (double)e * proj[k]; }
      d[j] = (float)(acc / s);                       // expected distance (grid units)
    }
    float x1 = ax[a] - d[0], y1 = ay[a] - d[1], x2 = ax[a] + d[2], y2 = ay[a] + d[3];
    float sc = st[a];
    out[0 * A + a] = x1 * sc;                         // x1
    out[1 * A + a] = y1 * sc;                         // y1
    out[2 * A + a] = x2 * sc;                         // x2
    out[3 * A + a] = y2 * sc;                         // y2
    for (int64_t c = 0; c < nc; ++c)
      out[(4 + c) * A + a] = 1.f / (1.f + std::exp(-pred_scores[a * nc + c]));
  }
}

static inline float iou_xyxy(const Det& a, const Det& b) {
  float iw = std::max(0.f, std::min(a.x2, b.x2) - std::max(a.x1, b.x1));
  float ih = std::max(0.f, std::min(a.y2, b.y2) - std::max(a.y1, b.y1));
  float inter = iw * ih;
  float ua = (a.x2 - a.x1) * (a.y2 - a.y1) + (b.x2 - b.x1) * (b.y2 - b.y1) - inter;
  return ua > 0 ? inter / ua : 0.f;
}

// Non-max suppression matching Ultralytics defaults (single best class per box,
// class-aware via a per-class coordinate offset, greedy by descending confidence).
inline std::vector<Det> nms(const std::vector<float>& pred, int64_t A, int64_t nc,
                            float conf_thres, float iou_thres, int max_det = 300) {
  const float max_wh = 7680.f;
  std::vector<Det> cand;
  for (int64_t a = 0; a < A; ++a) {
    int best = 0; float bconf = pred[4 * A + a];
    for (int64_t c = 1; c < nc; ++c) { float v = pred[(4 + c) * A + a]; if (v > bconf) { bconf = v; best = (int)c; } }
    if (bconf <= conf_thres) continue;
    cand.push_back({pred[0 * A + a], pred[1 * A + a], pred[2 * A + a], pred[3 * A + a], bconf, best});
  }
  std::sort(cand.begin(), cand.end(), [](const Det& p, const Det& q){ return p.conf > q.conf; });
  std::vector<Det> keep;
  std::vector<char> removed(cand.size(), 0);
  for (size_t i = 0; i < cand.size() && (int)keep.size() < max_det; ++i) {
    if (removed[i]) continue;
    keep.push_back(cand[i]);
    Det bi = cand[i]; float off_i = cand[i].cls * max_wh;
    bi.x1 += off_i; bi.x2 += off_i; bi.y1 += off_i; bi.y2 += off_i;   // class offset
    for (size_t j = i + 1; j < cand.size(); ++j) {
      if (removed[j] || cand[j].cls != cand[i].cls) continue;
      Det bj = cand[j]; float off_j = cand[j].cls * max_wh;
      bj.x1 += off_j; bj.x2 += off_j; bj.y1 += off_j; bj.y2 += off_j;
      if (iou_xyxy(bi, bj) > iou_thres) removed[j] = 1;
    }
  }
  return keep;
}
