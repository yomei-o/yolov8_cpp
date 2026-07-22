// Real data loader for training: load an image (stb_image), letterbox it to SxS, and
// load YOLO labels into ground-truth boxes/labels in that letterboxed image space
// (the units TAL expects). The including TU must define STB_IMAGE_IMPLEMENTATION once.
#pragma once
#include "autograd.hpp"
#include "stb_image.h"                  // add -Ipure/third_party
#include <string>
#include <fstream>
#include <vector>
#include <cmath>
#include <algorithm>

struct Letterbox { float r; int left, top, w0, h0; };

// Load + letterbox to SxS (grey 114, bilinear), RGB, /255 -> (1,3,S,S). Fills lb.
inline Tensor load_image_letterbox(const std::string& path, int64_t S, Letterbox& lb) {
  int w0, h0, ch;
  unsigned char* im = stbi_load(path.c_str(), &w0, &h0, &ch, 3);
  if (!im) { printf("cannot load %s\n", path.c_str()); std::exit(1); }
  float r = std::min((float)S / w0, (float)S / h0);
  int nw = (int)std::round(w0 * r), nh = (int)std::round(h0 * r);
  int left = (int)((S - nw) / 2), top = (int)((S - nh) / 2);
  auto x = make_tensor({1, 3, S, S});
  for (auto& v : x->data) v = 114.f / 255.f;
  for (int y = 0; y < nh; ++y)
    for (int xx = 0; xx < nw; ++xx) {
      float sx = (xx + 0.5f) / r - 0.5f, sy = (y + 0.5f) / r - 0.5f;
      int x0 = (int)std::floor(sx), y0 = (int)std::floor(sy);
      float fx = sx - x0, fy = sy - y0;
      int x1 = std::min(x0 + 1, w0 - 1), y1 = std::min(y0 + 1, h0 - 1);
      x0 = std::clamp(x0, 0, w0 - 1); y0 = std::clamp(y0, 0, h0 - 1);
      for (int c = 0; c < 3; ++c) {
        float p00 = im[(y0 * w0 + x0) * 3 + c], p01 = im[(y0 * w0 + x1) * 3 + c];
        float p10 = im[(y1 * w0 + x0) * 3 + c], p11 = im[(y1 * w0 + x1) * 3 + c];
        float v = (p00 * (1 - fx) + p01 * fx) * (1 - fy) + (p10 * (1 - fx) + p11 * fx) * fy;
        x->data[(c * S + top + y) * S + left + xx] = v / 255.f;
      }
    }
  stbi_image_free(im);
  lb = {r, left, top, w0, h0};
  return x;
}

// Labels file: one "cls x1 y1 x2 y2" per line, xyxy already in letterboxed image units.
// Returns M; fills gt_boxes (M*4) and gt_labels (M).
inline int load_labels(const std::string& path, std::vector<float>& gt_boxes, std::vector<int64_t>& gt_labels) {
  std::ifstream f(path);
  if (!f) { printf("cannot open %s\n", path.c_str()); std::exit(1); }
  gt_boxes.clear(); gt_labels.clear();
  int cls; float x1, y1, x2, y2; int M = 0;
  while (f >> cls >> x1 >> y1 >> x2 >> y2) {
    gt_labels.push_back(cls);
    gt_boxes.insert(gt_boxes.end(), {x1, y1, x2, y2});
    ++M;
  }
  return M;
}
