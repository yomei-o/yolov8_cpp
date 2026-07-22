// M6 demo: end-to-end yolov8n inference on a REAL image with zero dependencies
// beyond two single-header libraries (stb_image / stb_image_write). Loads a JPEG/PNG,
// letterboxes it, runs the pure forward + DFL decode + NMS, maps boxes back to the
// original image, draws them and writes an annotated PNG.
//
//   build:  cl /std:c++20 /O2 /EHsc pure/m6_demo.cpp        (or g++ -std=c++20 -O2)
//   run:    m6_demo pure/ref/assets/bus.jpg out.png [imgsz]
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"                  // add -Ipure/third_party
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"
#include "net.hpp"
#include "infer.hpp"
#include <cstdio>
#include <string>
#include <algorithm>

static const char* COCO[80] = {
  "person","bicycle","car","motorcycle","airplane","bus","train","truck","boat","traffic light",
  "fire hydrant","stop sign","parking meter","bench","bird","cat","dog","horse","sheep","cow",
  "elephant","bear","zebra","giraffe","backpack","umbrella","handbag","tie","suitcase","frisbee",
  "skis","snowboard","sports ball","kite","baseball bat","baseball glove","skateboard","surfboard","tennis racket","bottle",
  "wine glass","cup","fork","knife","spoon","bowl","banana","apple","sandwich","orange",
  "broccoli","carrot","hot dog","pizza","donut","cake","chair","couch","potted plant","bed",
  "dining table","toilet","tv","laptop","mouse","remote","keyboard","cell phone","microwave","oven",
  "toaster","sink","refrigerator","book","clock","vase","scissors","teddy bear","hair drier","toothbrush"};

struct Color { unsigned char r, g, b; };
static Color class_color(int c) {                       // vivid, well-separated hues
  float h = (float)((c * 47) % 360) / 60.f;             // spread classes around the wheel
  float xf = 1.f - std::abs(std::fmod(h, 2.f) - 1.f);
  float R = 0, G = 0, B = 0;
  if (h < 1) { R = 1; G = xf; } else if (h < 2) { R = xf; G = 1; }
  else if (h < 3) { G = 1; B = xf; } else if (h < 4) { G = xf; B = 1; }
  else if (h < 5) { R = xf; B = 1; } else { R = 1; B = xf; }
  return { (unsigned char)(R * 255), (unsigned char)(G * 255), (unsigned char)(B * 255) };
}

int main(int argc, char** argv) {
  std::string img = argc > 1 ? argv[1] : "assets/bus.jpg";
  std::string outp = argc > 2 ? argv[2] : "out.png";
  int64_t S = argc > 3 ? atoll(argv[3]) : 640;
  std::string wdir = argc > 4 ? argv[4] : "weights/yolov8n/";     // shipped packed weights
  const int64_t NC = 80, RM = 16;
  const float CONF = 0.25f, IOU = 0.7f;

  int w0, h0, ch;
  unsigned char* im = stbi_load(img.c_str(), &w0, &h0, &ch, 3);   // force RGB
  if (!im) { printf("cannot load image %s\n", img.c_str()); return 1; }
  printf("loaded %s  %dx%d\n", img.c_str(), w0, h0);

  // letterbox to SxS (grey 114), preserving aspect ratio (bilinear resize)
  float r = std::min((float)S / w0, (float)S / h0);
  int nw = (int)std::round(w0 * r), nh = (int)std::round(h0 * r);
  int left = (int)((S - nw) / 2), top = (int)((S - nh) / 2);
  auto x = make_tensor({1, 3, S, S});
  for (auto& v : x->data) v = 114.f / 255.f;                     // pad colour
  for (int y = 0; y < nh; ++y)
    for (int xx = 0; xx < nw; ++xx) {
      float sx = (xx + 0.5f) / r - 0.5f, sy = (y + 0.5f) / r - 0.5f;   // bilinear sample
      int x0 = (int)std::floor(sx), y0 = (int)std::floor(sy);
      float fx = sx - x0, fy = sy - y0;
      int x1 = std::min(x0 + 1, w0 - 1), y1 = std::min(y0 + 1, h0 - 1);
      x0 = std::clamp(x0, 0, w0 - 1); y0 = std::clamp(y0, 0, h0 - 1);
      for (int c = 0; c < 3; ++c) {
        float p00 = im[(y0 * w0 + x0) * 3 + c], p01 = im[(y0 * w0 + x1) * 3 + c];
        float p10 = im[(y1 * w0 + x0) * 3 + c], p11 = im[(y1 * w0 + x1) * 3 + c];
        float v = (p00 * (1 - fx) + p01 * fx) * (1 - fy) + (p10 * (1 - fx) + p11 * fx) * fy;
        int oy = top + y, ox = left + xx;
        x->data[(c * S + oy) * S + ox] = v / 255.f;
      }
    }

  printf("forward (%lldx%lld, im2col+GEMM)...\n", (long long)S, (long long)S);
  auto prov = load_net_blob(wdir);
  prov.i = 0;
  auto lv = yolov8n_forward(x, prov);
  int64_t A = 0; for (auto& p : lv) A += p.first->shape[2] * p.first->shape[3];
  std::vector<Tensor> boxes = {lv[0].first, lv[1].first, lv[2].first};
  std::vector<Tensor> clses = {lv[0].second, lv[1].second, lv[2].second};
  auto pd = pack_levels(boxes, 1, A, 4 * RM);
  auto ps = pack_levels(clses, 1, A, NC);

  std::vector<float> ax, ay, st; make_anchors(S, ax, ay, st);
  std::vector<float> pred;
  decode_predictions(pd->data, ps->data, ax, ay, st, A, NC, RM, pred);
  auto dets = nms(pred, A, NC, CONF, IOU, 300);

  // map boxes back to the original image and draw
  auto putpix = [&](int px, int py, Color col) {
    if (px < 0 || py < 0 || px >= w0 || py >= h0) return;
    unsigned char* p = &im[(py * w0 + px) * 3]; p[0] = col.r; p[1] = col.g; p[2] = col.b;
  };
  auto rect = [&](int a, int b, int cc, int d, Color col, int th) {
    for (int t = 0; t < th; ++t) {
      for (int px = a; px <= cc; ++px) { putpix(px, b + t, col); putpix(px, d - t, col); }
      for (int py = b; py <= d; ++py) { putpix(a + t, py, col); putpix(cc - t, py, col); }
    }
  };
  printf("\n%zu detections:\n", dets.size());
  for (auto& dt : dets) {
    int x1 = (int)std::round((dt.x1 - left) / r), y1 = (int)std::round((dt.y1 - top) / r);
    int x2 = (int)std::round((dt.x2 - left) / r), y2 = (int)std::round((dt.y2 - top) / r);
    x1 = std::clamp(x1, 0, w0 - 1); y1 = std::clamp(y1, 0, h0 - 1);
    x2 = std::clamp(x2, 0, w0 - 1); y2 = std::clamp(y2, 0, h0 - 1);
    Color col = class_color(dt.cls);
    rect(x1, y1, x2, y2, col, 3);
    printf("  %-14s conf=%.2f  xyxy=(%d,%d,%d,%d)\n",
           dt.cls < 80 ? COCO[dt.cls] : "?", dt.conf, x1, y1, x2, y2);
  }

  if (!stbi_write_png(outp.c_str(), w0, h0, 3, im, w0 * 3)) { printf("write failed\n"); return 1; }
  printf("\nwrote %s\n", outp.c_str());
  stbi_image_free(im);
  return 0;
}
