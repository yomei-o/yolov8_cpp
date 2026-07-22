// M5b: end-to-end training of yolov8n in the PURE engine (no LibTorch).
// forward -> pack -> TAL (plain) -> pure v8 loss -> backward -> SGD. Loss must drop.
#include "net.hpp"
#include "v8pure.hpp"
#include "tal.hpp"
#include <cstdio>
#include <random>

int main() {
  const std::string D = "pure/ref/data_net/";
  auto prov = load_net(D);
  std::vector<Tensor> params;
  for (auto& c : prov.convs) { params.push_back(c.w); params.push_back(c.b); }

  const int64_t B = 2, IMG = 64, NC = 80, RM = 16, M = 3, TOPK = 10;
  const float ALPHA = 0.5f, BETA = 6.0f;

  // anchors (grid units) + strides for levels 8/16/32 at imgsz 64 -> A=84
  struct Lv { int64_t h, w; float s; };
  std::vector<Lv> levels = {{8, 8, 8}, {4, 4, 16}, {2, 2, 32}};
  std::vector<float> ancx_a, ancy_a, stride_a, anc_img;
  for (auto& L : levels)
    for (int64_t y = 0; y < L.h; ++y) for (int64_t x = 0; x < L.w; ++x) {
      ancx_a.push_back(x + 0.5f); ancy_a.push_back(y + 0.5f); stride_a.push_back(L.s);
      anc_img.push_back((x + 0.5f) * L.s); anc_img.push_back((y + 0.5f) * L.s);
    }
  int64_t A = (int64_t)stride_a.size(), R = B * A;
  std::vector<float> ancx(R), ancy(R), stride(R);
  for (int64_t r = 0; r < R; ++r) { int64_t a = r % A; ancx[r]=ancx_a[a]; ancy[r]=ancy_a[a]; stride[r]=stride_a[a]; }

  // fixed synthetic batch
  auto img = make_tensor({B, 3, IMG, IMG});
  { std::mt19937 rng(0); std::normal_distribution<float> nd(0,1); for (auto& v: img->data) v=nd(rng); }
  std::vector<float> gb = {8,8,40,44, 30,10,60,40, 10,30,34,60,  4,4,44,52, 28,24,60,60, 6,34,38,60};
  std::vector<int64_t> gl = {12,40,7, 63,3,25};
  std::vector<float> mg(B*M, 1.f);

  float lr = 1e-4f;
  printf("iter |   total     box      cls      dfl\n");
  for (int it = 0; it <= 40; ++it) {
    prov.i = 0;
    auto lvs = yolov8n_forward(img, prov);
    std::vector<Tensor> boxes = {lvs[0].first, lvs[1].first, lvs[2].first};
    std::vector<Tensor> clses = {lvs[0].second, lvs[1].second, lvs[2].second};
    auto pred_distri = pack_levels(boxes, B, A, 4 * RM);   // (R,64)
    auto pred_scores = pack_levels(clses, B, A, NC);        // (R,80)

    // TAL inputs (plain, detached): decode boxes to image units + sigmoid scores
    std::vector<float> pdb(R*4), pss(R*NC);
    for (int64_t r = 0; r < R; ++r) {
      for (int j = 0; j < 4; ++j) {
        float mx=-1e30f; for (int k=0;k<RM;++k) mx=std::max(mx, pred_distri->data[r*64+j*RM+k]);
        double s=0; float d=0; std::vector<float> e(RM);
        for (int k=0;k<RM;++k){ e[k]=std::exp(pred_distri->data[r*64+j*RM+k]-mx); s+=e[k]; }
        for (int k=0;k<RM;++k) d += (float)(e[k]/s)*k;
        pdb[r*4+j]=d;   // distance (grid units), convert below
      }
      float ax=ancx[r], ay=ancy[r], st=stride[r];
      float l=pdb[r*4+0],t=pdb[r*4+1],rr=pdb[r*4+2],bb=pdb[r*4+3];
      pdb[r*4+0]=(ax-l)*st; pdb[r*4+1]=(ay-t)*st; pdb[r*4+2]=(ax+rr)*st; pdb[r*4+3]=(ay+bb)*st;
      for (int64_t c=0;c<NC;++c) pss[r*NC+c]=1.f/(1.f+std::exp(-pred_scores->data[r*NC+c]));
    }
    auto tal = tal_assign(pss, pdb, anc_img, gl, gb, mg, B,A,M,NC,TOPK,ALPHA,BETA);

    auto L = pure_v8_loss(pred_distri, pred_scores, ancx, ancy, stride, tal.tb, tal.ts, R, NC, RM);
    backward(L.total);
    for (auto& p : params) for (int64_t i=0;i<p->numel();++i) p->data[i] -= lr * p->grad[i];

    if (it % 5 == 0)
      printf("%4d | %8.4f %8.4f %8.4f %8.4f\n", it, L.total->data[0],
             L.box->data[0], L.cls->data[0], L.dfl->data[0]);
  }
  printf("done.\n");
  return 0;
}
