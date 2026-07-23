// A-1: train yolov8n in the PURE engine with conv+BN kept separate, then write the
// updated weights (conv, BN gamma/beta/running_mean/var) back to flat .bin so a Python
// bridge can drop them into yolov8n.pt. Also dumps an eval-mode forward on a fixed
// input so the round-trip can be checked: Ultralytics loading the new .pt must
// reproduce these boxes/scores. forward(train) -> TAL -> v8 loss -> backward -> SGD.
#include <fstream>
#include "net_unfused.hpp"
#include "v8pure.hpp"
#include "tal.hpp"
#include <cstdio>
#include <random>
#include <filesystem>

int main(int argc, char** argv) {
  const int ITERS = argc > 1 ? atoi(argv[1]) : 20;
  const std::string D = "pure/ref/data_net/", D2 = "pure/ref/data_wb/";
  std::filesystem::create_directories(D2);
  auto prov = load_net_unfused(D);
  std::vector<int64_t> dep; { std::ifstream f(D + "depths.txt"); int64_t v; while (f >> v) dep.push_back(v); }

  // trainable params: conv weights + BN affine (or plain conv bias)
  std::vector<Tensor> params;
  for (auto& L : prov.layers) {
    params.push_back(L.w);
    if (L.kind == 1) { params.push_back(L.gamma); params.push_back(L.beta); }
    else params.push_back(L.b);
  }

  const int64_t B = 2, IMG = 64, NC = 80, RM = 16, M = 3, TOPK = 10;
  const float ALPHA = 0.5f, BETA = 6.0f;
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

  auto img = make_tensor({B, 3, IMG, IMG});
  { std::mt19937 rng(0); std::normal_distribution<float> nd(0,1); for (auto& v: img->data) v=nd(rng); }
  std::vector<float> gb = {8,8,40,44, 30,10,60,40, 10,30,34,60,  4,4,44,52, 28,24,60,60, 6,34,38,60};
  std::vector<int64_t> gl = {12,40,7, 63,3,25};
  std::vector<float> mg(B*M, 1.f);

  float lr = 1e-4f;
  printf("iter |   total     box      cls      dfl\n");
  for (int it = 0; it < ITERS; ++it) {
    prov.i = 0;
    auto lvs = yolov8n_forward_u(img, prov, true, dep);
    std::vector<Tensor> boxes = {lvs[0].first, lvs[1].first, lvs[2].first};
    std::vector<Tensor> clses = {lvs[0].second, lvs[1].second, lvs[2].second};
    auto pred_distri = pack_levels(boxes, B, A, 4 * RM);
    auto pred_scores = pack_levels(clses, B, A, NC);

    std::vector<float> pdb(R*4), pss(R*NC);
    for (int64_t r = 0; r < R; ++r) {
      for (int j = 0; j < 4; ++j) {
        float mx=-1e30f; for (int k=0;k<RM;++k) mx=std::max(mx, pred_distri->data[r*64+j*RM+k]);
        double s=0; float d=0; std::vector<float> e(RM);
        for (int k=0;k<RM;++k){ e[k]=std::exp(pred_distri->data[r*64+j*RM+k]-mx); s+=e[k]; }
        for (int k=0;k<RM;++k) d += (float)(e[k]/s)*k;
        pdb[r*4+j]=d;
      }
      float ax=ancx[r], ay=ancy[r], st=stride[r];
      float l=pdb[r*4+0],t=pdb[r*4+1],rr=pdb[r*4+2],bb=pdb[r*4+3];
      pdb[r*4+0]=(ax-l)*st; pdb[r*4+1]=(ay-t)*st; pdb[r*4+2]=(ax+rr)*st; pdb[r*4+3]=(ay+bb)*st;
      for (int64_t c=0;c<NC;++c) pss[r*NC+c]=1.f/(1.f+std::exp(-pred_scores->data[r*NC+c]));
    }
    auto tal = tal_assign(pss, pdb, anc_img, gl, gb, mg, B,A,M,NC,TOPK,ALPHA,BETA);
    auto Lo = pure_v8_loss(pred_distri, pred_scores, ancx, ancy, stride, tal.tb, tal.ts, R, NC, RM);
    backward(Lo.total);
    for (auto& p : params) for (int64_t i=0;i<p->numel();++i) p->data[i] -= lr * p->grad[i];
    if (it % 5 == 0)
      printf("%4d | %8.4f %8.4f %8.4f %8.4f\n", it, Lo.total->data[0], Lo.box->data[0], Lo.cls->data[0], Lo.dfl->data[0]);
  }
  printf("training done.\n");

  // write updated weights back (canonical order, names match export_unfused.py)
  auto wr = [&](const std::string& n, const std::vector<float>& v) {
    std::ofstream f(D2 + n, std::ios::binary); f.write((const char*)v.data(), v.size() * sizeof(float)); };
  for (size_t i = 0; i < prov.layers.size(); ++i) {
    auto& L = prov.layers[i]; std::string s = std::to_string(i);
    wr("cw" + s + ".bin", L.w->data);
    if (L.kind == 1) { wr("bg"+s+".bin", L.gamma->data); wr("bb"+s+".bin", L.beta->data);
                       wr("rm"+s+".bin", L.rm); wr("rv"+s+".bin", L.rv); }
    else wr("cb" + s + ".bin", L.b->data);
  }

  // eval-mode forward on a fixed input -> the round-trip target for Ultralytics
  auto x = from_data({1, 3, IMG, IMG}, rd(D + "x.bin"));
  prov.i = 0;
  auto lv = yolov8n_forward_u(x, prov, false, dep);
  int64_t Atot = 0; for (auto& p : lv) Atot += p.first->shape[2] * p.first->shape[3];
  int64_t NB = lv[0].first->shape[1], NS = lv[0].second->shape[1];
  std::vector<float> bx(NB*Atot), sc(NS*Atot);
  auto pack = [&](std::vector<float>& dst, bool box){ int64_t off=0; for (auto& p: lv){ const Tensor& t=box?p.first:p.second;
    int64_t C=t->shape[1], hw=t->shape[2]*t->shape[3]; for(int64_t c=0;c<C;++c) for(int64_t a=0;a<hw;++a) dst[c*Atot+off+a]=t->data[c*hw+a]; off+=hw; } };
  pack(bx, true); pack(sc, false);
  wr("cpp_boxes.bin", bx); wr("cpp_scores.bin", sc);
  { std::ofstream f(D2 + "io.txt"); f << IMG << " " << NB << " " << NS << " " << Atot << "\n"; }
  printf("wrote updated weights + eval forward to %s\n", D2.c_str());
  return 0;
}
