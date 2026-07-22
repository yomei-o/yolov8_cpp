// E-1: end-to-end training on REAL images with REAL labels, in the pure engine, with
// mini-batches. stb image load + letterbox + label load -> unfused conv+BN forward
// (BN in train mode) -> TAL -> v8 loss -> backward -> Adam (cosine LR). Loss must drop.
//   build: cl /std:c++20 /O2 /EHsc /Ipure\third_party pure/m14_train_real.cpp
//   run:   m14_train_real [iters]      (needs export_unfused.py + make_labels.py)
#define STB_IMAGE_IMPLEMENTATION        // dataset.hpp includes stb_image.h (add -Ipure/third_party)
#include "dataset.hpp"
#include "net_unfused.hpp"
#include "v8pure.hpp"
#include "tal.hpp"
#include "optim.hpp"
#include <cstdio>

int main(int argc, char** argv) {
  const int ITERS = argc > 1 ? atoi(argv[1]) : 20;
  const std::string DN = "pure/ref/data_net/", DT = "pure/ref/data_train/";
  const int64_t NC = 80, RM = 16, TOPK = 10;
  const float ALPHA = 0.5f, BETA = 6.0f;

  // real mini-batch (images + labels)
  Batch bt = load_batch(DT + "list.txt");
  int64_t B = bt.B, M = bt.M, S = bt.x->shape[2];
  printf("batch: %lld images, imgsz %lld, up to %lld labels each\n", (long long)B, (long long)S, (long long)M);

  // net (unfused, trainable) + Adam
  auto prov = load_net_unfused(DN);
  std::vector<Tensor> params;
  for (auto& L : prov.layers) { params.push_back(L.w); if (L.kind == 1) { params.push_back(L.gamma); params.push_back(L.beta); } else params.push_back(L.b); }
  Adam opt(params, 1e-3f, 0.9f, 0.999f, 1e-8f, 0.f, false);

  // anchors for imgsz S (levels stride 8/16/32); anc_img (A,2) for TAL, and per-row
  // ancx/ancy/stride replicated to R = B*A for the loss.
  struct Lv { int64_t h, w; float s; };
  std::vector<Lv> levels = {{S/8,S/8,8.f}, {S/16,S/16,16.f}, {S/32,S/32,32.f}};
  std::vector<float> ax, ay, ss, anc_img;
  for (auto& L : levels)
    for (int64_t y = 0; y < L.h; ++y) for (int64_t x = 0; x < L.w; ++x) {
      ax.push_back(x + 0.5f); ay.push_back(y + 0.5f); ss.push_back(L.s);
      anc_img.push_back((x + 0.5f) * L.s); anc_img.push_back((y + 0.5f) * L.s);
    }
  int64_t A = (int64_t)ss.size(), R = B * A;
  std::vector<float> ancx(R), ancy(R), stride(R);
  for (int64_t r = 0; r < R; ++r) { int64_t a = r % A; ancx[r]=ax[a]; ancy[r]=ay[a]; stride[r]=ss[a]; }

  printf("iter |   total     box      cls      dfl      lr\n");
  for (int it = 0; it < ITERS; ++it) {
    prov.i = 0;
    auto lvs = yolov8n_forward_u(bt.x, prov, /*training=*/true);
    std::vector<Tensor> boxes = {lvs[0].first, lvs[1].first, lvs[2].first};
    std::vector<Tensor> clses = {lvs[0].second, lvs[1].second, lvs[2].second};
    auto pred_distri = pack_levels(boxes, B, A, 4 * RM);
    auto pred_scores = pack_levels(clses, B, A, NC);

    // TAL inputs (plain, detached): decode boxes to image units + sigmoid scores
    std::vector<float> pdb(R*4), pss(R*NC);
    for (int64_t r = 0; r < R; ++r) {
      int64_t a = r % A;
      for (int j = 0; j < 4; ++j) {
        float mx=-1e30f; for (int k=0;k<RM;++k) mx=std::max(mx, pred_distri->data[r*64+j*RM+k]);
        double s=0; float d=0; std::vector<float> e(RM);
        for (int k=0;k<RM;++k){ e[k]=std::exp(pred_distri->data[r*64+j*RM+k]-mx); s+=e[k]; }
        for (int k=0;k<RM;++k) d += (float)(e[k]/s)*k;
        pdb[r*4+j]=d;
      }
      float axr=ax[a], ayr=ay[a], st=ss[a];
      float l=pdb[r*4+0],t=pdb[r*4+1],rr=pdb[r*4+2],bb=pdb[r*4+3];
      pdb[r*4+0]=(axr-l)*st; pdb[r*4+1]=(ayr-t)*st; pdb[r*4+2]=(axr+rr)*st; pdb[r*4+3]=(ayr+bb)*st;
      for (int64_t c=0;c<NC;++c) pss[r*NC+c]=1.f/(1.f+std::exp(-pred_scores->data[r*NC+c]));
    }
    auto tal = tal_assign(pss, pdb, anc_img, bt.gt_labels, bt.gt_boxes, bt.mask, B, A, M, NC, TOPK, ALPHA, BETA);
    auto Lo = pure_v8_loss(pred_distri, pred_scores, ancx, ancy, stride, tal.tb, tal.ts, R, NC, RM);
    backward(Lo.total);
    opt.lr = cosine_lr(it, ITERS, 1e-3f, /*warmup=*/3);
    opt.step();
    if (it % 5 == 0 || it == ITERS - 1)
      printf("%4d | %8.4f %8.4f %8.4f %8.4f  %.2e\n", it, Lo.total->data[0],
             Lo.box->data[0], Lo.cls->data[0], Lo.dfl->data[0], opt.lr);
  }
  printf("done — batched training on %lld real images (weights -> .pt via m9's write-back).\n", (long long)B);
  return 0;
}
