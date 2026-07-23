// yolov8 training CLI — a real training loop, pure C++ / no Python at run time:
// dataset scan -> shuffled mini-batches (+ hflip/brightness aug) over epochs -> TAL ->
// v8 loss -> Adam (warmup + cosine) -> per-epoch validation mAP -> save last.pt / best.pt
// (via the pure-C++ .pt writer). Initial weights + state_dict key names come from a .pt
// read entirely in C++.
//   build: cl /std:c++20 /O2 /EHsc /Ipure\third_party pure/train_cli.cpp
//   run:   train_cli <train_list> <val_list> <epochs> <batch> <init.pt>
#define STB_IMAGE_IMPLEMENTATION
#include "dataset.hpp"
#include "net_unfused.hpp"
#include "v8pure.hpp"
#include "tal.hpp"
#include "optim.hpp"
#include "infer.hpp"
#include "metrics.hpp"
#include "ptio.hpp"
#include <cstdio>
#include <numeric>
#include <algorithm>
#include <random>

static const int64_t NC = 80, RM = 16;

// decode the head predictions (grid units) to image-unit xyxy for TAL, + sigmoid scores
static void decode_for_tal(const Tensor& pd, const Tensor& ps, const std::vector<float>& ax,
                           const std::vector<float>& ay, const std::vector<float>& ss,
                           int64_t R, int64_t A, std::vector<float>& pdb, std::vector<float>& pss) {
  pdb.assign(R*4, 0); pss.assign(R*NC, 0);
  for (int64_t r = 0; r < R; ++r) {
    for (int j = 0; j < 4; ++j) { float mx=-1e30f; for (int k=0;k<RM;++k) mx=std::max(mx, pd->data[r*64+j*RM+k]);
      double s=0; float d=0; std::vector<float> e(RM); for (int k=0;k<RM;++k){e[k]=std::exp(pd->data[r*64+j*RM+k]-mx);s+=e[k];}
      for (int k=0;k<RM;++k) d+=(float)(e[k]/s)*k; pdb[r*4+j]=d; }
    float axr=ax[r%A], ayr=ay[r%A], st=ss[r%A]; float l=pdb[r*4],t=pdb[r*4+1],rr=pdb[r*4+2],bb=pdb[r*4+3];
    pdb[r*4]=(axr-l)*st; pdb[r*4+1]=(ayr-t)*st; pdb[r*4+2]=(axr+rr)*st; pdb[r*4+3]=(ayr+bb)*st;
    for (int64_t c=0;c<NC;++c) pss[r*NC+c]=1.f/(1.f+std::exp(-ps->data[r*NC+c]));
  }
}

int main(int argc, char** argv) {
  std::string trainL = argc>1?argv[1]:"pure/ref/data_synth/list.txt";
  std::string valL   = argc>2?argv[2]:"pure/ref/data_synth/val.txt";
  int EPOCHS = argc>3?atoi(argv[3]):8, BATCH = argc>4?atoi(argv[4]):4;
  std::string initpt = argc>5?argv[5]:"yolov8n.pt";
  int imgsz = argc>6?atoi(argv[6]):0;                 // >0 => standard-YOLO dataset (dir/list + normalised labels)
  bool mosaic = argc>7?atoi(argv[7])!=0:(imgsz>0);    // mosaic on by default in YOLO mode
  const std::string DN = "pure/ref/data_net/";

  Dataset tr, va;
  if (imgsz>0) { tr = read_yolo_dataset(trainL, imgsz); va = read_yolo_dataset(valL, imgsz); }
  else { tr = read_dataset(trainL); va = read_dataset(valL); }
  int64_t S = tr.S;
  printf("train=%zu val=%zu imgsz=%lld batch=%d epochs=%d fmt=%s mosaic=%d\n",
         tr.items.size(), va.items.size(), (long long)S, BATCH, EPOCHS, tr.yolo?"yolo":"legacy", (int)mosaic);

  // initial weights: from the init .pt (pure-C++ read, arch from the tiny manifest) if it
  // exists, else from the Python-exported .bin files.
  ProviderU prov; { std::ifstream t(initpt); if (t.good()) { printf("init weights <- %s (pure C++)\n", initpt.c_str()); prov = load_net_unfused_pt(DN, initpt); } else prov = load_net_unfused(DN); }
  std::vector<int64_t> dep; { std::ifstream f(DN + "depths.txt"); int64_t v; while (f >> v) dep.push_back(v); }
  std::vector<Tensor> params; for (auto& L : prov.layers) { params.push_back(L.w); if (L.kind==1){params.push_back(L.gamma);params.push_back(L.beta);} else params.push_back(L.b); }
  Adam opt(params, 2e-3f, 0.9f, 0.999f, 1e-8f, 5e-4f, false);

  // anchors for imgsz S (strides 8/16/32)
  struct Lv { int64_t h,w; float s; }; std::vector<Lv> lv = {{S/8,S/8,8.f},{S/16,S/16,16.f},{S/32,S/32,32.f}};
  std::vector<float> ax,ay,ss,anc_img; for (auto& L:lv) for (int64_t y=0;y<L.h;++y) for (int64_t x=0;x<L.w;++x){ax.push_back(x+.5f);ay.push_back(y+.5f);ss.push_back(L.s);anc_img.push_back((x+.5f)*L.s);anc_img.push_back((y+.5f)*L.s);}
  int64_t A = ss.size();
  std::vector<float> axv,ayv,sv; make_anchors(S, axv, ayv, sv);   // for inference decode

  // state_dict KEY per engine tensor (engine/C2f emit order != state_dict order, so pair
  // by name — load_state_dict matches by key). names.txt is emitted by export_unfused.py.
  std::vector<std::string> names; { std::ifstream f(DN + "names.txt"); std::string s; while (f >> s) names.push_back(s); }
  auto save_ckpt = [&](const std::string& path) {
    std::vector<pt::Tensor> ck; size_t k = 0;
    auto push = [&](const std::vector<float>& d, const std::vector<int64_t>& shp){ pt::Tensor t; if (k<names.size()) t.name=names[k]; t.shape=shp; t.data=d; ck.push_back(t); ++k; };
    for (auto& L : prov.layers) { std::vector<int64_t> ws(L.w->shape.begin(),L.w->shape.end());
      push(L.w->data, ws);
      if (L.kind==1){ std::vector<int64_t> c={L.gamma->shape[0]}; push(L.gamma->data,c); push(L.beta->data,c); push(L.rm,c); push(L.rv,c); }
      else push(L.b->data, {L.b->shape[0]}); }
    pt::save_pt(ck, path);
  };

  auto validate = [&]() -> double {
    std::vector<mapeval::Image> imgs;
    for (auto& s : va.items) {
      Letterbox lb; auto xi = load_image_letterbox(s.img, S, lb);   // no aug/mosaic at eval
      prov.i=0; auto h = yolov8n_forward_u(xi, prov, false, dep);
      std::vector<Tensor> bx={h[0].first,h[1].first,h[2].first}, cs={h[0].second,h[1].second,h[2].second};
      auto pd=pack_levels(bx,1,A,4*RM); auto ps=pack_levels(cs,1,A,NC);
      std::vector<float> pred; decode_predictions(pd->data, ps->data, axv,ayv,sv, A, NC, RM, pred);
      auto dets = nms(pred, A, NC, 0.25f, 0.5f, 100);
      mapeval::Image im;
      for (auto& d : dets) im.dts.push_back({d.x1,d.y1,d.x2,d.y2,d.cls,d.conf});
      std::vector<float> gb; std::vector<int64_t> gl; int m = load_boxes_orig(s.lbl, va.yolo, lb.w0, lb.h0, gb, gl); lb_map(gb, lb);
      for (int j=0;j<m;++j) im.gts.push_back({gb[j*4],gb[j*4+1],gb[j*4+2],gb[j*4+3],(int)gl[j]});
      imgs.push_back(im);
    }
    return mapeval::coco_map(imgs).first;   // mAP@0.50
  };

  std::vector<int> order(tr.items.size()); std::iota(order.begin(), order.end(), 0);
  std::mt19937 rng(0);
  int steps_per_epoch = ((int)tr.items.size() + BATCH - 1) / BATCH, total = EPOCHS * steps_per_epoch, gstep = 0;
  double best = -1;
  for (int ep = 0; ep < EPOCHS; ++ep) {
    std::shuffle(order.begin(), order.end(), rng); double eloss = 0; int nb = 0;
    for (size_t off = 0; off < order.size(); off += BATCH) {
      std::vector<int> idx(order.begin()+off, order.begin()+std::min(order.size(), off+BATCH));
      Batch bt = load_minibatch(tr, idx, true, rng(), mosaic);
      int64_t B = bt.B, R = B*A, Mx = bt.M;
      std::vector<float> ancx(R),ancy(R),strd(R); for (int64_t r=0;r<R;++r){int64_t a=r%A;ancx[r]=ax[a];ancy[r]=ay[a];strd[r]=ss[a];}
      prov.i=0; auto h = yolov8n_forward_u(bt.x, prov, true, dep);
      std::vector<Tensor> bx={h[0].first,h[1].first,h[2].first}, cs={h[0].second,h[1].second,h[2].second};
      auto pd=pack_levels(bx,B,A,4*RM); auto ps=pack_levels(cs,B,A,NC);
      std::vector<float> pdb,pss; decode_for_tal(pd,ps,ax,ay,ss,R,A,pdb,pss);
      auto tal = tal_assign(pss,pdb,anc_img, bt.gt_labels, bt.gt_boxes, bt.mask, B,A,Mx,NC,10,0.5f,6.0f);
      auto Lo = pure_v8_loss(pd,ps,ancx,ancy,strd,tal.tb,tal.ts,R,NC,RM);
      backward(Lo.total);
      opt.lr = cosine_lr(gstep, total, 2e-3f, std::max(1, total/20)); opt.step(); ++gstep;
      eloss += Lo.total->data[0]; ++nb;
    }
    double m50 = validate();
    printf("epoch %2d/%d  loss %6.3f  val mAP@0.5 %.3f%s\n", ep+1, EPOCHS, eloss/nb, m50, m50>best?"  *best*":"");
    save_ckpt("last.pt"); if (m50 > best) { best = m50; save_ckpt("best.pt"); }
  }
  printf("done. best val mAP@0.5 = %.3f. wrote last.pt / best.pt (pure C++)\n", best);
  return 0;
}
