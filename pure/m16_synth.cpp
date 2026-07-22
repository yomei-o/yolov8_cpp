// End-to-end test: train yolov8n on a small synthetic dataset (real image files +
// labels) then run inference on a held-out image and draw the detections.
//   build: cl /std:c++20 /O2 /EHsc /Ipure\third_party pure/m16_synth.cpp
//   run:   m16_synth [iters]     (needs make_synth.py + export_unfused.py)
#define STB_IMAGE_IMPLEMENTATION        // dataset.hpp includes stb_image.h (add -Ipure/third_party)
#include "dataset.hpp"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"
#include "net_unfused.hpp"
#include "v8pure.hpp"
#include "tal.hpp"
#include "optim.hpp"
#include "infer.hpp"
#include <cstdio>
#include <string>
#include <filesystem>

static const char* NAMES[3] = {"red", "green", "blue"};

int main(int argc, char** argv) {
  const int ITERS = argc > 1 ? atoi(argv[1]) : 150;
  const std::string DN = "pure/ref/data_net/", DS = "pure/ref/data_synth/";
  const int64_t NC = 80, RM = 16, M_MAX_TOPK = 10;
  const float ALPHA = 0.5f, BETA = 6.0f;

  Batch bt = load_batch(DS + "list.txt");
  int64_t B = bt.B, M = bt.M, S = bt.x->shape[2];
  printf("train batch: %lld images %lldpx, up to %lld objects each\n", (long long)B, (long long)S, (long long)M);

  auto prov = load_net_unfused(DN);
  std::vector<Tensor> params;
  for (auto& L : prov.layers) { params.push_back(L.w); if (L.kind == 1) { params.push_back(L.gamma); params.push_back(L.beta); } else params.push_back(L.b); }
  Adam opt(params, 2e-3f, 0.9f, 0.999f, 1e-8f, 0.f, false);

  struct Lv { int64_t h, w; float s; };
  std::vector<Lv> levels = {{S/8,S/8,8.f}, {S/16,S/16,16.f}, {S/32,S/32,32.f}};
  std::vector<float> ax, ay, ss, anc_img;
  for (auto& L : levels) for (int64_t y=0;y<L.h;++y) for (int64_t x=0;x<L.w;++x){ ax.push_back(x+.5f);ay.push_back(y+.5f);ss.push_back(L.s);anc_img.push_back((x+.5f)*L.s);anc_img.push_back((y+.5f)*L.s);}
  int64_t A = ss.size(), R = B * A;
  std::vector<float> ancx(R),ancy(R),stride(R);
  for (int64_t r=0;r<R;++r){int64_t a=r%A;ancx[r]=ax[a];ancy[r]=ay[a];stride[r]=ss[a];}

  printf("training %d iters...\n", ITERS);
  for (int it = 0; it < ITERS; ++it) {
    prov.i = 0;
    auto lvs = yolov8n_forward_u(bt.x, prov, true);
    std::vector<Tensor> boxes={lvs[0].first,lvs[1].first,lvs[2].first}, clses={lvs[0].second,lvs[1].second,lvs[2].second};
    auto pd = pack_levels(boxes, B, A, 4*RM); auto ps = pack_levels(clses, B, A, NC);
    std::vector<float> pdb(R*4), pss(R*NC);
    for (int64_t r=0;r<R;++r){ for(int j=0;j<4;++j){float mx=-1e30f;for(int k=0;k<RM;++k)mx=std::max(mx,pd->data[r*64+j*RM+k]);double s=0;float d=0;std::vector<float>e(RM);for(int k=0;k<RM;++k){e[k]=std::exp(pd->data[r*64+j*RM+k]-mx);s+=e[k];}for(int k=0;k<RM;++k)d+=(float)(e[k]/s)*k;pdb[r*4+j]=d;}
      float axr=ax[r%A],ayr=ay[r%A],st=ss[r%A];float l=pdb[r*4],t=pdb[r*4+1],rr=pdb[r*4+2],bb=pdb[r*4+3];
      pdb[r*4]=(axr-l)*st;pdb[r*4+1]=(ayr-t)*st;pdb[r*4+2]=(axr+rr)*st;pdb[r*4+3]=(ayr+bb)*st;
      for(int64_t c=0;c<NC;++c)pss[r*NC+c]=1.f/(1.f+std::exp(-ps->data[r*NC+c])); }
    auto tal = tal_assign(pss, pdb, anc_img, bt.gt_labels, bt.gt_boxes, bt.mask, B, A, M, NC, M_MAX_TOPK, ALPHA, BETA);
    auto Lo = pure_v8_loss(pd, ps, ancx, ancy, stride, tal.tb, tal.ts, R, NC, RM);
    backward(Lo.total);
    opt.lr = cosine_lr(it, ITERS, 2e-3f, 5); opt.step();
    if (it % 25 == 0 || it == ITERS-1) printf("  iter %3d  loss %7.3f\n", it, Lo.total->data[0]);
  }
  printf("training done.\n\n");

  // write trained weights for the .pt round-trip (Ultralytics compatibility test)
  std::filesystem::create_directories("pure/ref/data_wb/");
  auto wr = [&](const std::string& n, const std::vector<float>& v){ std::ofstream f("pure/ref/data_wb/"+n,std::ios::binary); f.write((const char*)v.data(), v.size()*sizeof(float)); };
  for (size_t i=0;i<prov.layers.size();++i){ auto& L=prov.layers[i]; std::string s=std::to_string(i);
    wr("cw"+s+".bin",L.w->data);
    if(L.kind==1){wr("bg"+s+".bin",L.gamma->data);wr("bb"+s+".bin",L.beta->data);wr("rm"+s+".bin",L.rm);wr("rv"+s+".bin",L.rv);} else wr("cb"+s+".bin",L.b->data); }
  printf("wrote trained weights to pure/ref/data_wb/ (for .pt round-trip)\n\n");

  // inference on a held-out test image (square, so letterbox is identity)
  std::string timg = DS + "te00.png";
  int w0,h0,ch; unsigned char* im = stbi_load(timg.c_str(), &w0, &h0, &ch, 3);
  auto x = make_tensor({1,3,S,S});
  for (int c=0;c<3;++c) for (int y=0;y<S;++y) for (int xx=0;xx<S;++xx) x->data[(c*S+y)*S+xx] = im[(y*w0+xx)*3+c]/255.f;
  prov.i = 0;
  auto lv = yolov8n_forward_u(x, prov, false);
  std::vector<Tensor> bx={lv[0].first,lv[1].first,lv[2].first}, cs={lv[0].second,lv[1].second,lv[2].second};
  auto pd = pack_levels(bx,1,A,4*RM); auto ps = pack_levels(cs,1,A,NC);
  std::vector<float> axv,ayv,sv; make_anchors(S,axv,ayv,sv);
  std::vector<float> pred; decode_predictions(pd->data, ps->data, axv,ayv,sv, A, NC, RM, pred);
  auto dets = nms(pred, A, NC, 0.25f, 0.5f, 50);
  printf("inference on te00.png: %zu detections\n", dets.size());
  auto putp=[&](int px,int py,int r,int g,int b){if(px<0||py<0||px>=w0||py>=h0)return;unsigned char*p=&im[(py*w0+px)*3];p[0]=r;p[1]=g;p[2]=b;};
  for (auto& d : dets) {
    printf("  cls %d(%s) conf %.2f  xyxy=(%.0f,%.0f,%.0f,%.0f)\n", d.cls, d.cls<3?NAMES[d.cls]:"?", d.conf, d.x1,d.y1,d.x2,d.y2);
    for (int t=0;t<2;++t){ for(int px=(int)d.x1;px<=(int)d.x2;++px){putp(px,(int)d.y1+t,255,255,0);putp(px,(int)d.y2-t,255,255,0);} for(int py=(int)d.y1;py<=(int)d.y2;++py){putp((int)d.x1+t,py,255,255,0);putp((int)d.x2-t,py,255,255,0);} }
  }
  stbi_write_png("synth_det.png", w0, h0, 3, im, w0*3);
  printf("wrote synth_det.png\n");
  return 0;
}
