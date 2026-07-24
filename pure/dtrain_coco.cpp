// Device-resident yolov8 training over a standard-YOLO dataset (e.g. COCO128), ANY size,
// with checkpoint save. Device fwd(train BN, EMA running stats)+bwd+Adam; trusted host v8
// loss/TAL bridged in. Saves last.pt / best.pt (state_dict, loads in Ultralytics). Times s/epoch.
//   run: dtrain_coco <images_dir> <imgsz> <batch> <epochs> [model=yolov8n]
//   GPU: nvcc -x cu -O2 -std=c++17 --extended-lambda -arch=native -DUSE_CUDA [-DUSE_CUBLAS -lcublas] -Ipure/third_party pure/dtrain_coco.cpp -o dtrain_coco
#define STB_IMAGE_IMPLEMENTATION
#include "dataset.hpp"        // read_yolo_dataset, load_minibatch
#include "dnet.hpp"           // device yolov8 (any size) + build/forward/params/save
#include "v8pure.hpp"
#include "tal.hpp"
#include <cstdio>
#include <cmath>
#include <chrono>
#include <numeric>
#include <algorithm>
#include <random>

static const int64_t NC = 80, RM = 16;
static void decode_for_tal(const Tensor& pd, const Tensor& ps, std::vector<float>& ax, std::vector<float>& ay,
                           std::vector<float>& ss, int64_t R, int64_t A, std::vector<float>& pdb, std::vector<float>& pss) {
  pdb.assign(R*4,0); pss.assign(R*NC,0);
  for (int64_t r=0;r<R;++r){
    for (int j=0;j<4;++j){ float mx=-1e30f; for(int k=0;k<RM;++k) mx=std::max(mx, pd->data[r*64+j*RM+k]);
      double s=0; float dd=0; std::vector<float> e(RM); for(int k=0;k<RM;++k){e[k]=std::exp(pd->data[r*64+j*RM+k]-mx);s+=e[k];} for(int k=0;k<RM;++k) dd+=(float)(e[k]/s)*k; pdb[r*4+j]=dd; }
    float axr=ax[r%A],ayr=ay[r%A],st=ss[r%A],l=pdb[r*4],t=pdb[r*4+1],rr=pdb[r*4+2],bb=pdb[r*4+3];
    pdb[r*4]=(axr-l)*st; pdb[r*4+1]=(ayr-t)*st; pdb[r*4+2]=(axr+rr)*st; pdb[r*4+3]=(ayr+bb)*st;
    for (int64_t c=0;c<NC;++c) pss[r*NC+c]=1.f/(1.f+std::exp(-ps->data[r*NC+c]));
  }
}

int main(int argc, char** argv) {
  setvbuf(stdout, nullptr, _IONBF, 0);
  std::string dir = argc>1?argv[1]:"pure/ref/data_yolo/images/train";
  int64_t S = argc>2?atoll(argv[2]):96;
  int BATCH = argc>3?atoi(argv[3]):4, EPOCHS = argc>4?atoi(argv[4]):2;
  std::string model = argc>5?argv[5]:"yolov8n";
  std::string arch = (model=="yolov8n") ? "pure/ref/data_net/" : "pure/ref/arch/"+model+"/";
  std::string weights = model + ".pt";

  ProvD prov = dnet_build(arch, weights);
  std::vector<DT> params = dnet_params(prov);
  DAdam opt(params, 1e-3f);
  std::vector<int64_t> dep = dnet_depths(arch);
  Dataset tr = read_yolo_dataset(dir, S);
  printf("model=%s train=%zu imgsz=%lld batch=%d epochs=%d depths=[", model.c_str(), tr.items.size(), (long long)S, BATCH, EPOCHS);
  for (size_t i=0;i<dep.size();++i) printf("%lld%s", (long long)dep[i], i+1<dep.size()?",":"]\n");

  struct Lv{int64_t h,w;float s;}; std::vector<Lv> lv={{S/8,S/8,8.f},{S/16,S/16,16.f},{S/32,S/32,32.f}};
  std::vector<float> ax,ay,ss,anc_img; for(auto&L:lv)for(int64_t y=0;y<L.h;++y)for(int64_t x=0;x<L.w;++x){ax.push_back(x+.5f);ay.push_back(y+.5f);ss.push_back(L.s);anc_img.push_back((x+.5f)*L.s);anc_img.push_back((y+.5f)*L.s);}
  int64_t A=(int64_t)ss.size();
  std::vector<int> order(tr.items.size()); std::iota(order.begin(),order.end(),0); std::mt19937 rng(0);
  double best = 1e30;

  for (int ep=0; ep<EPOCHS; ++ep) {
    std::shuffle(order.begin(),order.end(),rng); double eloss=0; int nb=0;
    auto t0 = std::chrono::steady_clock::now();
    for (size_t off=0; off<order.size(); off+=BATCH) {
      std::vector<int> idx(order.begin()+off, order.begin()+std::min(order.size(),off+BATCH));
      Batch bt = load_minibatch(tr, idx, false, rng());
      int64_t B=bt.B, R=B*A, Mx=bt.M;
      std::vector<float> ancx(R),ancy(R),strd(R); for(int64_t r=0;r<R;++r){int64_t a=r%A;ancx[r]=ax[a];ancy[r]=ay[a];strd[r]=ss[a];}
      opt.zero_grad();
      prov.i=0; auto dev = dnet_forward(dfrom({B,3,S,S}, bt.x->data), prov, dep, true);
      std::vector<Tensor> bx,cs;
      for(int l=0;l<3;++l){auto&bl=dev[l].first;auto&cl=dev[l].second;
        bx.push_back(from_data({bl->shape[0],bl->shape[1],bl->shape[2],bl->shape[3]}, dto_host(bl), true));
        cs.push_back(from_data({cl->shape[0],cl->shape[1],cl->shape[2],cl->shape[3]}, dto_host(cl), true));}
      auto pdp=pack_levels(bx,B,A,4*RM); auto psp=pack_levels(cs,B,A,NC);
      std::vector<float> pdb,pss; decode_for_tal(pdp,psp,ax,ay,ss,R,A,pdb,pss);
      auto tal=tal_assign(pss,pdb,anc_img, bt.gt_labels,bt.gt_boxes,bt.mask, B,A,Mx,NC,10,0.5f,6.0f);
      auto Lo=pure_v8_loss(pdp,psp,ancx,ancy,strd, tal.tb,tal.ts, R,NC,RM);
      backward(Lo.total);
      for(int l=0;l<3;++l){thrust::copy(bx[l]->grad.begin(),bx[l]->grad.end(),dev[l].first->grad.begin());thrust::copy(cs[l]->grad.begin(),cs[l]->grad.end(),dev[l].second->grad.begin());}
      std::vector<DT> heads; for(int l=0;l<3;++l){heads.push_back(dev[l].first);heads.push_back(dev[l].second);}
      dbackward_from(heads); opt.step();
      eloss += Lo.total->data[0]; ++nb;
    }
    bk::sync();
    double secs = std::chrono::duration<double>(std::chrono::steady_clock::now()-t0).count();
    double avg = eloss/nb;
    dnet_save(prov, arch, "last.pt");
    if (avg < best) { best = avg; dnet_save(prov, arch, "best.pt"); }
    printf("epoch %d/%d  loss %.4f  %.1f s/epoch%s\n", ep+1, EPOCHS, avg, secs, avg<=best?"  *best*":"");
  }
  printf("done. best loss %.4f. wrote last.pt / best.pt (pure C++, %s)\n", best, model.c_str());
#if defined(__CUDACC__)
  printf("backend: GPU (CUDA)\n");
#else
  printf("backend: CPU (host)\n");
#endif
  return 0;
}
