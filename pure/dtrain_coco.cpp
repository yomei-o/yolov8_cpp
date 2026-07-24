// Phase-6b: device-resident yolov8n training over a real dataset (COCO128), measuring
// per-epoch time on the selected backend. Device forward(train BN)+backward+Adam; trusted
// host v8 loss/TAL bridged in (as Phase 6). Reports seconds/epoch so we can compare the
// device path (GPU) to the current hosted-GEMM baseline (~4.6 min/epoch).
//   run: dtrain_coco <images_dir> <imgsz> <batch> <epochs>
//   GPU (Colab): nvcc -x cu -O2 -std=c++17 --extended-lambda -arch=native -DUSE_CUDA -Ipure/third_party pure/dtrain_coco.cpp -o dtrain_coco
#define STB_IMAGE_IMPLEMENTATION
#include "dataset.hpp"        // read_yolo_dataset, load_minibatch (host)
#include "net_unfused.hpp"    // pack_levels, ProviderU, load_net_unfused_pt
#include "v8pure.hpp"
#include "tal.hpp"
#include "dtensor.hpp"
#include <cstdio>
#include <cmath>
#include <chrono>
#include <numeric>
#include <algorithm>
#include <random>

static const int64_t NC = 80, RM = 16;
struct DTL { int kind; int64_t stride; DT w, gamma, beta, b; };
struct ProvDT { std::vector<DTL> L; size_t i = 0; DTL& next() { return L[i++]; } };
static ProvDT build_dt(ProviderU& pu) { ProvDT p;
  for (auto& L : pu.layers) { DTL d; d.kind=L.kind; d.stride=L.stride; int64_t Co=L.w->shape[0],Ci=L.w->shape[1],k=L.w->shape[2];
    d.w=dfrom({Co,Ci,k,k},L.w->data); if(L.kind==1){d.gamma=dfrom({Co},L.gamma->data);d.beta=dfrom({Co},L.beta->data);} else d.b=dfrom({Co},L.b->data);
    p.L.push_back(d);} return p; }
static DT applyDT(DT x, DTL& L){ int64_t pad=L.w->shape[2]/2; if(L.kind==1) return dsilu(dbn(dconv2d(x,L.w,DT(),L.stride,pad),L.gamma,L.beta)); return dconv2d(x,L.w,L.b,L.stride,pad); }
static DT cLd(DT x, ProvDT&p){return applyDT(x,p.next());}
static DT c2f(DT x, ProvDT&p,int64_t n,bool sc){ DT y0=applyDT(x,p.next()); int64_t twoc=y0->shape[1],c=twoc/2; std::vector<DT> outs={dslice(y0,0,c),dslice(y0,c,twoc)}; DT last=outs[1];
  for(int64_t i=0;i<n;++i){DT h=applyDT(last,p.next());h=applyDT(h,p.next());last=sc?dadd(h,last):h;outs.push_back(last);} return applyDT(dconcat(outs),p.next()); }
static DT sppf(DT x, ProvDT&p){DT x1=applyDT(x,p.next());DT q1=dmaxpool2d(x1,5,1,2),q2=dmaxpool2d(q1,5,1,2),q3=dmaxpool2d(q2,5,1,2);return applyDT(dconcat({x1,q1,q2,q3}),p.next());}
static std::pair<DT,DT> det(DT x, ProvDT&p){DT hb=cLd(x,p);hb=cLd(hb,p);DT box=applyDT(hb,p.next());DT hc=cLd(x,p);hc=cLd(hc,p);DT cls=applyDT(hc,p.next());return{box,cls};}
static std::vector<std::pair<DT,DT>> fwd(DT x, ProvDT&p, std::vector<int64_t> d){
  DT x0=cLd(x,p),x1=cLd(x0,p);DT x2=c2f(x1,p,d[0],true);DT x3=cLd(x2,p);DT x4=c2f(x3,p,d[1],true);DT x5=cLd(x4,p);DT x6=c2f(x5,p,d[2],true);DT x7=cLd(x6,p);
  DT x8=c2f(x7,p,d[3],true);DT x9=sppf(x8,p);DT u10=dupsample2x(x9);DT x12=c2f(dconcat({u10,x6}),p,d[4],false);DT u13=dupsample2x(x12);DT x15=c2f(dconcat({u13,x4}),p,d[5],false);
  DT x16=cLd(x15,p);DT x18=c2f(dconcat({x16,x12}),p,d[6],false);DT x19=cLd(x18,p);DT x21=c2f(dconcat({x19,x9}),p,d[7],false);
  std::vector<std::pair<DT,DT>> o; DT lv[3]={x15,x18,x21}; for(auto&xi:lv)o.push_back(det(xi,p)); return o; }
static void decode_for_tal(const Tensor& pd,const Tensor& ps,std::vector<float>&ax,std::vector<float>&ay,std::vector<float>&ss,int64_t R,int64_t A,std::vector<float>&pdb,std::vector<float>&pss){
  pdb.assign(R*4,0);pss.assign(R*NC,0);
  for(int64_t r=0;r<R;++r){for(int j=0;j<4;++j){float mx=-1e30f;for(int k=0;k<RM;++k)mx=std::max(mx,pd->data[r*64+j*RM+k]);double s=0;float dd=0;std::vector<float>e(RM);for(int k=0;k<RM;++k){e[k]=std::exp(pd->data[r*64+j*RM+k]-mx);s+=e[k];}for(int k=0;k<RM;++k)dd+=(float)(e[k]/s)*k;pdb[r*4+j]=dd;}
    float axr=ax[r%A],ayr=ay[r%A],st=ss[r%A],l=pdb[r*4],t=pdb[r*4+1],rr=pdb[r*4+2],bb=pdb[r*4+3];pdb[r*4]=(axr-l)*st;pdb[r*4+1]=(ayr-t)*st;pdb[r*4+2]=(axr+rr)*st;pdb[r*4+3]=(ayr+bb)*st;
    for(int64_t c=0;c<NC;++c)pss[r*NC+c]=1.f/(1.f+std::exp(-ps->data[r*NC+c]));}
}

int main(int argc, char** argv) {
  setvbuf(stdout, nullptr, _IONBF, 0);
  std::string dir = argc>1?argv[1]:"pure/ref/data_yolo/images/train";
  int64_t S = argc>2?atoll(argv[2]):96;
  int BATCH = argc>3?atoi(argv[3]):4, EPOCHS = argc>4?atoi(argv[4]):2;

  ProviderU pu = load_net_unfused_pt("pure/ref/data_net/", "yolov8n.pt");
  ProvDT dev_p = build_dt(pu);
  std::vector<DT> params; for(auto&L:dev_p.L){params.push_back(L.w); if(L.kind==1){params.push_back(L.gamma);params.push_back(L.beta);} else params.push_back(L.b);}
  DAdam opt(params, 1e-3f);
  Dataset tr = read_yolo_dataset(dir, S);
  printf("train=%zu imgsz=%lld batch=%d epochs=%d\n", tr.items.size(), (long long)S, BATCH, EPOCHS);

  struct Lv{int64_t h,w;float s;}; std::vector<Lv> lv={{S/8,S/8,8.f},{S/16,S/16,16.f},{S/32,S/32,32.f}};
  std::vector<float> ax,ay,ss,anc_img; for(auto&L:lv)for(int64_t y=0;y<L.h;++y)for(int64_t x=0;x<L.w;++x){ax.push_back(x+.5f);ay.push_back(y+.5f);ss.push_back(L.s);anc_img.push_back((x+.5f)*L.s);anc_img.push_back((y+.5f)*L.s);}
  int64_t A=(int64_t)ss.size();
  std::vector<int64_t> dep={1,2,2,1,1,1,1,1};
  std::vector<int> order(tr.items.size()); std::iota(order.begin(),order.end(),0); std::mt19937 rng(0);

  for (int ep=0; ep<EPOCHS; ++ep) {
    std::shuffle(order.begin(),order.end(),rng); double eloss=0; int nb=0;
    auto t0 = std::chrono::steady_clock::now();
    for (size_t off=0; off<order.size(); off+=BATCH) {
      std::vector<int> idx(order.begin()+off, order.begin()+std::min(order.size(),off+BATCH));
      Batch bt = load_minibatch(tr, idx, false, rng());
      int64_t B=bt.B, R=B*A, Mx=bt.M;
      std::vector<float> ancx(R),ancy(R),strd(R); for(int64_t r=0;r<R;++r){int64_t a=r%A;ancx[r]=ax[a];ancy[r]=ay[a];strd[r]=ss[a];}
      opt.zero_grad();
      dev_p.i=0; auto dev = fwd(dfrom({B,3,S,S}, bt.x->data), dev_p, dep);
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
    printf("epoch %d/%d  loss %.4f  %.1f s/epoch  (%d steps, batch %d)\n", ep+1, EPOCHS, eloss/nb, secs, nb, BATCH);
  }
#if defined(__CUDACC__)
  printf("backend: GPU (CUDA)\n");
#else
  printf("backend: CPU (host)\n");
#endif
  return 0;
}
