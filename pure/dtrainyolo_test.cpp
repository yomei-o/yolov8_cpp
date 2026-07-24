// Phase-6: REAL yolov8 training step, device-resident (hybrid). The whole yolov8n forward
// + backward + Adam run device-resident (GPU under nvcc); the trusted host v8 loss + TAL
// (pure_v8_loss / tal_assign) are reused unchanged, bridged in: device head outputs -> host
// -> loss -> grads injected back onto the device heads -> dbackward_from -> device Adam.
// A few steps on a synthetic image with 2 GT boxes; the real v8 loss must DECREASE.
//   CPU (MSVC): cl /std:c++17 /O2 /EHsc /Zc:preprocessor /DNOMINMAX
//        /DTHRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_CPP /I"%CUDA%/include/cccl" /I"%CUDA%/include" /Ipure\third_party pure\dtrainyolo_test.cpp
//   GPU (Colab): nvcc -x cu -O2 -std=c++17 --extended-lambda -arch=native -DUSE_CUDA -Ipure/third_party pure/dtrainyolo_test.cpp -o dtrainyolo_gpu
#define STB_IMAGE_IMPLEMENTATION
#include "net_unfused.hpp"     // CPU engine: pack_levels, ProviderU, load_net_unfused_pt
#include "v8pure.hpp"          // pure_v8_loss (trusted)
#include "tal.hpp"             // tal_assign (trusted)
#include "dtensor.hpp"         // device engine
#include <cstdio>
#include <cmath>

static const int64_t NC = 80, RM = 16;

struct DTL { int kind; int64_t stride; DT w, gamma, beta, b; };
struct ProvDT { std::vector<DTL> L; size_t i = 0; DTL& next() { return L[i++]; } };
static ProvDT build_dt(ProviderU& pu) {
  ProvDT p;
  for (auto& L : pu.layers) { DTL d; d.kind = L.kind; d.stride = L.stride;
    int64_t Co=L.w->shape[0], Ci=L.w->shape[1], k=L.w->shape[2];
    d.w = dfrom({Co,Ci,k,k}, L.w->data);
    if (L.kind==1) { d.gamma = dfrom({Co}, L.gamma->data); d.beta = dfrom({Co}, L.beta->data); }
    else d.b = dfrom({Co}, L.b->data);
    p.L.push_back(d); }
  return p;
}
static DT applyDT(DT x, DTL& L) { int64_t pad = L.w->shape[2]/2;
  if (L.kind==1) return dsilu(dbn(dconv2d(x, L.w, DT(), L.stride, pad), L.gamma, L.beta));
  return dconv2d(x, L.w, L.b, L.stride, pad);
}
static DT cLd(DT x, ProvDT& p) { return applyDT(x, p.next()); }
static DT c2f(DT x, ProvDT& p, int64_t n, bool sc) {
  DT y0 = applyDT(x, p.next()); int64_t twoc = y0->shape[1], c = twoc/2;
  std::vector<DT> outs = { dslice(y0,0,c), dslice(y0,c,twoc) }; DT last = outs[1];
  for (int64_t i=0;i<n;++i){ DT h=applyDT(last,p.next()); h=applyDT(h,p.next()); last=sc?dadd(h,last):h; outs.push_back(last); }
  return applyDT(dconcat(outs), p.next());
}
static DT sppf(DT x, ProvDT& p) { DT x1=applyDT(x,p.next()); DT q1=dmaxpool2d(x1,5,1,2),q2=dmaxpool2d(q1,5,1,2),q3=dmaxpool2d(q2,5,1,2); return applyDT(dconcat({x1,q1,q2,q3}),p.next()); }
static std::pair<DT,DT> det(DT x, ProvDT& p) { DT hb=cLd(x,p);hb=cLd(hb,p);DT box=applyDT(hb,p.next()); DT hc=cLd(x,p);hc=cLd(hc,p);DT cls=applyDT(hc,p.next()); return {box,cls}; }
static std::vector<std::pair<DT,DT>> fwd(DT x, ProvDT& p, std::vector<int64_t> d) {
  DT x0=cLd(x,p),x1=cLd(x0,p); DT x2=c2f(x1,p,d[0],true); DT x3=cLd(x2,p);
  DT x4=c2f(x3,p,d[1],true); DT x5=cLd(x4,p); DT x6=c2f(x5,p,d[2],true); DT x7=cLd(x6,p);
  DT x8=c2f(x7,p,d[3],true); DT x9=sppf(x8,p);
  DT u10=dupsample2x(x9); DT x12=c2f(dconcat({u10,x6}),p,d[4],false);
  DT u13=dupsample2x(x12); DT x15=c2f(dconcat({u13,x4}),p,d[5],false);
  DT x16=cLd(x15,p); DT x18=c2f(dconcat({x16,x12}),p,d[6],false);
  DT x19=cLd(x18,p); DT x21=c2f(dconcat({x19,x9}),p,d[7],false);
  std::vector<std::pair<DT,DT>> out; DT levs[3]={x15,x18,x21}; for(auto&xi:levs) out.push_back(det(xi,p)); return out;
}
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

int main() {
  const int64_t S = 64, B = 1;
  ProviderU pu = load_net_unfused_pt("pure/ref/data_net/", "yolov8n.pt");
  ProvDT dev_p = build_dt(pu);
  std::vector<DT> params; for (auto& L : dev_p.L) { params.push_back(L.w); if (L.kind==1){params.push_back(L.gamma);params.push_back(L.beta);} else params.push_back(L.b); }
  DAdam opt(params, 1e-3f);

  struct Lv{int64_t h,w; float s;}; std::vector<Lv> lv={{S/8,S/8,8.f},{S/16,S/16,16.f},{S/32,S/32,32.f}};
  std::vector<float> ax,ay,ss,anc_img; for(auto&L:lv)for(int64_t y=0;y<L.h;++y)for(int64_t x=0;x<L.w;++x){ax.push_back(x+.5f);ay.push_back(y+.5f);ss.push_back(L.s);anc_img.push_back((x+.5f)*L.s);anc_img.push_back((y+.5f)*L.s);}
  int64_t A=(int64_t)ss.size(), R=B*A;
  std::vector<float> ancx(R),ancy(R),strd(R); for(int64_t r=0;r<R;++r){int64_t a=r%A;ancx[r]=ax[a];ancy[r]=ay[a];strd[r]=ss[a];}
  int64_t M=2; std::vector<float> gt_boxes={10,10,30,30, 35,40,55,60}; std::vector<int64_t> gt_labels={0,1}; std::vector<float> mask={1,1};
  std::vector<float> xh(1*3*S*S); for(size_t i=0;i<xh.size();++i) xh[i]=std::sin(0.05f*i)*0.5f+0.1f;
  std::vector<int64_t> dep={1,2,2,1,1,1,1,1};

  fprintf(stderr,"[0] start: device fwd + host v8 loss + device bwd + device Adam\n");
  float first=0, last=0;
  for (int step=0; step<12; ++step) {
    opt.zero_grad();
    dev_p.i=0; auto dev = fwd(dfrom({1,3,S,S}, xh), dev_p, dep);      // DEVICE forward
    std::vector<Tensor> bx, cs;                                       // CPU leaf mirrors of heads
    for (int l=0;l<3;++l){ auto& bl=dev[l].first; auto& cl=dev[l].second;
      bx.push_back(from_data({bl->shape[0],bl->shape[1],bl->shape[2],bl->shape[3]}, dto_host(bl), true));
      cs.push_back(from_data({cl->shape[0],cl->shape[1],cl->shape[2],cl->shape[3]}, dto_host(cl), true)); }
    auto pdp=pack_levels(bx,B,A,4*RM); auto psp=pack_levels(cs,B,A,NC);
    std::vector<float> pdb,pss; decode_for_tal(pdp,psp,ax,ay,ss,R,A,pdb,pss);
    auto tal=tal_assign(pss,pdb,anc_img, gt_labels,gt_boxes,mask, B,A,M,NC,10,0.5f,6.0f);
    auto Lo=pure_v8_loss(pdp,psp,ancx,ancy,strd, tal.tb,tal.ts, R,NC,RM);   // TRUSTED host loss
    backward(Lo.total);                                              // host backward -> bx/cs grads
    for (int l=0;l<3;++l){ thrust::copy(bx[l]->grad.begin(),bx[l]->grad.end(), dev[l].first->grad.begin());
                           thrust::copy(cs[l]->grad.begin(),cs[l]->grad.end(), dev[l].second->grad.begin()); }
    std::vector<DT> heads; for(int l=0;l<3;++l){heads.push_back(dev[l].first);heads.push_back(dev[l].second);}
    dbackward_from(heads);                                           // DEVICE backward through the net
    opt.step();                                                      // DEVICE Adam
    float lvv=Lo.total->data[0]; if(step==0)first=lvv; last=lvv;
    printf("step %2d  v8 loss %.5f\n", step, lvv);
  }
  bk::sync();
  printf("v8 loss %.5f -> %.5f  %s\n", first, last, last<first ? "DECREASING (real v8 device training)" : "NOT DECREASING");
#if defined(__CUDACC__)
  printf("backend: GPU (CUDA)\n");
#else
  printf("backend: CPU (host)\n");
#endif
  return 0;
}
