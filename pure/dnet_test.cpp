// Phase-4b: full yolov8n forward, device-resident, checked against the trusted CPU engine
// (net_unfused.hpp yolov8n_forward_u, which matches Ultralytics). Same weights + same input;
// BN folded into conv for the eval forward. Self-contained — no Python, no committed ref.
//   CPU (MSVC): cl /std:c++17 /O2 /EHsc /Zc:preprocessor /DNOMINMAX
//        /DTHRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_CPP /I"%CUDA%/include/cccl" /I"%CUDA%/include" /Ipure\third_party pure\dnet_test.cpp
//   GPU (Colab): nvcc -x cu -O2 -std=c++17 --extended-lambda -arch=native -DUSE_CUDA -Ipure/third_party pure/dnet_test.cpp -o dnet_gpu
#define STB_IMAGE_IMPLEMENTATION
#include "net_unfused.hpp"     // CPU engine (Tensor): ProviderU, load_net_unfused_pt, yolov8n_forward_u
#include "dtensor.hpp"         // device engine (DT)
#include <cstdio>
#include <cmath>
#include <numeric>

struct DLayer { DT w, b; int64_t stride; bool act; };
struct ProvD { std::vector<DLayer> L; size_t i = 0; DLayer& next() { return L[i++]; } };

// Build a device provider from the CPU provider, folding conv+BN -> conv (eval): for each
// out channel co, scale = gamma/sqrt(rv+eps); w*=scale; b = beta - rm*scale.
static ProvD build_provd(ProviderU& pu) {
  ProvD pd;
  for (auto& L : pu.layers) {
    DLayer d; d.stride = L.stride;
    int64_t Co = L.w->shape[0], Ci = L.w->shape[1], kh = L.w->shape[2], kw = L.w->shape[3], ksz = Ci*kh*kw;
    if (L.kind == 1) { d.act = true;
      std::vector<float> wf(L.w->data.size()), bf(Co);
      for (int64_t co = 0; co < Co; ++co) { float sc = L.gamma->data[co] / std::sqrt(L.rv[co] + L.eps);
        for (int64_t j = 0; j < ksz; ++j) wf[co*ksz+j] = L.w->data[co*ksz+j] * sc;
        bf[co] = L.beta->data[co] - L.rm[co]*sc; }
      d.w = dfrom({Co,Ci,kh,kw}, wf); d.b = dfrom({Co}, bf);
    } else { d.act = false; d.w = dfrom({Co,Ci,kh,kw}, L.w->data); d.b = dfrom({Co}, L.b->data); }
    pd.L.push_back(d);
  }
  return pd;
}

static DT applyD(DT x, DLayer& L) { int64_t pad = L.w->shape[2]/2; DT y = dconv2d(x, L.w, L.b, L.stride, pad); return L.act ? dsilu(y) : y; }
static DT cLD(DT x, ProvD& p) { return applyD(x, p.next()); }
static DT c2f_d(DT x, ProvD& p, int64_t n, bool sc) {
  DT y0 = applyD(x, p.next()); int64_t twoc = y0->shape[1], c = twoc/2;
  std::vector<DT> outs = { dslice(y0,0,c), dslice(y0,c,twoc) }; DT last = outs[1];
  for (int64_t i = 0; i < n; ++i) { DT h = applyD(last, p.next()); h = applyD(h, p.next()); last = sc ? dadd(h,last) : h; outs.push_back(last); }
  return applyD(dconcat(outs), p.next());
}
static DT sppf_d(DT x, ProvD& p) {
  DT x1 = applyD(x, p.next()); DT q1 = dmaxpool2d(x1,5,1,2), q2 = dmaxpool2d(q1,5,1,2), q3 = dmaxpool2d(q2,5,1,2);
  return applyD(dconcat({x1,q1,q2,q3}), p.next());
}
static std::pair<DT,DT> detect_d(DT x, ProvD& p) {
  DT hb = cLD(x,p); hb = cLD(hb,p); DT box = applyD(hb, p.next());
  DT hc = cLD(x,p); hc = cLD(hc,p); DT cls = applyD(hc, p.next());
  return {box, cls};
}
static std::vector<std::pair<DT,DT>> forward_d(DT x, ProvD& p, std::vector<int64_t> d) {
  DT x0 = cLD(x,p), x1 = cLD(x0,p); DT x2 = c2f_d(x1,p,d[0],true); DT x3 = cLD(x2,p);
  DT x4 = c2f_d(x3,p,d[1],true); DT x5 = cLD(x4,p); DT x6 = c2f_d(x5,p,d[2],true); DT x7 = cLD(x6,p);
  DT x8 = c2f_d(x7,p,d[3],true); DT x9 = sppf_d(x8,p);
  DT u10 = dupsample2x(x9); DT x12 = c2f_d(dconcat({u10,x6}),p,d[4],false);
  DT u13 = dupsample2x(x12); DT x15 = c2f_d(dconcat({u13,x4}),p,d[5],false);
  DT x16 = cLD(x15,p); DT x18 = c2f_d(dconcat({x16,x12}),p,d[6],false);
  DT x19 = cLD(x18,p); DT x21 = c2f_d(dconcat({x19,x9}),p,d[7],false);
  std::vector<std::pair<DT,DT>> out; DT levels[3] = {x15,x18,x21};
  for (auto& xi : levels) out.push_back(detect_d(xi,p));
  return out;
}

static float maxdiff(const std::vector<float>& a, const Tensor& b) {
  float m = 0; for (size_t i = 0; i < a.size(); ++i) m = std::max(m, std::abs(a[i] - b->data[i])); return m;
}

int main() {
  const int64_t S = 64;
  fprintf(stderr, "[0] loading yolov8n.pt\n");
  ProviderU pu = load_net_unfused_pt("pure/ref/data_net/", "yolov8n.pt");
  fprintf(stderr, "[1] %zu layers; building device provider (fold BN)\n", pu.layers.size());
  ProvD pd = build_provd(pu);

  std::vector<float> xh(1*3*S*S); for (size_t i = 0; i < xh.size(); ++i) xh[i] = std::sin(0.05f*i)*0.5f + 0.1f;
  std::vector<int64_t> dep = {1,2,2,1,1,1,1,1};

  pu.i = 0; auto cpu = yolov8n_forward_u(from_data({1,3,S,S}, xh), pu, /*training=*/false, dep);
  fprintf(stderr, "[2] CPU-engine forward done\n");
  pd.i = 0; auto dev = forward_d(dfrom({1,3,S,S}, xh), pd, dep); bk::sync();
  fprintf(stderr, "[3] device forward done\n");

  float worst = 0;
  const char* nm[3] = {"P3","P4","P5"};
  for (int l = 0; l < 3; ++l) {
    float db = maxdiff(dto_host(dev[l].first),  cpu[l].first);
    float dc = maxdiff(dto_host(dev[l].second), cpu[l].second);
    printf("  level %s: box[%lldx%lld] max|d|=%.3e   cls max|d|=%.3e\n",
           nm[l], (long long)cpu[l].first->shape[1], (long long)cpu[l].first->shape[2], db, dc);
    worst = std::max({worst, db, dc});
  }
  printf("full yolov8n forward: worst |device - CPU-engine| = %.3e   %s\n", worst, worst < 2e-3f ? "MATCH" : "MISMATCH");
#if defined(__CUDACC__)
  printf("backend: GPU (CUDA)\n");
#else
  printf("backend: CPU (host)\n");
#endif
  return 0;
}
