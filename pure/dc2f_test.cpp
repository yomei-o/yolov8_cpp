// Phase-4 test: assemble yolov8's signature C2f block from device-resident ops and
// finite-difference grad-check it end to end (conv+BN+SiLU + channel split + bottleneck
// with residual add + concat + conv). Validates the composition at block scale.
//   C2f: y0 = CBS_1x1(x) -> 2c ch; split a,b; for each bottleneck: h=CBS3(CBS3(b)), b=+h;
//        cat(a, b0, b1..); out = CBS_1x1(cat).   CBS = SiLU(BN(conv, no bias)).
//   CPU (MSVC): cl /std:c++17 /O2 /EHsc /Zc:preprocessor /DNOMINMAX
//        /DTHRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_CPP /I"%CUDA%/include/cccl" /I"%CUDA%/include" pure\dc2f_test.cpp
//   GPU (Colab): nvcc -x cu -O2 -std=c++17 --extended-lambda -arch=native -DUSE_CUDA pure/dc2f_test.cpp -o dc2f_gpu
#include "dtensor.hpp"
#include <cstdio>
#include <numeric>
#include <cmath>

static std::vector<float> fill(int64_t n, float ph) { std::vector<float> v(n); for (int64_t i=0;i<n;++i) v[i]=0.3f*std::sin(0.7f*i+ph); return v; }
static std::vector<float> ones(int64_t n, float a){ std::vector<float> v(n,a); return v; }

static const std::vector<int64_t> IS{1,4,8,8};
// weights (deterministic). conv shapes: [Cout,Cin,k,k]
static const std::vector<float> CV1 = fill(4*4*1*1, 0.1f);       // 4->4 1x1
static const std::vector<float> BC1 = fill(2*2*3*3, 0.2f);       // 2->2 3x3
static const std::vector<float> BC2 = fill(2*2*3*3, 0.3f);       // 2->2 3x3
static const std::vector<float> CC2 = fill(4*6*1*1, 0.4f);       // 6->4 1x1
static const std::vector<float> G4 = ones(4,1.1f), B4 = fill(4,0.5f), G2 = ones(2,0.9f), B2 = fill(2,0.6f);

static DT cbs(DT x, const std::vector<float>& cw, std::vector<int64_t> ws,
              const std::vector<float>& g, const std::vector<float>& b, int64_t s, int64_t p) {
  DT w = dfrom(ws, cw), gg = dfrom({ws[0]}, g), bb = dfrom({ws[0]}, b);
  return dsilu(dbn(dconv2d(x, w, DT(), s, p), gg, bb));
}

static DT build(const std::vector<float>& xh) {
  DT x  = dfrom(IS, xh);
  DT y0 = cbs(x, CV1, {4,4,1,1}, G4, B4, 1, 0);            // [1,4,8,8]
  DT a  = dslice(y0, 0, 2), b = dslice(y0, 2, 4);          // 2 + 2
  DT h  = cbs(b, BC1, {2,2,3,3}, G2, B2, 1, 1);
  h     = cbs(h, BC2, {2,2,3,3}, G2, B2, 1, 1);
  DT last = dadd(h, b);                                     // residual
  DT cat = dconcat({a, b, last});                          // [1,6,8,8]
  DT out = cbs(cat, CC2, {4,6,1,1}, G4, B4, 1, 0);         // [1,4,8,8]
  return dsum(out);
}

int main() {
  std::vector<float> XH = fill(1*4*8*8, 0.05f);
  fprintf(stderr, "[0] start\n");
  DT L = build(XH);   bk::sync(); fprintf(stderr, "[1] C2f fwd ok  L=%.6f\n", dto_host(L)[0]);
  // rebuild retaining the input node so we can read its grad
  DT x = dfrom(IS, XH);
  { DT y0 = cbs(x, CV1,{4,4,1,1},G4,B4,1,0); DT a=dslice(y0,0,2),b=dslice(y0,2,4);
    DT h=cbs(b,BC1,{2,2,3,3},G2,B2,1,1); h=cbs(h,BC2,{2,2,3,3},G2,B2,1,1);
    DT last=dadd(h,b); DT cat=dconcat({a,b,last}); DT out=cbs(cat,CC2,{4,6,1,1},G4,B4,1,0); DT LL=dsum(out);
    dbackward(LL); bk::sync(); fprintf(stderr, "[2] backward ok\n");
    float Lval = dto_host(LL)[0];
    std::vector<float> xg(x->numel()); thrust::copy(x->grad.begin(), x->grad.end(), xg.begin());
    printf("L = %.6f\n", Lval);
    printf("checksum sum(dL/dX) = %.6f\n", std::accumulate(xg.begin(), xg.end(), 0.0));
    const float eps = 1e-3f; int bad = 0;
    for (int i : {0, 33, 100, 200, 255}) {
      if (i >= (int)XH.size()) continue;
      std::vector<float> xp = XH; xp[i] += eps;
      float fd = (dto_host(build(xp))[0] - Lval)/eps, an = xg[i];
      printf("  dL/dX[%3d] analytic %.5f  fd %.5f  |d| %.2e\n", i, an, fd, std::abs(an-fd));
      if (std::abs(an-fd) > 6e-2f) ++bad;
    }
#if defined(__CUDACC__)
    printf("backend: GPU (CUDA)  %s\n", bad ? "GRAD MISMATCH" : "grad-check OK");
#else
    printf("backend: CPU (host)  %s\n", bad ? "GRAD MISMATCH" : "grad-check OK");
#endif
  }
  return 0;
}
