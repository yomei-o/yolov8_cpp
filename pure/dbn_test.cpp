// Phase-3b test: device-resident BatchNorm2d (pure/dtensor.hpp dbn), training mode.
//   L = sum(SiLU(BN(X; gamma,beta)))  then backward; finite-difference grad-check on X/gamma/beta.
//   CPU (MSVC): cl /std:c++17 /O2 /EHsc /Zc:preprocessor /DNOMINMAX
//        /DTHRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_CPP /I"%CUDA%/include/cccl" /I"%CUDA%/include" pure\dbn_test.cpp
//   GPU (Colab): nvcc -x cu -O2 -std=c++17 --extended-lambda -arch=native -DUSE_CUDA pure/dbn_test.cpp -o dbn_gpu
#include "dtensor.hpp"
#include <cstdio>
#include <numeric>

static const std::vector<int64_t> IS{2, 3, 4, 4}, PS{3};
static std::vector<float> XH = []{ std::vector<float> v(2*3*4*4); for (size_t i=0;i<v.size();++i) v[i] = std::sin(0.3f*i) * (1.f + 0.02f*i); return v; }();
static std::vector<float> GH = { 1.2f, 0.8f, 1.5f };
static std::vector<float> BH = { 0.1f, -0.3f, 0.2f };

static float forward_L(const std::vector<float>& xh, const std::vector<float>& gh, const std::vector<float>& bh) {
  DT x = dfrom(IS, xh), g = dfrom(PS, gh), b = dfrom(PS, bh);
  return dto_host(dsum(dsilu(dbn(x, g, b))))[0];
}

int main() {
  fprintf(stderr, "[0] start\n");
  DT x = dfrom(IS, XH), g = dfrom(PS, GH), b = dfrom(PS, BH);   bk::sync(); fprintf(stderr, "[1] inputs\n");
  DT L = dsum(dsilu(dbn(x, g, b)));                            bk::sync(); fprintf(stderr, "[2] fwd ok  L=%.6f\n", dto_host(L)[0]);
  dbackward(L);                                                bk::sync(); fprintf(stderr, "[3] backward ok\n");

  float Lval = dto_host(L)[0];
  std::vector<float> xg(x->numel()), gg(g->numel()), bg(b->numel());
  thrust::copy(x->grad.begin(), x->grad.end(), xg.begin());
  thrust::copy(g->grad.begin(), g->grad.end(), gg.begin());
  thrust::copy(b->grad.begin(), b->grad.end(), bg.begin());
  printf("L = %.6f\n", Lval);
  printf("checksum dX=%.6f dGamma=%.6f dBeta=%.6f\n",
         std::accumulate(xg.begin(),xg.end(),0.0), std::accumulate(gg.begin(),gg.end(),0.0), std::accumulate(bg.begin(),bg.end(),0.0));

  const float eps = 1e-3f; int bad = 0;
  auto ck = [&](const char* nm, int i, float an, int which) {
    std::vector<float> xp=XH, gp=GH, bp=BH;
    if (which==0) xp[i]+=eps; else if (which==1) gp[i]+=eps; else bp[i]+=eps;
    float fd = (forward_L(xp,gp,bp) - Lval)/eps;
    printf("  d%s[%d] analytic %.5f  fd %.5f  |d| %.2e\n", nm, i, an, fd, std::abs(an-fd));
    if (std::abs(an-fd) > 6e-2f) ++bad;
  };
  ck("X", 10, xg[10], 0); ck("X", 40, xg[40], 0);
  ck("Gamma", 0, gg[0], 1); ck("Gamma", 2, gg[2], 1);
  ck("Beta", 1, bg[1], 2);
#if defined(__CUDACC__)
  printf("backend: GPU (CUDA)  %s\n", bad ? "GRAD MISMATCH" : "grad-check OK");
#else
  printf("backend: CPU (host)  %s\n", bad ? "GRAD MISMATCH" : "grad-check OK");
#endif
  return 0;
}
