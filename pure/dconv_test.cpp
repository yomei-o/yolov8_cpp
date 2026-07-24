// Phase-2 test: device-resident conv2d (pure/dtensor.hpp dconv2d).
//   L = sum(SiLU(conv2d(X, W, b)))  then backward; finite-difference grad-check on X/W/b.
//   CPU (MSVC): cl /std:c++17 /O2 /EHsc /Zc:preprocessor /DNOMINMAX
//        /DTHRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_CPP /I"%CUDA%/include/cccl" /I"%CUDA%/include" pure\dconv_test.cpp
//   CPU (Colab g++): g++ -O2 -std=c++17 -I/usr/local/cuda/include -DTHRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_CPP pure/dconv_test.cpp -o dconv_cpu
//   GPU (Colab nvcc): nvcc -x cu -O2 -std=c++17 --extended-lambda -arch=native -DUSE_CUDA pure/dconv_test.cpp -o dconv_gpu
#include "dtensor.hpp"
#include <cstdio>
#include <numeric>

static const std::vector<int64_t> IS{1,2,4,4}, WS{3,2,3,3}, BS{3};
static const int64_t STR = 1, PAD = 1;

static std::vector<float> ramp(int64_t n, float a, float step) {
  std::vector<float> v(n); for (int64_t i = 0; i < n; ++i) v[i] = a + step*i; return v;
}
static std::vector<float> XH = ramp(1*2*4*4, -1.0f, 0.07f);
static std::vector<float> WH = ramp(3*2*3*3, -0.5f, 0.03f);
static std::vector<float> BH = { 0.1f, -0.2f, 0.05f };

static float forward_L(const std::vector<float>& xh, const std::vector<float>& wh, const std::vector<float>& bh) {
  DT x = dfrom(IS, xh), w = dfrom(WS, wh), b = dfrom(BS, bh);
  DT L = dsum(dsilu(dconv2d(x, w, b, STR, PAD)));
  return dto_host(L)[0];
}

int main() {
  fprintf(stderr, "[0] start\n");
  DT x = dfrom(IS, XH), w = dfrom(WS, WH), b = dfrom(BS, BH);   bk::sync(); fprintf(stderr, "[1] inputs\n");
  DT c = dconv2d(x, w, b, STR, PAD);                            bk::sync(); fprintf(stderr, "[2] conv fwd ok\n");
  DT L = dsum(dsilu(c));                                        bk::sync(); fprintf(stderr, "[3] L=%.6f\n", dto_host(L)[0]);
  dbackward(L);                                                 bk::sync(); fprintf(stderr, "[4] backward ok\n");

  float Lval = dto_host(L)[0];
  std::vector<float> xg(x->numel()), wg(w->numel()), bg(b->numel());
  thrust::copy(x->grad.begin(), x->grad.end(), xg.begin());
  thrust::copy(w->grad.begin(), w->grad.end(), wg.begin());
  thrust::copy(b->grad.begin(), b->grad.end(), bg.begin());
  printf("L = %.6f\n", Lval);
  printf("checksum dX=%.6f dW=%.6f dB=%.6f\n",
         std::accumulate(xg.begin(),xg.end(),0.0), std::accumulate(wg.begin(),wg.end(),0.0),
         std::accumulate(bg.begin(),bg.end(),0.0));

  const float eps = 1e-3f; int bad = 0;
  auto check = [&](const char* nm, int i, float an, const std::vector<float>& base, int which) {
    std::vector<float> xp=XH, wp=WH, bp=BH;
    if (which==0) xp[i]+=eps; else if (which==1) wp[i]+=eps; else bp[i]+=eps;
    float fd = (forward_L(xp,wp,bp) - Lval)/eps;
    printf("  d%s[%2d] analytic %.5f  fd %.5f  |d| %.2e\n", nm, i, an, fd, std::abs(an-fd));
    if (std::abs(an-fd) > 5e-2f) ++bad;
  };
  check("X", 5,  xg[5],  XH, 0); check("X", 20, xg[20], XH, 0);
  check("W", 0,  wg[0],  WH, 1); check("W", 26, wg[26], WH, 1);
  check("B", 1,  bg[1],  BH, 2);
#if defined(__CUDACC__)
  printf("backend: GPU (CUDA)  %s\n", bad ? "GRAD MISMATCH" : "grad-check OK");
#else
  printf("backend: CPU (host)  %s\n", bad ? "GRAD MISMATCH" : "grad-check OK");
#endif
  return 0;
}
