// Phase-3 test: device-resident structural ops (dmaxpool2d, dupsample2x, dslice, dconcat).
//   L = sum(SiLU( concat[ slice(up(maxpool(X)),0:1), slice(up(maxpool(X)),1:2) ] ))
//   backward + finite-difference grad-check on X (validates all four ops' backward).
//   CPU (MSVC): cl /std:c++17 /O2 /EHsc /Zc:preprocessor /DNOMINMAX
//        /DTHRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_CPP /I"%CUDA%/include/cccl" /I"%CUDA%/include" pure\dstruct_test.cpp
//   GPU (Colab): nvcc -x cu -O2 -std=c++17 --extended-lambda -arch=native -DUSE_CUDA pure/dstruct_test.cpp -o dstruct_gpu
#include "dtensor.hpp"
#include <cstdio>
#include <numeric>

static const std::vector<int64_t> IS{1, 2, 4, 4};
static std::vector<float> XH = []{ std::vector<float> v(1*2*4*4); for (size_t i=0;i<v.size();++i) v[i] = -0.9f + 0.11f*i; return v; }();

static DT build(DT x) {
  DT m = dmaxpool2d(x, 3, 1, 1);      // [1,2,4,4]
  DT u = dupsample2x(m);              // [1,2,8,8]
  DT c = dconcat({ dslice(u, 0, 1), dslice(u, 1, 2) });   // [1,2,8,8]
  return dsum(dsilu(c));
}
static float forward_L(const std::vector<float>& xh) { return dto_host(build(dfrom(IS, xh)))[0]; }

int main() {
  fprintf(stderr, "[0] start\n");
  DT x = dfrom(IS, XH);                bk::sync(); fprintf(stderr, "[1] input\n");
  DT L = build(x);                     bk::sync(); fprintf(stderr, "[2] fwd ok  L=%.6f\n", dto_host(L)[0]);
  dbackward(L);                        bk::sync(); fprintf(stderr, "[3] backward ok\n");

  float Lval = dto_host(L)[0];
  std::vector<float> xg(x->numel()); thrust::copy(x->grad.begin(), x->grad.end(), xg.begin());
  printf("L = %.6f\n", Lval);
  printf("checksum sum(dL/dX) = %.6f\n", std::accumulate(xg.begin(), xg.end(), 0.0));

  const float eps = 1e-3f; int bad = 0;
  for (int i : {0, 7, 15, 22, 31}) {
    if (i >= (int)XH.size()) continue;
    std::vector<float> xp = XH; xp[i] += eps;
    float fd = (forward_L(xp) - Lval) / eps, an = xg[i];
    printf("  dL/dX[%2d] analytic %.5f  fd %.5f  |d| %.2e\n", i, an, fd, std::abs(an - fd));
    if (std::abs(an - fd) > 5e-2f) ++bad;
  }
#if defined(__CUDACC__)
  printf("backend: GPU (CUDA)  %s\n", bad ? "GRAD MISMATCH" : "grad-check OK");
#else
  printf("backend: CPU (host)  %s\n", bad ? "GRAD MISMATCH" : "grad-check OK");
#endif
  return 0;
}
