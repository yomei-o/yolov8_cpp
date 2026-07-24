// Phase-5 test: a full DEVICE-RESIDENT training step loop. A tiny trainable conv net,
// loss = sum(out^2) (minimise output energy), device Adam. forward -> backward -> Adam step,
// repeated — the loss must decrease monotonically, proving the whole train loop (incl. the
// optimizer) runs device-resident on the selected backend.
//   CPU (MSVC): cl /std:c++17 /O2 /EHsc /Zc:preprocessor /DNOMINMAX
//        /DTHRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_CPP /I"%CUDA%/include/cccl" /I"%CUDA%/include" pure\dtrain_test.cpp
//   GPU (Colab): nvcc -x cu -O2 -std=c++17 --extended-lambda -arch=native -DUSE_CUDA pure/dtrain_test.cpp -o dtrain_gpu
#include "dtensor.hpp"
#include <cstdio>
#include <cmath>

int main() {
  const int64_t S = 8;
  std::vector<float> xh(1*3*S*S); for (size_t i = 0; i < xh.size(); ++i) xh[i] = std::sin(0.2f*i);
  DT x = dfrom({1,3,S,S}, xh);                                  // fixed input
  auto rnd = [](int64_t n, float ph){ std::vector<float> v(n); for (int64_t i=0;i<n;++i) v[i]=0.2f*std::sin(0.9f*i+ph); return v; };
  DT w1 = dfrom({8,3,3,3}, rnd(8*3*3*3, 0.1f));                 // conv1 3->8
  DT w2 = dfrom({4,8,3,3}, rnd(4*8*3*3, 0.2f));                 // conv2 8->4
  DAdam opt({w1, w2}, 5e-3f);

  fprintf(stderr, "[0] start\n");
  float first = 0, last = 0;
  for (int step = 0; step < 30; ++step) {
    opt.zero_grad();
    DT h  = dsilu(dconv2d(x, w1, DT(), 1, 1));                  // [1,8,8,8]
    DT o  = dconv2d(h, w2, DT(), 1, 1);                         // [1,4,8,8]
    DT L  = dsum(dmul(o, o));                                   // sum of squares
    dbackward(L);
    opt.step();
    float lv = dto_host(L)[0];
    if (step == 0) first = lv;
    last = lv;
    if (step % 5 == 0 || step == 29) printf("step %2d  loss %.5f\n", step, lv);
  }
  bk::sync();
  bool ok = last < first * 0.5f;                               // clearly decreasing
  printf("loss %.5f -> %.5f  (%.1f%% of start)  %s\n", first, last, 100.f*last/first, ok ? "TRAIN OK" : "NOT DECREASING");
#if defined(__CUDACC__)
  printf("backend: GPU (CUDA)\n");
#else
  printf("backend: CPU (host)\n");
#endif
  return 0;
}
