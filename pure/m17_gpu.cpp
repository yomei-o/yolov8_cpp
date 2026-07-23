// M17 / C-4: exercise the device backend seam (backend.hpp). Same source runs on
// the CPU today (no GPU) and, built with nvcc -DUSE_CUDA, on a real GPU.
//   CPU : g++  -std=c++20 -O2 [-fopenmp]                    pure/m17_gpu.cpp -o m17
//   GPU : nvcc -x cu -std=c++17 --extended-lambda -DUSE_CUDA -O2 pure/m17_gpu.cpp -o m17
#include "backend.hpp"
#include <vector>
#include <random>
#include <cmath>
#include <cstdio>

int main() {
  printf("backend: %s\n", bk::backend_name());
  std::mt19937 rng(0);
  std::normal_distribution<float> nd(0, 1);
  double worst = 0;

  // ---- 1) vector add: c = a + b on device buffers ----
  {
    int64_t n = 1 << 16;
    std::vector<float> a(n), b(n), c(n), ref(n);
    for (auto& v : a) v = nd(rng);
    for (auto& v : b) v = nd(rng);
    bk::Buffer<float> da, db, dc;
    da.from_host(a.data(), n); db.from_host(b.data(), n); dc.alloc(n);
    const float* pa = da.data(); const float* pb = db.data(); float* pc = dc.data();
    bk::parallel_for(n, [=] BK_HD (int64_t i) { pc[i] = pa[i] + pb[i]; });
    dc.to_host(c.data());
    double d = 0; for (int64_t i = 0; i < n; ++i) d = std::max(d, (double)std::abs(c[i] - (a[i] + b[i])));
    printf("[vecadd] n=%lld  max|diff| = %.3e  %s\n", (long long)n, d, d == 0 ? "OK" : "FAIL");
    worst = std::max(worst, d);
  }

  // ---- 2) gemm: C = A*B, checked against a plain CPU triple loop ----
  {
    int64_t M = 64, K = 48, N = 40;
    std::vector<float> A(M*K), B(K*N), C(M*N), ref(M*N, 0.f);
    for (auto& v : A) v = nd(rng);
    for (auto& v : B) v = nd(rng);
    for (int64_t m = 0; m < M; ++m) for (int64_t k = 0; k < K; ++k)
      for (int64_t n = 0; n < N; ++n) ref[m*N+n] += A[m*K+k] * B[k*N+n];

    bk::Buffer<float> dA, dB, dC;
    dA.from_host(A.data(), M*K); dB.from_host(B.data(), K*N); dC.alloc(M*N);
    bk::gemm(dA.data(), dB.data(), dC.data(), M, K, N);
    dC.to_host(C.data());
    double d = 0; for (int64_t i = 0; i < M*N; ++i) d = std::max(d, (double)std::abs(C[i] - ref[i]));
    printf("[gemm  ] %lldx%lld*%lldx%lld  max|diff| = %.3e  %s\n",
           (long long)M,(long long)K,(long long)K,(long long)N, d, d < 1e-4 ? "OK" : "FAIL");
    worst = std::max(worst, d);
  }

  bool ok = worst < 1e-4;
  printf("\n%s (backend seam: same source -> CUDA on a GPU, CPU here)\n", ok ? "M17 OK" : "FAIL");
  return ok ? 0 : 1;
}
