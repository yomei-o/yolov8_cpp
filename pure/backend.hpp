// Single-header device backend seam. Same kernel source runs on CUDA (nvcc,
// -DUSE_CUDA) or on the CPU (g++/MSVC, no GPU needed — the CPU path IS the
// "emulation"). Kernels are written once as `bk::parallel_for(n, [=] BK_HD (i){...})`.
//
//   CPU  : g++  -std=c++20 -O2 [-fopenmp]              pure/m17_gpu.cpp
//   CUDA : nvcc -x cu -std=c++17 --extended-lambda -DUSE_CUDA -O2 pure/m17_gpu.cpp
#pragma once
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <cstdio>

#ifdef USE_CUDA
  #include <cuda_runtime.h>
  #define BK_HD __host__ __device__
  #define BK_CHECK(x) do { cudaError_t e_=(x); if (e_) { \
      fprintf(stderr,"CUDA error %s at %s:%d\n",cudaGetErrorString(e_),__FILE__,__LINE__); std::abort(); } } while(0)
#else
  #include "parallel.hpp"          // CPU parallel_for (OpenMP / std::thread)
  #define BK_HD
#endif

namespace bk {

inline const char* backend_name() {
#ifdef USE_CUDA
  return "CUDA (GPU)";
#else
  return "CPU (emulation — no GPU)";
#endif
}

// Owning device buffer. On CPU it is just host memory; on CUDA it is device memory
// with explicit host<->device copies, so call sites are already GPU-shaped.
template <class T>
struct Buffer {
  T* p_ = nullptr; int64_t n_ = 0;
  Buffer() = default;
  explicit Buffer(int64_t n) { alloc(n); }
  Buffer(const Buffer&) = delete; Buffer& operator=(const Buffer&) = delete;
  ~Buffer() { free_(); }

  T* data() { return p_; }
  const T* data() const { return p_; }
  int64_t size() const { return n_; }

#ifdef USE_CUDA
  void alloc(int64_t n) { free_(); n_ = n; BK_CHECK(cudaMalloc(&p_, n * sizeof(T))); }
  void from_host(const T* h, int64_t n) { alloc(n); BK_CHECK(cudaMemcpy(p_, h, n*sizeof(T), cudaMemcpyHostToDevice)); }
  void to_host(T* h) const { BK_CHECK(cudaMemcpy(h, p_, n_*sizeof(T), cudaMemcpyDeviceToHost)); }
  void free_() { if (p_) cudaFree(p_); p_ = nullptr; }
#else
  void alloc(int64_t n) { free_(); n_ = n; p_ = (T*)std::malloc(n * sizeof(T)); }
  void from_host(const T* h, int64_t n) { alloc(n); std::memcpy(p_, h, n * sizeof(T)); }
  void to_host(T* h) const { std::memcpy(h, p_, n_ * sizeof(T)); }
  void free_() { std::free(p_); p_ = nullptr; }
#endif
};

#ifdef USE_CUDA
template <class F> __global__ void _grid_kernel(int64_t n, F f) {
  int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) f(i);
}
template <class F> inline void parallel_for(int64_t n, F f) {
  if (n <= 0) return;
  const int block = 256;
  int64_t grid = (n + block - 1) / block;
  _grid_kernel<<<(unsigned)grid, block>>>(n, f);
  BK_CHECK(cudaGetLastError());
  BK_CHECK(cudaDeviceSynchronize());
}
inline void sync() { BK_CHECK(cudaDeviceSynchronize()); }
#else
template <class F> inline void parallel_for(int64_t n, F f) {
  ::parallel_for(n, f);            // reuse the CPU parallel_for
}
inline void sync() {}
#endif

// C(M,N) = A(M,K) * B(K,N). Pointers live in the active memory space (device under
// CUDA, host on CPU).
#ifdef USE_CUDA
// One thread per output element.
inline void gemm(const float* A, const float* B, float* C, int64_t M, int64_t K, int64_t N) {
  parallel_for(M * N, [=] BK_HD (int64_t idx) {
    int64_t m = idx / N, n = idx % N;
    float s = 0.f;
    for (int64_t k = 0; k < K; ++k) s += A[m * K + k] * B[k * N + n];
    C[idx] = s;
  });
}
#else
// One thread per row m; inner loop over the contiguous N so it auto-vectorises.
inline void gemm(const float* A, const float* B, float* C, int64_t M, int64_t K, int64_t N) {
  parallel_for(M, [=] (int64_t m) {
    float* cr = C + m * N;
    for (int64_t n = 0; n < N; ++n) cr[n] = 0.f;
    for (int64_t k = 0; k < K; ++k) {
      float av = A[m * K + k]; const float* br = B + k * N;
      for (int64_t n = 0; n < N; ++n) cr[n] += av * br[n];
    }
  });
}
#endif

// Host-facing GEMM: inputs/outputs are host pointers. On CUDA it stages to the
// device, runs, and copies back (per-call staging for now — device-resident later);
// on CPU it runs in place with no copies. This is the single place the engine's ops
// (conv im2col-GEMM, matmul) route through to reach the device.
inline void gemm_hosted(const float* A, const float* B, float* C,
                        int64_t M, int64_t K, int64_t N) {
#ifdef USE_CUDA
  Buffer<float> dA, dB, dC;
  dA.from_host(A, M * K); dB.from_host(B, K * N); dC.alloc(M * N);
  gemm(dA.data(), dB.data(), dC.data(), M, K, N);
  dC.to_host(C);
#else
  gemm(A, B, C, M, K, N);
#endif
}

} // namespace bk
