// POC: ONE source file, runs on CPU or GPU, chosen at compile time via Thrust's switchable
// "device system" — so device-resident buffers do NOT lock us into CUDA.
//
//   CPU (MSVC, CCCL headers):
//     cl /std:c++17 /O2 /EHsc /DTHRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_CPP \
//        /I<cccl>/thrust /I<cccl>/cub /I<cccl>/libcudacxx/include pure\thrust_poc.cpp
//   CPU (Colab g++):
//     g++ -O2 -std=c++17 -I/usr/local/cuda/include \
//        -DTHRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_CPP pure/thrust_poc.cpp -o poc_cpu
//     (or ...=THRUST_DEVICE_SYSTEM_OMP -fopenmp  for multicore CPU)
//   GPU (Colab nvcc):
//     nvcc -x cu -O2 -std=c++17 --extended-lambda pure/thrust_poc.cpp -o poc_gpu
//
// Shows a device-resident buffer (thrust::device_vector) with two CHAINED ops (SiLU then
// scale) + a reduction — all on the selected backend, with NO host round-trip between ops.
// CPU and GPU builds must print the same sum (numerical parity of the switch).
#include <cstdio>
#include <cmath>
#include <thrust/device_vector.h>
#include <thrust/transform.h>
#include <thrust/reduce.h>
#include <thrust/sequence.h>

#if defined(__CUDACC__)
  #define POC_HD __host__ __device__       // real CUDA build: functors callable on device
#else
  #define POC_HD                            // host-only build: annotations are no-ops
#endif

struct SiLU  { POC_HD float operator()(float x) const { return x / (1.f + expf(-x)); } };
struct Scale { float a;  POC_HD float operator()(float x) const { return x * a; } };

int main() {
  const int N = 1 << 20;
  thrust::device_vector<float> x(N);                            // device-resident buffer
  thrust::sequence(x.begin(), x.end(), -2.0f, 4.0f / N);        // ramp -2 .. ~2
  thrust::transform(x.begin(), x.end(), x.begin(), SiLU());     // op1 on the backend
  thrust::transform(x.begin(), x.end(), x.begin(), Scale{0.5f});// op2 chained, no host copy
  float s = thrust::reduce(x.begin(), x.end(), 0.0f);           // reduction, same backend
  printf("N=%d  sum(0.5*SiLU(ramp)) = %.6f\n", N, s);
#if defined(__CUDACC__)
  printf("backend: GPU (CUDA device system)\n");
#else
  printf("backend: CPU (host device system)\n");
#endif
  return 0;
}
