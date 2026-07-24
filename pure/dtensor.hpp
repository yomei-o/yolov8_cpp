// Device-resident autograd core (Phase 1 of the GPU device-residency work).
// A tensor whose data/grad live in a thrust::device_vector, so intermediate results stay on
// the selected backend across ops — no per-op host<->device copies. The SAME source builds
// CPU (Thrust host device system, g++/MSVC) or GPU (nvcc/CUDA); elementwise ops go through
// thrust::transform, reductions through thrust::reduce, and matmul reuses the existing
// CPU/GPU-switchable bk::gemm (backend.hpp). Reverse-mode autograd via a small tape.
#pragma once
#include "backend.hpp"                 // bk::gemm / gemm_nt / gemm_tn (host or device)
#include <thrust/device_vector.h>
#include <thrust/transform.h>
#include <thrust/reduce.h>
#include <thrust/fill.h>
#include <thrust/copy.h>
#include <thrust/functional.h>
#include <cmath>
#include <memory>
#include <vector>
#include <functional>
#include <unordered_set>

#if defined(__CUDACC__)
  #define DHD __host__ __device__
#else
  #define DHD
#endif

// ---- functors (portable: real device functions under nvcc, plain host otherwise) ----
struct SiLUf   { DHD float operator()(float x) const { return x / (1.f + expf(-x)); } };
struct dSiLUf  { DHD float operator()(float x) const { float s = 1.f/(1.f+expf(-x)); return s*(1.f + x*(1.f - s)); } };
struct AddCf   { float g; DHD float operator()(float v) const { return v + g; } };

struct DNode;
using DT = std::shared_ptr<DNode>;
struct DNode {
  std::vector<int64_t> shape;
  thrust::device_vector<float> data, grad;   // device-resident (host under CPP backend)
  std::vector<DT> parents;
  std::function<void()> backward_fn;
  int64_t numel() const { int64_t n = 1; for (auto d : shape) n *= d; return n; }
  float* dp()  { return thrust::raw_pointer_cast(data.data()); }
  float* gp()  { return thrust::raw_pointer_cast(grad.data()); }
};

inline DT dmake(std::vector<int64_t> shape) {
  auto n = std::make_shared<DNode>(); n->shape = std::move(shape);
  int64_t k = n->numel(); n->data.resize(k, 0.f); n->grad.resize(k, 0.f); return n;
}
inline DT dfrom(std::vector<int64_t> shape, const std::vector<float>& host) {
  DT t = dmake(std::move(shape)); thrust::copy(host.begin(), host.end(), t->data.begin()); return t;
}
inline std::vector<float> dto_host(const DT& t) {
  std::vector<float> h(t->numel()); thrust::copy(t->data.begin(), t->data.end(), h.begin()); return h;
}

// ---- elementwise (same-shape) ----
inline DT dadd(DT a, DT b) {
  DT y = dmake(a->shape);
  thrust::transform(a->data.begin(), a->data.end(), b->data.begin(), y->data.begin(), thrust::plus<float>());
  y->parents = {a, b};
  y->backward_fn = [a, b, y]() {
    thrust::transform(a->grad.begin(), a->grad.end(), y->grad.begin(), a->grad.begin(), thrust::plus<float>());
    thrust::transform(b->grad.begin(), b->grad.end(), y->grad.begin(), b->grad.begin(), thrust::plus<float>());
  };
  return y;
}
inline DT dmul(DT a, DT b) {                 // Hadamard
  DT y = dmake(a->shape);
  thrust::transform(a->data.begin(), a->data.end(), b->data.begin(), y->data.begin(), thrust::multiplies<float>());
  y->parents = {a, b};
  y->backward_fn = [a, b, y]() {
    thrust::device_vector<float> t(a->numel());
    thrust::transform(y->grad.begin(), y->grad.end(), b->data.begin(), t.begin(), thrust::multiplies<float>());
    thrust::transform(a->grad.begin(), a->grad.end(), t.begin(), a->grad.begin(), thrust::plus<float>());
    thrust::transform(y->grad.begin(), y->grad.end(), a->data.begin(), t.begin(), thrust::multiplies<float>());
    thrust::transform(b->grad.begin(), b->grad.end(), t.begin(), b->grad.begin(), thrust::plus<float>());
  };
  return y;
}
inline DT dsilu(DT x) {
  DT y = dmake(x->shape);
  thrust::transform(x->data.begin(), x->data.end(), y->data.begin(), SiLUf());
  y->parents = {x};
  y->backward_fn = [x, y]() {
    thrust::device_vector<float> d(x->numel());
    thrust::transform(x->data.begin(), x->data.end(), d.begin(), dSiLUf());       // silu'(x)
    thrust::transform(y->grad.begin(), y->grad.end(), d.begin(), d.begin(), thrust::multiplies<float>());
    thrust::transform(x->grad.begin(), x->grad.end(), d.begin(), x->grad.begin(), thrust::plus<float>());
  };
  return y;
}

// ---- matmul: Y[M,N] = A[M,K] * B[K,N] (reuses bk::gemm, CPU/GPU switchable) ----
inline DT dmatmul(DT A, DT B) {
  int64_t M = A->shape[0], K = A->shape[1], N = B->shape[1];
  DT Y = dmake({M, N});
  bk::gemm(A->dp(), B->dp(), Y->dp(), M, K, N);
  Y->parents = {A, B};
  Y->backward_fn = [A, B, Y, M, K, N]() {
    bk::gemm_nt(Y->gp(), B->dp(), A->gp(), M, K, N, 1.f);   // dA[M,K] += dY[M,N] * B[K,N]^T
    bk::gemm_tn(A->dp(), Y->gp(), B->gp(), K, N, M, 1.f);   // dB[K,N] += A[M,K]^T * dY[M,N]
  };
  return Y;
}

// ---- reduction: sum -> scalar ----
inline DT dsum(DT x) {
  DT y = dmake({1});
  y->data[0] = thrust::reduce(x->data.begin(), x->data.end(), 0.f);
  y->parents = {x};
  y->backward_fn = [x, y]() {
    float g = y->grad[0];                                  // scalar broadcast to all inputs
    thrust::transform(x->grad.begin(), x->grad.end(), x->grad.begin(), AddCf{g});
  };
  return y;
}

// ---- reverse-mode backward over the tape ----
inline void dbackward(DT root) {
  std::vector<DT> topo; std::unordered_set<DNode*> seen;
  std::function<void(const DT&)> build = [&](const DT& n) {
    if (!n || seen.count(n.get())) return; seen.insert(n.get());
    for (auto& p : n->parents) build(p); topo.push_back(n);
  };
  build(root);
  thrust::fill(root->grad.begin(), root->grad.end(), 1.f);  // seed dL/dL = 1
  for (auto it = topo.rbegin(); it != topo.rend(); ++it)
    if ((*it)->backward_fn) (*it)->backward_fn();
}
