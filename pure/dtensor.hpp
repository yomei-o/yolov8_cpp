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

// ---- conv2d (im2col + bk::gemm), device-resident ----
// im2col gather: col[K,P] from I[Cin,H,W]; K=Cin*kh*kw, P=OH*OW. One thread per col element.
inline void dim2col(const float* I, int64_t Cin, int64_t H, int64_t W, int64_t kh, int64_t kw,
                    int64_t OH, int64_t OW, int64_t stride, int64_t pad, float* col) {
  int64_t P = OH * OW, tot = Cin * kh * kw * P;
  bk::parallel_for(tot, [=] BK_HD (int64_t idx) {
    int64_t p = idx % P, row = idx / P;
    int64_t s = row % kw, t = row / kw; int64_t r = t % kh, ci = t / kh;
    int64_t oh = p / OW, ow = p % OW;
    int64_t ih = oh * stride - pad + r, iw = ow * stride - pad + s;
    col[idx] = (ih >= 0 && ih < H && iw >= 0 && iw < W) ? I[(ci * H + ih) * W + iw] : 0.f;
  });
}
// col2im scatter-add: dI[Cin,H,W] += from dcol[K,P]. Parallelised over Cin so each thread
// writes a disjoint channel — race-free on host (threads) AND device (kernel), no atomics.
inline void dcol2im(const float* dcol, int64_t Cin, int64_t H, int64_t W, int64_t kh, int64_t kw,
                    int64_t OH, int64_t OW, int64_t stride, int64_t pad, float* dI) {
  int64_t P = OH * OW;
  bk::parallel_for(Cin, [=] BK_HD (int64_t ci) {
    for (int64_t r = 0; r < kh; ++r) for (int64_t s = 0; s < kw; ++s) {
      const float* dcrow = dcol + (((ci * kh + r) * kw + s) * P);
      for (int64_t oh = 0; oh < OH; ++oh) {
        int64_t ih = oh * stride - pad + r; if (ih < 0 || ih >= H) continue;
        float* girow = dI + (ci * H + ih) * W;
        for (int64_t ow = 0; ow < OW; ++ow) { int64_t iw = ow * stride - pad + s; if (iw < 0 || iw >= W) continue; girow[iw] += dcrow[oh * OW + ow]; }
      }
    }
  });
}
// in (N,Cin,H,W), w (Cout,Cin,kh,kw), bias (Cout) or null. groups=1.
inline DT dconv2d(DT in, DT w, DT bias, int64_t stride, int64_t pad) {
  int64_t N = in->shape[0], Cin = in->shape[1], H = in->shape[2], Wd = in->shape[3];
  int64_t Cout = w->shape[0], kh = w->shape[2], kw = w->shape[3];
  int64_t OH = (H + 2*pad - kh)/stride + 1, OW = (Wd + 2*pad - kw)/stride + 1;
  int64_t K = Cin*kh*kw, P = OH*OW;
  DT o = dmake({N, Cout, OH, OW});
  { thrust::device_vector<float> col(K*P); float* colp = thrust::raw_pointer_cast(col.data());
    for (int64_t n = 0; n < N; ++n) {
      dim2col(in->dp() + n*Cin*H*Wd, Cin, H, Wd, kh, kw, OH, OW, stride, pad, colp);
      bk::gemm(w->dp(), colp, o->dp() + n*Cout*P, Cout, K, P);              // O(Cout,P)=W(Cout,K)*col(K,P)
      if (bias) { float* B = bias->dp(); float* On = o->dp() + n*Cout*P;
        bk::parallel_for(Cout*P, [=] BK_HD (int64_t idx) { On[idx] += B[idx/P]; }); }
    }
  }
  o->parents = bias ? std::vector<DT>{in,w,bias} : std::vector<DT>{in,w};
  o->backward_fn = [in,w,bias,o,N,Cin,H,Wd,Cout,kh,kw,OH,OW,stride,pad,K,P]() {
    if (bias) { float* GB = bias->gp(); float* GO = o->gp();
      bk::parallel_for(Cout, [=] BK_HD (int64_t co) { float a = 0.f;
        for (int64_t n = 0; n < N; ++n) { const float* g = GO + (n*Cout+co)*P; for (int64_t p = 0; p < P; ++p) a += g[p]; }
        GB[co] += a; }); }
    thrust::device_vector<float> col(K*P), dcol(K*P);
    float* colp = thrust::raw_pointer_cast(col.data()), *dcolp = thrust::raw_pointer_cast(dcol.data());
    for (int64_t n = 0; n < N; ++n) {
      dim2col(in->dp() + n*Cin*H*Wd, Cin, H, Wd, kh, kw, OH, OW, stride, pad, colp);
      bk::gemm_nt(o->gp() + n*Cout*P, colp, w->gp(), Cout, K, P, 1.f);      // dW += dO(Cout,P)*col(K,P)^T
      bk::gemm_tn(w->dp(), o->gp() + n*Cout*P, dcolp, K, P, Cout, 0.f);     // dcol = W^T * dO
      dcol2im(dcolp, Cin, H, Wd, kh, kw, OH, OW, stride, pad, in->gp() + n*Cin*H*Wd);
    }
  };
  return o;
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
