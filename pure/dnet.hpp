// Device-resident yolov8 net (any size) + trainable provider + checkpoint save.
// Loads a size's arch (manifest_unfused.txt + names.txt [+ depths.txt]) and weights (.pt),
// builds a device provider (conv w + BN gamma/beta + running_mean/var, or plain conv w/b),
// runs the yolov8 forward from device ops, and can save a state_dict .pt (eval-ready, since
// BN running stats are EMA-tracked during training). One source CPU/GPU (see dtensor.hpp).
#pragma once
#include "net_unfused.hpp"     // ProviderU, load_net_unfused_pt, pack_levels; pulls ptio (pt::save_pt)
#include "dtensor.hpp"
#include <string>
#include <fstream>

struct DTLayer { int kind; int64_t stride; float eps; DT w, gamma, beta, b, rm, rv; };
struct ProvD { std::vector<DTLayer> L; size_t i = 0; DTLayer& next() { return L[i++]; } };

// Build a device provider from a size's arch dir + weights .pt.
inline ProvD dnet_build(const std::string& arch_dir, const std::string& weights_pt) {
  ProviderU pu = load_net_unfused_pt(arch_dir, weights_pt);
  ProvD p;
  for (auto& L : pu.layers) {
    DTLayer d; d.kind = L.kind; d.stride = L.stride; d.eps = L.eps;
    int64_t Co = L.w->shape[0], Ci = L.w->shape[1], k = L.w->shape[2];
    d.w = dfrom({Co,Ci,k,k}, L.w->data);
    if (L.kind == 1) { d.gamma = dfrom({Co}, L.gamma->data); d.beta = dfrom({Co}, L.beta->data);
                       d.rm = dfrom({Co}, L.rm); d.rv = dfrom({Co}, L.rv); }
    else d.b = dfrom({Co}, L.b->data);
    p.L.push_back(std::move(d));
  }
  return p;
}
inline std::vector<DT> dnet_params(ProvD& p) {
  std::vector<DT> ps;
  for (auto& L : p.L) { ps.push_back(L.w); if (L.kind==1){ps.push_back(L.gamma);ps.push_back(L.beta);} else ps.push_back(L.b); }
  return ps;
}
inline std::vector<int64_t> dnet_depths(const std::string& arch_dir) {
  std::vector<int64_t> d; std::ifstream f(arch_dir + "depths.txt"); int64_t v; while (f >> v) d.push_back(v);
  if (d.empty()) d = {1,2,2,1,1,1,1,1};   // yolov8n/s default
  return d;
}

static inline DT dnet_apply(DT x, DTLayer& L, bool train) {
  int64_t pad = L.w->shape[2]/2;
  if (L.kind == 1) return dsilu(dbn(dconv2d(x, L.w, DT(), L.stride, pad), L.gamma, L.beta, L.eps,
                                    train ? L.rm : DT(), train ? L.rv : DT(), 0.03f));
  return dconv2d(x, L.w, L.b, L.stride, pad);
}
static inline DT dnet_cL(DT x, ProvD& p, bool tr) { return dnet_apply(x, p.next(), tr); }
static inline DT dnet_c2f(DT x, ProvD& p, int64_t n, bool sc, bool tr) {
  DT y0 = dnet_apply(x, p.next(), tr); int64_t twoc = y0->shape[1], c = twoc/2;
  std::vector<DT> outs = { dslice(y0,0,c), dslice(y0,c,twoc) }; DT last = outs[1];
  for (int64_t i=0;i<n;++i){ DT h=dnet_apply(last,p.next(),tr); h=dnet_apply(h,p.next(),tr); last=sc?dadd(h,last):h; outs.push_back(last); }
  return dnet_apply(dconcat(outs), p.next(), tr);
}
static inline DT dnet_sppf(DT x, ProvD& p, bool tr) {
  DT x1=dnet_apply(x,p.next(),tr); DT q1=dmaxpool2d(x1,5,1,2),q2=dmaxpool2d(q1,5,1,2),q3=dmaxpool2d(q2,5,1,2);
  return dnet_apply(dconcat({x1,q1,q2,q3}), p.next(), tr);
}
static inline std::pair<DT,DT> dnet_det(DT x, ProvD& p, bool tr) {
  DT hb=dnet_cL(x,p,tr);hb=dnet_cL(hb,p,tr);DT box=dnet_apply(hb,p.next(),tr);
  DT hc=dnet_cL(x,p,tr);hc=dnet_cL(hc,p,tr);DT cls=dnet_apply(hc,p.next(),tr);
  return {box,cls};
}
inline std::vector<std::pair<DT,DT>> dnet_forward(DT x, ProvD& p, const std::vector<int64_t>& d, bool tr) {
  DT x0=dnet_cL(x,p,tr),x1=dnet_cL(x0,p,tr);DT x2=dnet_c2f(x1,p,d[0],true,tr);DT x3=dnet_cL(x2,p,tr);
  DT x4=dnet_c2f(x3,p,d[1],true,tr);DT x5=dnet_cL(x4,p,tr);DT x6=dnet_c2f(x5,p,d[2],true,tr);DT x7=dnet_cL(x6,p,tr);
  DT x8=dnet_c2f(x7,p,d[3],true,tr);DT x9=dnet_sppf(x8,p,tr);
  DT u10=dupsample2x(x9);DT x12=dnet_c2f(dconcat({u10,x6}),p,d[4],false,tr);
  DT u13=dupsample2x(x12);DT x15=dnet_c2f(dconcat({u13,x4}),p,d[5],false,tr);
  DT x16=dnet_cL(x15,p,tr);DT x18=dnet_c2f(dconcat({x16,x12}),p,d[6],false,tr);
  DT x19=dnet_cL(x18,p,tr);DT x21=dnet_c2f(dconcat({x19,x9}),p,d[7],false,tr);
  std::vector<std::pair<DT,DT>> o; DT lv[3]={x15,x18,x21}; for(auto&xi:lv)o.push_back(dnet_det(xi,p,tr)); return o;
}

// Save the trained provider as a state_dict .pt (loads in Ultralytics). names.txt from arch_dir.
inline void dnet_save(ProvD& p, const std::string& arch_dir, const std::string& path) {
  std::vector<std::string> names; { std::ifstream f(arch_dir + "names.txt"); std::string s; while (f >> s) names.push_back(s); }
  std::vector<pt::Tensor> ck; size_t k = 0;
  auto push = [&](DT t) { pt::Tensor o; if (k < names.size()) o.name = names[k];
    o.shape.assign(t->shape.begin(), t->shape.end()); o.data = dto_host(t); ck.push_back(std::move(o)); ++k; };
  for (auto& L : p.L) { push(L.w);
    if (L.kind==1){ push(L.gamma); push(L.beta); push(L.rm); push(L.rv); } else push(L.b); }
  pt::save_pt(ck, path);
}
