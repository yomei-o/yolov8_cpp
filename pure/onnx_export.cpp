// A-2 (write): export the yolov8 forward to a standard .onnx (opset 13) from the arch
// manifest + fused weights, using a hand-rolled protobuf writer (no deps). Ops emitted:
// Conv, Sigmoid+Mul (SiLU), MaxPool (SPPF), Resize (nearest upsample), Concat, Add,
// Slice (C2f channel split). Outputs the 6 raw detect tensors (box/cls per level).
//   run: onnx_export [model]   (default yolov8n; needs pure/ref/data_arch_<model>/)
#include "net_dyn.hpp"          // load_arch, load_net (Provider/ConvW), rd
#include "onnx.hpp"
#include <string>
using namespace onx;

int main(int argc, char** argv) {
  std::string model = argc > 1 ? argv[1] : "yolov8n";
  const std::string D = "pure/ref/data_arch_" + model + "/";
  auto arch = load_arch(D + "arch.txt");
  auto prov = load_net(D);
  std::ifstream io(D + "io.txt"); int64_t IMG; io >> IMG;

  Graph g; g.opset = 13;
  g.inputs.push_back({"images", {1, 3, IMG, IMG}});
  int uid = 0;
  auto uniq = [&](const char* p) { return std::string(p) + std::to_string(uid++); };

  // emit a Conv (+ optional SiLU) consuming the next fused conv; returns output name.
  auto conv = [&](const std::string& in) -> std::string {
    ConvW& c = prov.next();
    int64_t Co = c.w->shape[0], Ci = c.w->shape[1], k = c.w->shape[2], pad = k / 2;
    std::string wn = uniq("w"), bn = uniq("b"), yn = uniq("conv");
    g.init_f.push_back({wn, {Co, Ci, k, k}, c.w->data});
    g.init_f.push_back({bn, {Co}, c.b->data});
    onx::Node n; n.op_type = "Conv"; n.name = yn; n.input = {in, wn, bn}; n.output = {yn};
    n.attr.push_back({"kernel_shape", A_INTS, 0, 0, "", {k, k}, {}});
    n.attr.push_back({"strides", A_INTS, 0, 0, "", {c.stride, c.stride}, {}});
    n.attr.push_back({"pads", A_INTS, 0, 0, "", {pad, pad, pad, pad}, {}});
    n.attr.push_back({"group", A_INT, 1, 0, "", {}, {}});
    g.nodes.push_back(n);
    if (!c.act) return yn;
    std::string sn = uniq("sig"), mn = uniq("silu");
    g.nodes.push_back({"Sigmoid", sn, {yn}, {sn}, {}});
    g.nodes.push_back({"Mul", mn, {yn, sn}, {mn}, {}});
    return mn;
  };
  auto slice = [&](const std::string& in, int64_t c0, int64_t c1) -> std::string {
    std::string s = uniq("st"), e = uniq("en"), a = uniq("ax"), yn = uniq("slice");
    g.init_i.push_back({s, {1}, {c0}}); g.init_i.push_back({e, {1}, {c1}}); g.init_i.push_back({a, {1}, {1}});
    g.nodes.push_back({"Slice", yn, {in, s, e, a}, {yn}, {}});
    return yn;
  };
  auto concat = [&](const std::vector<std::string>& xs) -> std::string {
    std::string yn = uniq("cat");
    onx::Node n{"Concat", yn, xs, {yn}, {}}; n.attr.push_back({"axis", A_INT, 1, 0, "", {}, {}});
    g.nodes.push_back(n); return yn;
  };
  auto add = [&](const std::string& x, const std::string& y) -> std::string {
    std::string yn = uniq("add"); g.nodes.push_back({"Add", yn, {x, y}, {yn}, {}}); return yn;
  };
  auto maxpool = [&](const std::string& in) -> std::string {
    std::string yn = uniq("mp");
    onx::Node n{"MaxPool", yn, {in}, {yn}, {}};
    n.attr.push_back({"kernel_shape", A_INTS, 0, 0, "", {5, 5}, {}});
    n.attr.push_back({"strides", A_INTS, 0, 0, "", {1, 1}, {}});
    n.attr.push_back({"pads", A_INTS, 0, 0, "", {2, 2, 2, 2}, {}});
    g.nodes.push_back(n); return yn;
  };
  auto resize2x = [&](const std::string& in) -> std::string {
    std::string sc = uniq("scales"), yn = uniq("up");
    g.init_f.push_back({sc, {4}, {1.f, 1.f, 2.f, 2.f}});
    onx::Node n{"Resize", yn, {in, "", sc}, {yn}, {}};                 // "" = optional roi omitted
    n.attr.push_back({"mode", A_STRING, 0, 0, "nearest", {}, {}});
    n.attr.push_back({"coordinate_transformation_mode", A_STRING, 0, 0, "asymmetric", {}, {}});
    n.attr.push_back({"nearest_mode", A_STRING, 0, 0, "floor", {}, {}});
    g.nodes.push_back(n); return yn;
  };

  auto c2f = [&](const std::string& x, int nbott, bool shortcut) -> std::string {
    std::string y0 = conv(x);
    int64_t twoc = prov.convs[prov.i - 1].w->shape[0], c = twoc / 2;   // cv1 out channels
    std::vector<std::string> outs = {slice(y0, 0, c), slice(y0, c, twoc)};
    std::string last = outs[1];
    for (int i = 0; i < nbott; ++i) { std::string h = conv(last); h = conv(h); last = shortcut ? add(h, last) : h; outs.push_back(last); }
    return conv(concat(outs));
  };
  auto sppf = [&](const std::string& x) -> std::string {
    std::string x1 = conv(x), q1 = maxpool(x1), q2 = maxpool(q1), q3 = maxpool(q2);
    return conv(concat({x1, q1, q2, q3}));
  };

  // walk the arch graph exactly like net_dyn, emitting ONNX nodes
  std::vector<std::string> outv(arch.nlayers);
  int det_lvl = 0;
  for (auto& L : arch.layers) {
    std::vector<std::string> in;
    for (int fv : L.from) in.push_back(fv == -1 ? (L.idx == 0 ? std::string("images") : outv[L.idx - 1]) : outv[fv]);
    switch (L.type) {
      case 0: outv[L.idx] = conv(in[0]); break;
      case 1: outv[L.idx] = c2f(in[0], L.nbott, L.shortcut); break;
      case 2: outv[L.idx] = sppf(in[0]); break;
      case 3: outv[L.idx] = resize2x(in[0]); break;
      case 4: outv[L.idx] = concat(in); break;
      case 5:
        for (auto& t : in) {
          std::string hb = conv(t); hb = conv(hb); std::string box = conv(hb);   // cv2: last is plain
          std::string hc = conv(t); hc = conv(hc); std::string cls = conv(hc);   // cv3
          int64_t st = 8 << det_lvl;
          // expose stable graph-output names via Identity, and register them
          std::string bn = "box" + std::to_string(det_lvl), cn = "cls" + std::to_string(det_lvl);
          g.nodes.push_back({"Identity", uniq("idb"), {box}, {bn}, {}});
          g.nodes.push_back({"Identity", uniq("idc"), {cls}, {cn}, {}});
          g.outputs.push_back({bn, {1, 4 * arch.reg_max, IMG / st, IMG / st}});
          g.outputs.push_back({cn, {1, arch.nc,          IMG / st, IMG / st}});
          ++det_lvl;
        }
        break;
    }
  }
  std::string outp = model + ".onnx";
  save_onnx(g, outp);
  printf("wrote %s  (%zu nodes, %zu float inits, %zu int inits, consumed %zu/%zu convs)\n",
         outp.c_str(), g.nodes.size(), g.init_f.size(), g.init_i.size(), prov.i, prov.convs.size());
  return 0;
}
