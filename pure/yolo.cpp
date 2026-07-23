// Unified pure-C++ CLI: `yolo <train|val|detect|export> [--flags]`, reading a standard
// Ultralytics data.yaml. No Python at run time.
//   train  --data d.yaml --weights init.pt [--epochs 100 --batch 16 --imgsz 640
//                                            --mosaic 1 --mixup 1 --close-mosaic 10]
//   val    --data d.yaml --weights best.pt [--imgsz 640]
//   detect --weights best.pt --source img.jpg [--imgsz 640 --conf 0.25 --out out.png]
//   export --weights best.pt [--imgsz 640]        (delegates to onnx_export — see note)
// build: cl /std:c++20 /O2 /EHsc /Ipure\third_party pure\yolo.cpp
#define STB_IMAGE_IMPLEMENTATION
#include "dataset.hpp"                  // includes stb_image.h once (with the impl above)
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"
#include "net_unfused.hpp"
#include "v8pure.hpp"
#include "tal.hpp"
#include "optim.hpp"
#include "infer.hpp"
#include "metrics.hpp"
#include "ptio.hpp"
#include <cstdio>
#include <numeric>
#include <algorithm>
#include <random>
#include <sstream>
#include <map>

static const int64_t NC0 = 80, RM = 16;
static const std::string DN = "pure/ref/data_net/";

// ------- data.yaml (minimal: path/train/val/nc/names, inline-list names) -------
struct DataYaml { std::string path, train, val; int nc = 80; std::vector<std::string> names; };
static std::string trim(std::string s) {
  size_t a = s.find_first_not_of(" \t\r\n'\""); size_t b = s.find_last_not_of(" \t\r\n'\"");
  return a == std::string::npos ? std::string() : s.substr(a, b - a + 1);
}
static DataYaml parse_yaml(const std::string& p) {
  std::ifstream f(p); if (!f) { printf("cannot open %s\n", p.c_str()); std::exit(1); }
  DataYaml d; std::string line;
  while (std::getline(f, line)) {
    auto h = line.find('#'); if (h != std::string::npos) line = line.substr(0, h);
    auto c = line.find(':'); if (c == std::string::npos) continue;
    std::string k = trim(line.substr(0, c)), v = trim(line.substr(c + 1));
    if (k == "path") d.path = v; else if (k == "train") d.train = v; else if (k == "val") d.val = v;
    else if (k == "nc") d.nc = atoi(v.c_str());
    else if (k == "names") {
      auto lb = v.find('['), rb = v.rfind(']');
      if (lb != std::string::npos && rb != std::string::npos) {
        std::stringstream ss(v.substr(lb + 1, rb - lb - 1)); std::string tok;
        while (std::getline(ss, tok, ',')) { tok = trim(tok); if (!tok.empty()) d.names.push_back(tok); }
      }
    }
  }
  return d;
}
static std::string join(const std::string& a, const std::string& b) { return a.empty() ? b : a + "/" + b; }

// ------- arg parsing: --key value / --key=value -------
struct Args { std::map<std::string, std::string> m;
  std::string get(const std::string& k, const std::string& def = "") const { auto it = m.find(k); return it == m.end() ? def : it->second; }
  int geti(const std::string& k, int def) const { auto it = m.find(k); return it == m.end() ? def : atoi(it->second.c_str()); }
  float getf(const std::string& k, float def) const { auto it = m.find(k); return it == m.end() ? def : (float)atof(it->second.c_str()); }
};
static Args parse_args(int argc, char** argv, int start) {
  Args a;
  for (int i = start; i < argc; ++i) { std::string s = argv[i];
    if (s.rfind("--", 0) != 0) continue; s = s.substr(2);
    auto eq = s.find('=');
    if (eq != std::string::npos) a.m[s.substr(0, eq)] = s.substr(eq + 1);
    else if (i + 1 < argc && std::string(argv[i+1]).rfind("--", 0) != 0) a.m[s] = argv[++i];
    else a.m[s] = "1";
  }
  return a;
}

static const char* COCO[80] = {
  "person","bicycle","car","motorcycle","airplane","bus","train","truck","boat","traffic light",
  "fire hydrant","stop sign","parking meter","bench","bird","cat","dog","horse","sheep","cow",
  "elephant","bear","zebra","giraffe","backpack","umbrella","handbag","tie","suitcase","frisbee",
  "skis","snowboard","sports ball","kite","baseball bat","baseball glove","skateboard","surfboard","tennis racket","bottle",
  "wine glass","cup","fork","knife","spoon","bowl","banana","apple","sandwich","orange",
  "broccoli","carrot","hot dog","pizza","donut","cake","chair","couch","potted plant","bed",
  "dining table","toilet","tv","laptop","mouse","remote","keyboard","cell phone","microwave","oven",
  "toaster","sink","refrigerator","book","clock","vase","scissors","teddy bear","hair drier","toothbrush"};

// load engine from a .pt (falls back to .bin export) + depths
static ProviderU load_model(const std::string& weights, std::vector<int64_t>& dep) {
  { std::ifstream f(DN + "depths.txt"); int64_t v; while (f >> v) dep.push_back(v); }
  std::ifstream t(weights);
  if (t.good()) { printf("weights <- %s (pure C++)\n", weights.c_str()); return load_net_unfused_pt(DN, weights); }
  return load_net_unfused(DN);
}

// forward one (1,3,S,S) image -> packed pd (A,4*RM) / ps (A,NC0), A anchors
static void forward_img(const Tensor& x, ProviderU& prov, const std::vector<int64_t>& dep,
                        int64_t A, Tensor& pd, Tensor& ps) {
  prov.i = 0; auto h = yolov8n_forward_u(x, prov, false, dep);
  std::vector<Tensor> bx = {h[0].first,h[1].first,h[2].first}, cs = {h[0].second,h[1].second,h[2].second};
  pd = pack_levels(bx, 1, A, 4*RM); ps = pack_levels(cs, 1, A, NC0);
}

// ---- shared: run val mAP over a Dataset (letterbox eval, format-aware GT) ----
static double run_val(Dataset& va, ProviderU& prov, const std::vector<int64_t>& dep, int64_t S) {
  std::vector<float> axv, ayv, sv; make_anchors(S, axv, ayv, sv); int64_t A = (int64_t)sv.size();
  std::vector<mapeval::Image> imgs;
  for (auto& s : va.items) {
    Letterbox lb; auto xi = load_image_letterbox(s.img, S, lb);
    Tensor pd, ps; forward_img(xi, prov, dep, A, pd, ps);
    std::vector<float> pred; decode_predictions(pd->data, ps->data, axv, ayv, sv, A, NC0, RM, pred);
    auto dets = nms(pred, A, NC0, 0.001f, 0.7f, 300);
    mapeval::Image im;
    for (auto& d : dets) im.dts.push_back({d.x1,d.y1,d.x2,d.y2,d.cls,d.conf});
    std::vector<float> gb; std::vector<int64_t> gl; int m = load_boxes_orig(s.lbl, va.yolo, lb.w0, lb.h0, gb, gl); lb_map(gb, lb);
    for (int j = 0; j < m; ++j) im.gts.push_back({gb[j*4],gb[j*4+1],gb[j*4+2],gb[j*4+3],(int)gl[j]});
    imgs.push_back(im);
  }
  auto mp = mapeval::coco_map(imgs);
  printf("val: mAP@0.5 %.4f   mAP@0.5:0.95 %.4f   (%zu images)\n", mp.first, mp.second, va.items.size());
  return mp.first;
}

// ============================== train ==============================
static int cmd_train(const Args& a) {
  DataYaml dy = parse_yaml(a.get("data"));
  int64_t S = a.geti("imgsz", 640);
  int EPOCHS = a.geti("epochs", 100), BATCH = a.geti("batch", 16);
  std::string weights = a.get("weights", "yolov8n.pt");
  AugCfg baseAug; baseAug.mosaic = a.geti("mosaic", 1) != 0; baseAug.mixup = a.geti("mixup", 1) != 0;
  baseAug.hsv = a.geti("hsv", 1) != 0; baseAug.affine = a.geti("affine", 1) != 0; baseAug.flip = a.geti("flip", 1) != 0;
  int closeMosaic = a.geti("close-mosaic", std::max(1, EPOCHS/10));
  if (dy.nc != NC0) printf("[warn] data.yaml nc=%d but the head is %lld-class; class ids must be < %lld "
                           "(custom-nc head re-init is a RESUME item)\n", dy.nc, (long long)NC0, (long long)NC0);

  Dataset tr = read_yolo_dataset(join(dy.path, dy.train), S);
  Dataset va = read_yolo_dataset(join(dy.path, dy.val),   S);
  printf("train=%zu val=%zu imgsz=%lld batch=%d epochs=%d mosaic=%d mixup=%d close-mosaic=%d\n",
         tr.items.size(), va.items.size(), (long long)S, BATCH, EPOCHS, (int)baseAug.mosaic, (int)baseAug.mixup, closeMosaic);

  std::vector<int64_t> dep; auto prov = load_model(weights, dep);
  std::vector<Tensor> params; for (auto& L : prov.layers) { params.push_back(L.w); if (L.kind==1){params.push_back(L.gamma);params.push_back(L.beta);} else params.push_back(L.b); }
  Adam opt(params, 2e-3f, 0.9f, 0.999f, 1e-8f, 5e-4f, false);

  struct Lv { int64_t h,w; float s; }; std::vector<Lv> lv = {{S/8,S/8,8.f},{S/16,S/16,16.f},{S/32,S/32,32.f}};
  std::vector<float> ax,ay,ss,anc_img; for (auto& L:lv) for (int64_t y=0;y<L.h;++y) for (int64_t x=0;x<L.w;++x){ax.push_back(x+.5f);ay.push_back(y+.5f);ss.push_back(L.s);anc_img.push_back((x+.5f)*L.s);anc_img.push_back((y+.5f)*L.s);}
  int64_t A = (int64_t)ss.size();

  std::vector<std::string> names; { std::ifstream f(DN + "names.txt"); std::string s; while (f >> s) names.push_back(s); }
  auto save_ckpt = [&](const std::string& path) {
    std::vector<pt::Tensor> ck; size_t k = 0;
    auto push = [&](const std::vector<float>& d, const std::vector<int64_t>& shp){ pt::Tensor t; if (k<names.size()) t.name=names[k]; t.shape=shp; t.data=d; ck.push_back(t); ++k; };
    for (auto& L : prov.layers) { std::vector<int64_t> ws(L.w->shape.begin(),L.w->shape.end()); push(L.w->data, ws);
      if (L.kind==1){ std::vector<int64_t> c={L.gamma->shape[0]}; push(L.gamma->data,c); push(L.beta->data,c); push(L.rm,c); push(L.rv,c); }
      else push(L.b->data, {L.b->shape[0]}); }
    pt::save_pt(ck, path);
  };

  std::vector<int> order(tr.items.size()); std::iota(order.begin(), order.end(), 0);
  std::mt19937 rng(0);
  int total = EPOCHS * (((int)tr.items.size()+BATCH-1)/BATCH), gstep = 0; double best = -1;
  for (int ep = 0; ep < EPOCHS; ++ep) {
    AugCfg aug = baseAug; if (ep >= EPOCHS - closeMosaic) { aug.mosaic = false; aug.mixup = false; }
    std::shuffle(order.begin(), order.end(), rng); double eloss = 0; int nb = 0;
    for (size_t off = 0; off < order.size(); off += BATCH) {
      std::vector<int> idx(order.begin()+off, order.begin()+std::min(order.size(), off+BATCH));
      Batch bt = load_minibatch(tr, idx, true, rng(), aug);
      int64_t B = bt.B, R = B*A, Mx = bt.M;
      std::vector<float> ancx(R),ancy(R),strd(R); for (int64_t r=0;r<R;++r){int64_t q=r%A;ancx[r]=ax[q];ancy[r]=ay[q];strd[r]=ss[q];}
      prov.i=0; auto h = yolov8n_forward_u(bt.x, prov, true, dep);
      std::vector<Tensor> bx={h[0].first,h[1].first,h[2].first}, cs={h[0].second,h[1].second,h[2].second};
      auto pd=pack_levels(bx,B,A,4*RM); auto ps=pack_levels(cs,B,A,NC0);
      std::vector<float> pdb,pss; { pdb.assign(R*4,0); pss.assign(R*NC0,0);
        for (int64_t r=0;r<R;++r){ for(int j=0;j<4;++j){float mx=-1e30f;for(int k=0;k<RM;++k)mx=std::max(mx,pd->data[r*64+j*RM+k]);double sm=0;float dd=0;std::vector<float>e(RM);for(int k=0;k<RM;++k){e[k]=std::exp(pd->data[r*64+j*RM+k]-mx);sm+=e[k];}for(int k=0;k<RM;++k)dd+=(float)(e[k]/sm)*k;pdb[r*4+j]=dd;}
          float axr=ax[r%A],ayr=ay[r%A],st=ss[r%A],l=pdb[r*4],t=pdb[r*4+1],rr=pdb[r*4+2],bb=pdb[r*4+3];pdb[r*4]=(axr-l)*st;pdb[r*4+1]=(ayr-t)*st;pdb[r*4+2]=(axr+rr)*st;pdb[r*4+3]=(ayr+bb)*st;
          for(int64_t c=0;c<NC0;++c)pss[r*NC0+c]=1.f/(1.f+std::exp(-ps->data[r*NC0+c])); } }
      auto tal = tal_assign(pss,pdb,anc_img, bt.gt_labels, bt.gt_boxes, bt.mask, B,A,Mx,NC0,10,0.5f,6.0f);
      auto Lo = pure_v8_loss(pd,ps,ancx,ancy,strd,tal.tb,tal.ts,R,NC0,RM);
      backward(Lo.total);
      opt.lr = cosine_lr(gstep, total, 2e-3f, std::max(1,total/20)); opt.step(); ++gstep;
      eloss += Lo.total->data[0]; ++nb;
    }
    Dataset vac = va; double m50 = run_val(vac, prov, dep, S);
    printf("epoch %d/%d  loss %.3f  (val above)%s\n", ep+1, EPOCHS, eloss/nb, m50>best?"  *best*":"");
    save_ckpt("last.pt"); if (m50 > best) { best = m50; save_ckpt("best.pt"); }
  }
  printf("done. best val mAP@0.5 = %.4f. wrote last.pt / best.pt (pure C++)\n", best);
  return 0;
}

// ============================== val ==============================
static int cmd_val(const Args& a) {
  DataYaml dy = parse_yaml(a.get("data"));
  int64_t S = a.geti("imgsz", 640);
  std::vector<int64_t> dep; auto prov = load_model(a.get("weights", "yolov8n.pt"), dep);
  Dataset va = read_yolo_dataset(join(dy.path, dy.val), S);
  run_val(va, prov, dep, S);
  return 0;
}

// ============================== detect ==============================
static int cmd_detect(const Args& a) {
  DataYaml dy; if (!a.get("data").empty()) dy = parse_yaml(a.get("data"));
  int64_t S = a.geti("imgsz", 640); float conf = a.getf("conf", 0.25f), iou = a.getf("iou", 0.7f);
  std::string src = a.get("source"), outp = a.get("out", "out.png");
  std::vector<int64_t> dep; auto prov = load_model(a.get("weights", "yolov8n.pt"), dep);
  int w0,h0,ch; unsigned char* im = stbi_load(src.c_str(), &w0,&h0,&ch,3);
  if (!im) { printf("cannot load %s\n", src.c_str()); return 1; }
  Letterbox lb; auto x = load_image_letterbox(src, S, lb);
  std::vector<float> axv,ayv,sv; make_anchors(S, axv,ayv,sv); int64_t A=(int64_t)sv.size();
  Tensor pd, ps; forward_img(x, prov, dep, A, pd, ps);
  std::vector<float> pred; decode_predictions(pd->data, ps->data, axv,ayv,sv, A, NC0, RM, pred);
  auto dets = nms(pred, A, NC0, conf, iou, 300);
  auto put=[&](int px,int py,unsigned char r,unsigned char g,unsigned char b){ if(px<0||py<0||px>=w0||py>=h0)return; unsigned char*p=&im[(py*w0+px)*3];p[0]=r;p[1]=g;p[2]=b; };
  printf("%zu detections:\n", dets.size());
  for (auto& d : dets) {
    int x1=(int)std::round((d.x1-lb.left)/lb.r), y1=(int)std::round((d.y1-lb.top)/lb.r);
    int x2=(int)std::round((d.x2-lb.left)/lb.r), y2=(int)std::round((d.y2-lb.top)/lb.r);
    x1=std::clamp(x1,0,w0-1);y1=std::clamp(y1,0,h0-1);x2=std::clamp(x2,0,w0-1);y2=std::clamp(y2,0,h0-1);
    for(int t=0;t<3;++t){for(int px=x1;px<=x2;++px){put(px,y1+t,255,60,60);put(px,y2-t,255,60,60);}for(int py=y1;py<=y2;++py){put(x1+t,py,255,60,60);put(x2-t,py,255,60,60);}}
    const char* nm = (d.cls < (int)dy.names.size()) ? dy.names[d.cls].c_str() : (d.cls<80?COCO[d.cls]:"?");
    printf("  %-14s conf=%.2f  xyxy=(%d,%d,%d,%d)\n", nm, d.conf, x1,y1,x2,y2);
  }
  if (!stbi_write_png(outp.c_str(), w0,h0,3, im, w0*3)) { printf("write failed\n"); return 1; }
  printf("wrote %s\n", outp.c_str()); stbi_image_free(im); return 0;
}

int main(int argc, char** argv) {
  setvbuf(stdout, nullptr, _IONBF, 0);
  std::string cmd = argc > 1 ? argv[1] : "";
  Args a = parse_args(argc, argv, 2);
  if (cmd == "train")  return cmd_train(a);
  if (cmd == "val")    return cmd_val(a);
  if (cmd == "detect") return cmd_detect(a);
  if (cmd == "export") { printf("export: use `onnx_export <model>` (onnxruntime-verified). "
                                "Unifying ONNX-export-from-.pt into this CLI is a RESUME item.\n"); return 0; }
  printf("usage: yolo <train|val|detect|export> --flags   (see header of pure/yolo.cpp)\n");
  return 1;
}
