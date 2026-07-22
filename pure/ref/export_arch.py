"""Export a yolov8 model's ARCHITECTURE (layer graph) plus fused weights and a
reference forward, so the pure engine can build any size (n/s/m/l/x) from data instead
of hardcoded topology. Writes to data_arch_<model>/. Usage: python export_arch.py [model] [imgsz]
  model: yolov8n | yolov8s | yolov8m | yolov8l | yolov8x  (default yolov8n)"""
import os, sys, torch
from ultralytics import YOLO

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL = sys.argv[1] if len(sys.argv) > 1 else "yolov8n"
IMG   = int(sys.argv[2]) if len(sys.argv) > 2 else 64
D = os.path.join(HERE, f"data_arch_{MODEL}"); os.makedirs(D, exist_ok=True)

ym = YOLO(f"{MODEL}.pt")
seq = ym.model.model.eval()

# --- fused conv dump, in canonical (block, sub-conv) order ---
convs = []
def fuse(cv):
    conv, bn = cv.conv, cv.bn
    std = torch.sqrt(bn.running_var + bn.eps)
    return conv.weight * (bn.weight / std).reshape(-1, 1, 1, 1), bn.bias - bn.weight * bn.running_mean / std
def emit(mod):
    if hasattr(mod, "bn"): w, b = fuse(mod); k, s, act = mod.conv.kernel_size[0], mod.conv.stride[0], 1
    else:                  w, b = mod.weight, mod.bias; k, s, act = mod.kernel_size[0], mod.stride[0], 0
    convs.append((w, b, k, s, act))
def emit_c2f(b):
    emit(b.cv1)
    for bott in b.m: emit(bott.cv1); emit(bott.cv2)
    emit(b.cv2)
def emit_sppf(b): emit(b.cv1); emit(b.cv2)

# --- architecture graph (type / from / params), one line per layer ---
# type codes: 0 Conv, 1 C2f, 2 SPPF, 3 Upsample, 4 Concat, 5 Detect
arch = []
det = None
for i, m in enumerate(seq):
    t = type(m).__name__
    f = m.f if isinstance(m.f, list) else [m.f]
    fs = " ".join(str(v) for v in f)
    if t == "Conv":     emit(m);      arch.append(f"{i} 0 {len(f)} {fs}")
    elif t == "C2f":    emit_c2f(m);  arch.append(f"{i} 1 {len(f)} {fs} {len(m.m)} {int(m.m[0].add)}")
    elif t == "SPPF":   emit_sppf(m); arch.append(f"{i} 2 {len(f)} {fs}")
    elif t == "Upsample": arch.append(f"{i} 3 {len(f)} {fs} {int(m.scale_factor)}")
    elif t == "Concat":   arch.append(f"{i} 4 {len(f)} {fs}")
    elif t == "Detect":
        det = m
        for j in range(3):
            for mm in m.cv2[j]: emit(mm)
            for mm in m.cv3[j]: emit(mm)
        arch.append(f"{i} 5 {len(f)} {fs} {m.nc} {m.reg_max}")
    else: raise SystemExit(f"unhandled layer {t}")

def save(n, t): t.detach().contiguous().float().cpu().numpy().tofile(os.path.join(D, n))
lines = [str(len(convs))]
for i, (w, b, k, s, act) in enumerate(convs):
    save(f"w{i}.bin", w); save(f"b{i}.bin", b)
    lines.append(f"{w.shape[0]} {w.shape[1]} {k} {s} {act}")
open(os.path.join(D, "manifest.txt"), "w").write("\n".join(lines) + "\n")
open(os.path.join(D, "arch.txt"), "w").write(f"{len(arch)} {det.nc} {det.reg_max}\n" + "\n".join(arch) + "\n")

# fixed input + reference boxes/scores (train-dict head, BN forced to eval)
torch.manual_seed(0)
x = torch.randn(1, 3, IMG, IMG)
dm = ym.model.float(); dm.train()
for mod in dm.modules():
    if isinstance(mod, torch.nn.BatchNorm2d): mod.eval()
with torch.no_grad(): y = dm(x)
save("x.bin", x); save("ref_boxes.bin", y["boxes"]); save("ref_scores.bin", y["scores"])
open(os.path.join(D, "io.txt"), "w").write(f"{IMG} {y['boxes'].shape[1]} {y['scores'].shape[1]} {det.reg_max} {y['boxes'].shape[2]}\n")
print(f"{MODEL}: {len(arch)} layers, {len(convs)} convs, imgsz={IMG}, Atot={y['boxes'].shape[2]}")
