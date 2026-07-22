"""Export yolov8n (BN-folded) convs in the exact order the pure C++ forward
consumes them, plus a fixed input and reference boxes/scores. Works with stock
Ultralytics (no patched _predict_once). Usage: python export_net.py [imgsz]"""
import os, sys, torch
from ultralytics import YOLO

HERE = os.path.dirname(os.path.abspath(__file__))
D = os.path.join(HERE, "data_net"); os.makedirs(D, exist_ok=True)
IMG = int(sys.argv[1]) if len(sys.argv) > 1 else 64

convs = []   # each: (w, b, k, stride, act)
def fuse(cv):
    conv, bn = cv.conv, cv.bn
    std = torch.sqrt(bn.running_var + bn.eps)
    w = conv.weight * (bn.weight / std).reshape(-1, 1, 1, 1)
    b = bn.bias - bn.weight * bn.running_mean / std
    return w, b
def emit(mod):
    if hasattr(mod, "bn"):                       # ultralytics Conv (conv+bn+silu)
        w, b = fuse(mod); k = mod.conv.kernel_size[0]; s = mod.conv.stride[0]; act = 1
    else:                                        # plain nn.Conv2d (detect tail)
        w, b = mod.weight, mod.bias; k = mod.kernel_size[0]; s = mod.stride[0]; act = 0
    convs.append((w, b, k, s, act))
def emit_c2f(blk):
    emit(blk.cv1)
    for bott in blk.m: emit(bott.cv1); emit(bott.cv2)
    emit(blk.cv2)
def emit_sppf(blk):
    emit(blk.cv1); emit(blk.cv2)

torch.manual_seed(0)
ym = YOLO("yolov8n.pt")
L = ym.model.model.eval()

emit(L[0]); emit(L[1]); emit_c2f(L[2]); emit(L[3]); emit_c2f(L[4])
emit(L[5]); emit_c2f(L[6]); emit(L[7]); emit_c2f(L[8]); emit_sppf(L[9])
emit_c2f(L[12]); emit_c2f(L[15]); emit(L[16]); emit_c2f(L[18]); emit(L[19]); emit_c2f(L[21])
det = L[22]
nc, rm = det.nc, det.reg_max
for i in range(3):
    for m in det.cv2[i]: emit(m)
    for m in det.cv3[i]: emit(m)

import numpy as np
def save(n, t): t.detach().contiguous().float().cpu().numpy().tofile(os.path.join(D, n))
lines = [str(len(convs))]
blob = []
for i, (w, b, k, s, act) in enumerate(convs):
    save(f"w{i}.bin", w); save(f"b{i}.bin", b)
    blob.append(w.detach().contiguous().float().cpu().numpy().ravel())
    blob.append(b.detach().contiguous().float().cpu().numpy().ravel())
    lines.append(f"{w.shape[0]} {w.shape[1]} {k} {s} {act}")
open(os.path.join(D, "manifest.txt"), "w").write("\n".join(lines) + "\n")
# single packed blob (manifest.txt + weights.bin) for the shipped, Python-free demo
np.concatenate(blob).astype(np.float32).tofile(os.path.join(D, "weights.bin"))

# reference boxes/scores. In train mode this Ultralytics head returns a dict with
# 'boxes' (1, 4*rm, Atot) and 'scores' (1, nc, Atot) already in the packed anchor
# layout the C++ expects. BN forced to eval so it uses running stats (matches folded).
x = torch.randn(1, 3, IMG, IMG)
dm = ym.model.float(); dm.train()
for mod in dm.modules():
    if isinstance(mod, torch.nn.BatchNorm2d):
        mod.eval()
with torch.no_grad():
    y = dm(x)
boxes, scores = y["boxes"], y["scores"]     # (1, 4*rm, Atot), (1, nc, Atot)
nb, Atot = boxes.shape[1], boxes.shape[2]
save("x.bin", x); save("ref_boxes.bin", boxes); save("ref_scores.bin", scores)
open(os.path.join(D, "io.txt"), "w").write(f"{IMG} {nb} {nc} {rm} {Atot}\n")
print(f"convs={len(convs)} imgsz={IMG} nb={nb} nc={nc} rm={rm} Atot={Atot}")
