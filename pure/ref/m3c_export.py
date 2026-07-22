"""M3c: dump ALL of yolov8n's convs (BN-folded) in the exact order the pure C++
forward consumes them, plus a fixed input and reference boxes/scores."""
import os, torch
from ultralytics import YOLO

HERE = os.path.dirname(os.path.abspath(__file__))
D = os.path.join(HERE, "data_net"); os.makedirs(D, exist_ok=True)

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
for i in range(3):
    for m in det.cv2[i]: emit(m)
    for m in det.cv3[i]: emit(m)

def save(n, t): t.detach().contiguous().float().cpu().numpy().tofile(os.path.join(D, n))
lines = [str(len(convs))]
for i, (w, b, k, s, act) in enumerate(convs):
    save(f"w{i}.bin", w); save(f"b{i}.bin", b)
    lines.append(f"{w.shape[0]} {w.shape[1]} {k} {s} {act}")
open(os.path.join(D, "manifest.txt"), "w").write("\n".join(lines) + "\n")

# fixed input + reference boxes/scores. Detect must stay in training mode to return
# the raw boxes/scores dict, but BN must use running stats (to match the folded convs).
x = torch.randn(1, 3, 64, 64)
dm = ym.model.float(); dm.train()
for mod in dm.modules():
    if isinstance(mod, torch.nn.BatchNorm2d):
        mod.eval()
with torch.no_grad():
    y = dm._predict_once(x)
save("x.bin", x); save("ref_boxes.bin", y["boxes"]); save("ref_scores.bin", y["scores"])
open(os.path.join(D, "io.txt"), "w").write(
    f"{tuple(y['boxes'].shape)} {tuple(y['scores'].shape)}\n")
print("convs:", len(convs), " boxes:", tuple(y["boxes"].shape), " scores:", tuple(y["scores"].shape))
