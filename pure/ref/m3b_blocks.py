"""M3b: export a real C2f block and a real SPPF block from yolov8n (all sub-Convs
BN-folded, in canonical order) plus reference input/output, for pure-engine parity."""
import os, torch
from ultralytics import YOLO

HERE = os.path.dirname(os.path.abspath(__file__))

def fuse(cv):
    conv, bn = cv.conv, cv.bn
    std = torch.sqrt(bn.running_var + bn.eps)
    w = conv.weight * (bn.weight / std).reshape(-1, 1, 1, 1)
    b = bn.bias - bn.weight * bn.running_mean / std
    return w, b

def dump(dirn, convs, n_bott, x, ref):
    D = os.path.join(HERE, dirn); os.makedirs(D, exist_ok=True)
    def save(n, t): t.detach().contiguous().float().cpu().numpy().tofile(os.path.join(D, n))
    save("x.bin", x); save("ref.bin", ref)
    lines = [f"{n_bott} {len(convs)} {x.shape[1]} {x.shape[2]} {x.shape[3]}"]
    for i, (w, b) in enumerate(convs):
        save(f"w{i}.bin", w); save(f"b{i}.bin", b)
        lines.append(f"{w.shape[0]} {w.shape[1]} {w.shape[2]}")
    open(os.path.join(D, "meta.txt"), "w").write("\n".join(lines) + "\n")
    print(dirn, "convs=", len(convs), "n_bott=", n_bott, "ref", tuple(ref.shape))

torch.manual_seed(0)
model = YOLO("yolov8n.pt").model.model

# --- C2f block (index 2) ---
c2f = model[2].eval()
convs = [fuse(c2f.cv1)]
for bott in c2f.m:
    convs += [fuse(bott.cv1), fuse(bott.cv2)]
convs.append(fuse(c2f.cv2))
Cin = c2f.cv1.conv.in_channels
x = torch.randn(1, Cin, 16, 16)
with torch.no_grad(): ref = c2f(x)
dump("data_c2f", convs, len(c2f.m), x, ref)
# report shortcut flags
print("  bottleneck add flags:", [bool(b.add) for b in c2f.m])

# --- SPPF block (index 9) ---
sppf = model[9].eval()
convs = [fuse(sppf.cv1), fuse(sppf.cv2)]
Cin = sppf.cv1.conv.in_channels
x = torch.randn(1, Cin, 4, 4)
with torch.no_grad(): ref = sppf(x)
dump("data_sppf", convs, 0, x, ref)
print("  SPPF k =", sppf.m.kernel_size if hasattr(sppf.m, "kernel_size") else 5)
