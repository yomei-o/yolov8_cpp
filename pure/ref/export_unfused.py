"""Export yolov8n WITHOUT folding BN, in canonical order, so the pure engine can run
conv + BatchNorm2d + SiLU as separate ops (needed for BN training and .pt write-back).
Writes into data_net/ alongside the fused export and reuses its x.bin / ref_*.bin.
Usage: python export_unfused.py [imgsz]"""
import os, sys, torch
from ultralytics import YOLO
from yolo_walk import walk

HERE = os.path.dirname(os.path.abspath(__file__))
D = os.path.join(HERE, "data_net"); os.makedirs(D, exist_ok=True)
IMG = int(sys.argv[1]) if len(sys.argv) > 1 else 64

MODEL = sys.argv[2] if len(sys.argv) > 2 else "yolov8n"
ym = YOLO(MODEL + ".pt")
ym.model.eval()
mods = walk(ym.model)
depths = [len(mm.m) for mm in ym.model.modules() if type(mm).__name__ == "C2f"]

def save(n, t): t.detach().contiguous().float().cpu().numpy().tofile(os.path.join(D, n))
lines = [str(len(mods))]
for i, m in enumerate(mods):
    if hasattr(m, "bn"):                                  # Conv = conv(no bias)+BN+SiLU
        conv, bn = m.conv, m.bn
        save(f"cw{i}.bin", conv.weight)
        save(f"bg{i}.bin", bn.weight); save(f"bb{i}.bin", bn.bias)
        save(f"rm{i}.bin", bn.running_mean); save(f"rv{i}.bin", bn.running_var)
        Co, Ci, k, s = conv.weight.shape[0], conv.weight.shape[1], conv.kernel_size[0], conv.stride[0]
        lines.append(f"1 {Co} {Ci} {k} {s} {bn.eps}")     # kind 1 = conv+bn+act
    else:                                                 # plain nn.Conv2d (bias, no act)
        save(f"cw{i}.bin", m.weight); save(f"cb{i}.bin", m.bias)
        Co, Ci, k, s = m.weight.shape[0], m.weight.shape[1], m.kernel_size[0], m.stride[0]
        lines.append(f"0 {Co} {Ci} {k} {s} 0")            # kind 0 = plain conv
open(os.path.join(D, "manifest_unfused.txt"), "w").write("\n".join(lines) + "\n")
open(os.path.join(D, "depths.txt"), "w").write(" ".join(map(str, depths)) + "\n")

# reuse fused export's input + reference if present; otherwise (re)generate them here.
if not os.path.exists(os.path.join(D, "ref_boxes.bin")):
    torch.manual_seed(0)
    x = torch.randn(1, 3, IMG, IMG)
    dm = ym.model.float(); dm.train()
    for mod in dm.modules():
        if isinstance(mod, torch.nn.BatchNorm2d): mod.eval()
    with torch.no_grad(): y = dm(x)
    save("x.bin", x); save("ref_boxes.bin", y["boxes"]); save("ref_scores.bin", y["scores"])
    open(os.path.join(D, "io.txt"), "w").write(f"{IMG} {y['boxes'].shape[1]} {y['scores'].shape[1]} 16 {y['boxes'].shape[2]}\n")

print(f"unfused: {len(mods)} layers  imgsz={IMG}")
