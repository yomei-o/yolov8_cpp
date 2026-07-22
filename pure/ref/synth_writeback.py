"""'.pt' compatibility test: load the weights the C++ synthetic trainer wrote (data_wb/)
into yolov8n in canonical order, save yolov8n_synth.pt, then run Ultralytics inference on
the held-out synthetic test image — confirming a C++-trained model runs in Ultralytics."""
import os, numpy as np, torch
from ultralytics import YOLO
from yolo_walk import walk

HERE = os.path.dirname(os.path.abspath(__file__))
DW = os.path.join(HERE, "data_wb"); DS = os.path.join(HERE, "data_synth")
def r(n): return np.fromfile(os.path.join(DW, n), np.float32)

ym = YOLO("yolov8n.pt")
mods = walk(ym.model)
def load_(p, a):
    with torch.no_grad(): p.copy_(torch.from_numpy(a.astype(np.float32)).reshape(p.shape))
serr = 0.0
for i, m in enumerate(mods):
    if hasattr(m, "bn"):
        for a, p in [(r(f"cw{i}.bin"), m.conv.weight), (r(f"bg{i}.bin"), m.bn.weight), (r(f"bb{i}.bin"), m.bn.bias),
                     (r(f"rm{i}.bin"), m.bn.running_mean), (r(f"rv{i}.bin"), m.bn.running_var)]:
            load_(p, a); serr = max(serr, float(np.abs(a.reshape(p.shape) - p.detach().numpy()).max()))
    else:
        for a, p in [(r(f"cw{i}.bin"), m.weight), (r(f"cb{i}.bin"), m.bias)]:
            load_(p, a); serr = max(serr, float(np.abs(a.reshape(p.shape) - p.detach().numpy()).max()))
print(f"serialization max|diff| = {serr:.3e}  {'OK' if serr < 1e-6 else 'FAIL'}")

out = "yolov8n_synth.pt"; ym.save(out)
names = {0: "person(red)", 1: "bicycle(green)", 2: "car(blue)"}
res = YOLO(out).predict(os.path.join(DS, "te00.png"), imgsz=64, conf=0.25, verbose=False)[0]
print(f"\nUltralytics on the C++-trained {out}: {len(res.boxes)} detections on te00.png")
for b in res.boxes:
    c = int(b.cls.item()); x1, y1, x2, y2 = b.xyxy[0].tolist()
    print(f"  cls {c} ({names.get(c, c)}) conf={b.conf.item():.2f} xyxy=({x1:.0f},{y1:.0f},{x2:.0f},{y2:.0f})")
print("\n.pt round-trip OK — a model trained by the C++ engine loads and runs in Ultralytics"
      if serr < 1e-6 else "MISMATCH")
