"""Inference reference for the pure track. Letterboxes a real image exactly like
Ultralytics, runs yolov8n eval forward to get decoded predictions (1, 4+nc, Atot),
and runs Ultralytics NMS. Dumps the preprocessed input, the decoded predictions,
and the final detections so the pure C++ can be checked bit-for-bit on decode and
match on NMS. Usage: python m6_infer_ref.py [imgsz] [image]"""
import os, sys, numpy as np, torch, cv2
from ultralytics import YOLO
from ultralytics.data.augment import LetterBox
from ultralytics.utils import ops
try:
    from ultralytics.utils.nms import non_max_suppression
except ImportError:                       # older layout
    from ultralytics.utils.ops import non_max_suppression

HERE = os.path.dirname(os.path.abspath(__file__))
D = os.path.join(HERE, "data_infer"); os.makedirs(D, exist_ok=True)
S    = int(sys.argv[1]) if len(sys.argv) > 1 else 640
IMGP = sys.argv[2] if len(sys.argv) > 2 else os.path.join(HERE, "assets", "bus.jpg")
CONF, IOU = 0.25, 0.7

ym = YOLO("yolov8n.pt")
dm = ym.model.float().eval()
names = ym.names

im0 = cv2.imread(IMGP)                    # BGR HxWx3
h0, w0 = im0.shape[:2]
lb = LetterBox((S, S), auto=False, scaleup=True, stride=32)
im = lb(image=im0)                        # SxSx3 BGR uint8
# letterbox params (to map boxes back to the original image in the demo)
r = min(S / h0, S / w0)
new_unpad = (round(w0 * r), round(h0 * r))
dw = (S - new_unpad[0]) / 2; dh = (S - new_unpad[1]) / 2

x = im[:, :, ::-1].transpose(2, 0, 1)[None]          # BGR->RGB, HWC->CHW, add batch
x = np.ascontiguousarray(x).astype(np.float32) / 255.0
xt = torch.from_numpy(x)

with torch.no_grad():
    out = dm(xt)
pred = out[0] if isinstance(out, (list, tuple)) else out      # (1, 4+nc, Atot)
nc = len(names); Atot = pred.shape[2]
assert pred.shape[1] == 4 + nc, pred.shape

# raw head logits on the SAME input (train-mode dict, BN forced to eval) so the pure
# forward can be checked bit-for-bit and the decode math can be checked on reference
# logits independent of any tiny forward numerical drift (which DFL amplifies).
dmt = ym.model.float(); dmt.train()
for mod in dmt.modules():
    if isinstance(mod, torch.nn.BatchNorm2d):
        mod.eval()
with torch.no_grad():
    hd = dmt(xt)
head_boxes, head_scores = hd["boxes"], hd["scores"]          # (1, 4*rm, Atot), (1, nc, Atot)

dets = non_max_suppression(pred, CONF, IOU, agnostic=False, max_det=300)[0]  # (n,6) xyxy,conf,cls

def save(n, t):
    (t.detach().cpu().numpy() if torch.is_tensor(t) else np.asarray(t)).astype(np.float32).tofile(os.path.join(D, n))
save("x.bin", xt)
save("ref_pred.bin", pred[0])            # (4+nc, Atot) channel-major
save("ref_head_boxes.bin", head_boxes[0])    # (4*rm, Atot)
save("ref_head_scores.bin", head_scores[0])  # (nc, Atot)
lines = [f"{S} {nc} {Atot} {CONF} {IOU}",
         f"{w0} {h0} {r} {dw} {dh}"]
lines.append(str(dets.shape[0]))
for d in dets.cpu().numpy():
    x1, y1, x2, y2, cf, cl = d
    lines.append(f"{x1:.4f} {y1:.4f} {x2:.4f} {y2:.4f} {cf:.6f} {int(cl)}")
open(os.path.join(D, "meta.txt"), "w").write("\n".join(lines) + "\n")
# class names for the C++ demo
open(os.path.join(D, "names.txt"), "w", encoding="utf-8").write(
    "\n".join(names[i] for i in range(nc)) + "\n")

print(f"imgsz={S} nc={nc} Atot={Atot} dets={dets.shape[0]}")
for d in dets.cpu().numpy():
    print(f"  {names[int(d[5])]:12s} conf={d[4]:.3f} xyxy=({d[0]:.1f},{d[1]:.1f},{d[2]:.1f},{d[3]:.1f})")
