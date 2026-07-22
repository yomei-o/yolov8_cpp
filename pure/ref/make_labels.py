"""Make pseudo-labels for real-data training: run Ultralytics on an image at imgsz S
and write the detections as 'cls x1 y1 x2 y2' in letterboxed-S image coordinates (the
space the pure trainer works in). Usage: python make_labels.py [imgsz] [image]"""
import os, sys, numpy as np
from ultralytics import YOLO

HERE = os.path.dirname(os.path.abspath(__file__))
S    = int(sys.argv[1]) if len(sys.argv) > 1 else 160
IMGP = sys.argv[2] if len(sys.argv) > 2 else os.path.join(HERE, "assets", "bus.jpg")
D = os.path.join(HERE, "data_train"); os.makedirs(D, exist_ok=True)

ym = YOLO("yolov8n.pt")
res = ym.predict(IMGP, imgsz=S, conf=0.25, verbose=False)[0]
import cv2
h0, w0 = cv2.imread(IMGP).shape[:2]
r = min(S / h0, S / w0)
dw = (S - round(w0 * r)) / 2; dh = (S - round(h0 * r)) / 2

lines = []
for b in res.boxes:
    cls = int(b.cls.item())
    x1, y1, x2, y2 = b.xyxy[0].tolist()           # original-image coords
    X1, Y1 = x1 * r + dw, y1 * r + dh              # -> letterboxed-S coords
    X2, Y2 = x2 * r + dw, y2 * r + dh
    lines.append(f"{cls} {X1:.3f} {Y1:.3f} {X2:.3f} {Y2:.3f}")
open(os.path.join(D, "labels.txt"), "w").write("\n".join(lines) + "\n")
open(os.path.join(D, "meta.txt"), "w").write(f"{S}\n")
print(f"wrote {len(lines)} labels at imgsz {S}:")
for l in lines: print("  ", l)
