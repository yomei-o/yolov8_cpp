"""Make pseudo-labels for real-data training: run Ultralytics on each image at imgsz S
and write detections as 'cls x1 y1 x2 y2' in letterboxed-S coords (the space the pure
trainer works in), plus a list.txt batch index. Usage: python make_labels.py [imgsz] [img ...]"""
import os, sys, cv2
from ultralytics import YOLO

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
S = int(sys.argv[1]) if len(sys.argv) > 1 else 160
imgs = sys.argv[2:] or [os.path.join(ROOT, "assets", "bus.jpg"), os.path.join(ROOT, "assets", "zidane.jpg")]
D = os.path.join(HERE, "data_train"); os.makedirs(D, exist_ok=True)

ym = YOLO("yolov8n.pt")
entries = []
for img in imgs:
    stem = os.path.splitext(os.path.basename(img))[0]
    res = ym.predict(img, imgsz=S, conf=0.25, verbose=False)[0]
    h0, w0 = cv2.imread(img).shape[:2]
    r = min(S / h0, S / w0); dw = (S - round(w0 * r)) / 2; dh = (S - round(h0 * r)) / 2
    lbl = os.path.join(D, f"{stem}.txt")
    lines = []
    for b in res.boxes:
        x1, y1, x2, y2 = b.xyxy[0].tolist()
        lines.append(f"{int(b.cls.item())} {x1*r+dw:.3f} {y1*r+dh:.3f} {x2*r+dw:.3f} {y2*r+dh:.3f}")
    open(lbl, "w").write("\n".join(lines) + "\n")
    entries.append((img.replace("\\", "/"), lbl.replace("\\", "/"), len(lines)))
    print(f"{stem}: {len(lines)} labels")

with open(os.path.join(D, "list.txt"), "w") as f:
    f.write(f"{S} {len(entries)}\n")
    for img, lbl, _ in entries:
        f.write(f"{img} {lbl}\n")
open(os.path.join(D, "meta.txt"), "w").write(f"{S}\n")
print(f"wrote list.txt with {len(entries)} images at imgsz {S}")
