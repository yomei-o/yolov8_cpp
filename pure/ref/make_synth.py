"""Generate a tiny synthetic detection dataset for the end-to-end train->infer test:
S x S images with a few solid colour rectangles (3 classes by colour), on grey(114).
Writes PNGs + YOLO label files ("cls x1 y1 x2 y2" in pixel coords; images are square so
letterboxing is identity) + list.txt for dataset.hpp's load_batch.
Usage: python make_synth.py [imgsz] [ntrain]"""
import os, sys, random
from PIL import Image, ImageDraw

HERE = os.path.dirname(os.path.abspath(__file__))
S     = int(sys.argv[1]) if len(sys.argv) > 1 else 128
NTR   = int(sys.argv[2]) if len(sys.argv) > 2 else 24
D = os.path.join(HERE, "data_synth"); os.makedirs(D, exist_ok=True)
random.seed(0)
COLORS = [(220, 40, 40), (40, 200, 40), (50, 90, 230)]   # class 0,1,2

def gen(name):
    im = Image.new("RGB", (S, S), (114, 114, 114)); dr = ImageDraw.Draw(im)
    lines = []
    for _ in range(random.randint(1, 3)):
        c = random.randint(0, 2)
        w = random.randint(S // 5, S // 3); h = random.randint(S // 5, S // 3)
        x1 = random.randint(2, S - w - 2); y1 = random.randint(2, S - h - 2)
        dr.rectangle([x1, y1, x1 + w, y1 + h], fill=COLORS[c])
        lines.append(f"{c} {x1} {y1} {x1 + w} {y1 + h}")
    im.save(os.path.join(D, name + ".png"))
    open(os.path.join(D, name + ".txt"), "w").write("\n".join(lines) + "\n")
    return name

names = [gen(f"tr{i:02d}") for i in range(NTR)]
for i in range(4): gen(f"te{i:02d}")
with open(os.path.join(D, "list.txt"), "w") as f:
    f.write(f"{S} {len(names)}\n")
    for n in names:
        f.write(f"{D.replace(chr(92),'/')}/{n}.png {D.replace(chr(92),'/')}/{n}.txt\n")
print(f"wrote {len(names)} train + 4 test images ({S}px) to {D}")
