"""Generate deterministic inputs + reference outputs (DFL decode + CIoU).
Writes raw float32 .bin files that the C++ program reads/compares against.
Math mirrors Ultralytics: DFL softmax-over-bins, dist2bbox, bbox_iou(CIoU=True)."""
import os, math, struct
import torch

torch.manual_seed(0)
REG_MAX = 16
N = 8

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data")
OUT = os.path.join(HERE, "..", "out")
os.makedirs(DATA, exist_ok=True)
os.makedirs(OUT, exist_ok=True)

def save(name, t):
    t.contiguous().to(torch.float32).cpu().numpy().tofile(os.path.join(DATA, name))

# --- inputs ---
pred_dist = torch.randn(N, 4 * REG_MAX)          # raw box distribution logits
anchors = torch.rand(N, 2) * 20 + 5              # anchor points (grid units)
strides = torch.full((N, 1), 8.0)               # single stride level
gt = torch.tensor([                              # target boxes (xyxy, image units)
    [10, 10, 50, 60], [0, 0, 30, 30], [100, 100, 140, 180], [5, 5, 25, 45],
    [60, 20, 90, 70], [15, 15, 55, 35], [200, 150, 260, 220], [8, 8, 40, 40],
], dtype=torch.float32)

save("pred_dist.bin", pred_dist)
save("anchors.bin", anchors)
save("strides.bin", strides)
save("gt.bin", gt)

# --- reference DFL decode ---
proj = torch.arange(REG_MAX, dtype=torch.float32)
prob = pred_dist.view(N, 4, REG_MAX).softmax(2)
dist = (prob * proj.view(1, 1, REG_MAX)).sum(2)          # (N,4) ltrb
lt, rb = dist[:, :2], dist[:, 2:]
boxes = torch.cat([anchors - lt, anchors + rb], 1) * strides  # xyxy

# --- reference CIoU (paired) ---
def ciou(b1, b2, eps=1e-7):
    x1, y1, x2, y2 = b1.unbind(1)
    X1, Y1, X2, Y2 = b2.unbind(1)
    w1, h1 = (x2 - x1).clamp(min=0), (y2 - y1).clamp(min=0)
    w2, h2 = (X2 - X1).clamp(min=0), (Y2 - Y1).clamp(min=0)
    inter = (torch.min(x2, X2) - torch.max(x1, X1)).clamp(min=0) * \
            (torch.min(y2, Y2) - torch.max(y1, Y1)).clamp(min=0)
    uni = w1 * h1 + w2 * h2 - inter + eps
    iou = inter / uni
    cw = torch.max(x2, X2) - torch.min(x1, X1)
    ch = torch.max(y2, Y2) - torch.min(y1, Y1)
    c2 = cw * cw + ch * ch + eps
    rho2 = ((x1 + x2) / 2 - (X1 + X2) / 2) ** 2 + ((y1 + y2) / 2 - (Y1 + Y2) / 2) ** 2
    v = (4 / math.pi ** 2) * (torch.atan(w2 / (h2 + eps)) - torch.atan(w1 / (h1 + eps))) ** 2
    with torch.no_grad():
        alpha = v / (v - iou + (1 + eps))
    return iou - (rho2 / c2 + v * alpha)

c = ciou(boxes, gt)

boxes.contiguous().numpy().tofile(os.path.join(OUT, "ref_boxes.bin"))
c.contiguous().numpy().tofile(os.path.join(OUT, "ref_ciou.bin"))
print(f"generated inputs (N={N}) and reference outputs")
print("ref boxes[0] =", boxes[0].tolist())
print("ref ciou     =", [round(x, 5) for x in c.tolist()])
