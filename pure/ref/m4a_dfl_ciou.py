"""M4a reference: DFL decode + CIoU (same math as v8loss step1), for pure engine."""
import os, math, torch
HERE = os.path.dirname(os.path.abspath(__file__))
D = os.path.join(HERE, "data_m4a"); os.makedirs(D, exist_ok=True)
def save(n, t): t.detach().contiguous().float().numpy().tofile(os.path.join(D, n))

torch.manual_seed(1)
N, RM = 8, 16
pred_dist = torch.randn(N, 4 * RM)
anc = torch.rand(N, 2) * 20 + 5
stride = torch.full((N, 1), 8.0)
gt = torch.tensor([[10, 10, 50, 60], [0, 0, 30, 30], [100, 100, 140, 180], [5, 5, 25, 45],
                   [60, 20, 90, 70], [15, 15, 55, 35], [200, 150, 260, 220], [8, 8, 40, 40]],
                  dtype=torch.float32)
save("pred_dist.bin", pred_dist); save("anc.bin", anc); save("stride.bin", stride); save("gt.bin", gt)
open(os.path.join(D, "meta.txt"), "w").write(f"{N} {RM}\n")

proj = torch.arange(RM, dtype=torch.float32)
prob = pred_dist.view(N, 4, RM).softmax(2)
dist = (prob * proj).sum(2)
lt, rb = dist[:, :2], dist[:, 2:]
boxes = torch.cat([anc - lt, anc + rb], 1) * stride

def ciou(b1, b2, eps=1e-7):
    x1, y1, x2, y2 = b1.unbind(1); X1, Y1, X2, Y2 = b2.unbind(1)
    w1, h1, w2, h2 = x2 - x1, y2 - y1, X2 - X1, Y2 - Y1
    inter = (torch.min(x2, X2) - torch.max(x1, X1)).clamp(0) * (torch.min(y2, Y2) - torch.max(y1, Y1)).clamp(0)
    uni = w1 * h1 + w2 * h2 - inter + eps
    iou = inter / uni
    cw = torch.max(x2, X2) - torch.min(x1, X1); ch = torch.max(y2, Y2) - torch.min(y1, Y1)
    c2 = cw * cw + ch * ch + eps
    rho2 = ((x1 + x2 - X1 - X2) ** 2 + (y1 + y2 - Y1 - Y2) ** 2) / 4
    v = (4 / math.pi ** 2) * (torch.atan(w2 / (h2 + eps)) - torch.atan(w1 / (h1 + eps))) ** 2
    with torch.no_grad(): alpha = v / (v - iou + (1 + eps))
    return iou - (rho2 / c2 + v * alpha)

save("ref_boxes.bin", boxes); save("ref_ciou.bin", ciou(boxes, gt))
print("m4a ref: N", N, "boxes", tuple(boxes.shape))
