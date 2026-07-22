"""Full v8DetectionLoss reference (box=CIoU, cls=BCE, dfl=DFL), faithful to Ultralytics.
Differentiable inputs: pred_distri, pred_scores. Saves forward losses AND gradients
so the same harness serves the forward (part2) and backward (part3) C++ parity checks.
Intermediates (anchors/strides/targets) are pre-generated to isolate the loss math from
make_anchors / target-prep plumbing (ported separately later)."""
import os, math, torch
import torch.nn.functional as F

torch.manual_seed(2)
HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data"); OUT = os.path.join(HERE, "..", "out")
os.makedirs(DATA, exist_ok=True); os.makedirs(OUT, exist_ok=True)

B, NC, M, GRID, REG_MAX, TOPK = 2, 4, 3, 5, 16, 10
A = GRID * GRID
ALPHA, BETA, EPS = 0.5, 6.0, 1e-9
G_BOX, G_CLS, G_DFL = 7.5, 0.5, 1.5
STRIDE = 8.0

# ---- inputs ----
ys, xs = torch.meshgrid(torch.arange(GRID), torch.arange(GRID), indexing="ij")
anc = torch.stack([xs.flatten(), ys.flatten()], 1).float() + 0.5      # (A,2) grid units
stride = torch.full((A, 1), STRIDE)                                    # (A,1)
pred_distri = torch.randn(B, A, 4 * REG_MAX, requires_grad=True)
pred_scores = torch.randn(B, A, NC, requires_grad=True)                # logits
gt_bboxes = torch.tensor([[[8, 8, 32, 32], [20, 6, 39, 30], [6, 18, 30, 39]],
                          [[4, 4, 28, 36], [18, 16, 39, 39], [2, 20, 24, 39]]],
                         dtype=torch.float32)                          # (B,M,4) image units
gt_labels = torch.tensor([[[1], [2], [0]], [[3], [0], [2]]], dtype=torch.float32)
mask_gt = torch.ones(B, M, 1)

def save(name, t): t.detach().contiguous().to(torch.float32).cpu().numpy().tofile(os.path.join(DATA, name))
save("L_pred_distri.bin", pred_distri); save("L_pred_scores.bin", pred_scores)
save("L_anc.bin", anc); save("L_stride.bin", stride)
save("L_gt_labels.bin", gt_labels); save("L_gt_bboxes.bin", gt_bboxes); save("L_mask_gt.bin", mask_gt)
with open(os.path.join(DATA, "loss_meta.txt"), "w") as f:
    f.write(f"{B} {NC} {M} {A} {REG_MAX} {TOPK} {ALPHA} {BETA} {G_BOX} {G_CLS} {G_DFL}\n")

proj = torch.arange(REG_MAX, dtype=torch.float32)

def bbox_ciou(b1, b2, eps=1e-7):
    x1, y1, x2, y2 = b1.unbind(-1); X1, Y1, X2, Y2 = b2.unbind(-1)
    w1, h1, w2, h2 = x2 - x1, y2 - y1, X2 - X1, Y2 - Y1
    inter = (torch.min(x2, X2) - torch.max(x1, X1)).clamp(0) * (torch.min(y2, Y2) - torch.max(y1, Y1)).clamp(0)
    uni = w1 * h1 + w2 * h2 - inter + eps
    iou = inter / uni
    cw = torch.max(x2, X2) - torch.min(x1, X1); ch = torch.max(y2, Y2) - torch.min(y1, Y1)
    c2 = cw * cw + ch * ch + eps
    rho2 = ((x1 + x2 - X1 - X2) ** 2 + (y1 + y2 - Y1 - Y2) ** 2) / 4
    v = (4 / math.pi ** 2) * (torch.atan(w2 / (h2 + eps)) - torch.atan(w1 / (h1 + eps))) ** 2
    with torch.no_grad():
        alpha = v / (v - iou + (1 + eps))
    return iou - (rho2 / c2 + v * alpha)                    # (...,) 1D over batch of boxes

def bbox_decode(anc, pred_dist):
    b, a, _ = pred_dist.shape
    d = pred_dist.view(b, a, 4, REG_MAX).softmax(3).matmul(proj)   # (b,a,4) ltrb
    lt, rb = d[..., :2], d[..., 2:]
    return torch.cat([anc - lt, anc + rb], -1)              # xyxy grid units

@torch.no_grad()
def assign(pd_scores, pd_bboxes, anc_pts, gt_labels, gt_bboxes, mask_gt):
    lt = gt_bboxes[..., None, :2]; rb = gt_bboxes[..., None, 2:]
    deltas = torch.cat([anc_pts[None, None] - lt, rb - anc_pts[None, None]], -1)
    mask_in = deltas.amin(-1).gt(EPS)
    mc = (mask_in * mask_gt.bool())
    ov = bbox_ciou(gt_bboxes[:, :, None], pd_bboxes[:, None]).clamp(0) * mc
    st = pd_scores.permute(0, 2, 1)
    ci = gt_labels.long().expand(B, M, A)
    bs = torch.gather(st, 1, ci) * mc
    am = bs.pow(ALPHA) * ov.pow(BETA)
    _, ti = torch.topk(am, TOPK, -1)
    tm = mask_gt.expand(-1, -1, TOPK).bool(); ti = ti.masked_fill(~tm, 0)
    cnt = torch.zeros_like(am, dtype=torch.int8); one = torch.ones_like(ti[:, :, :1], dtype=torch.int8)
    for k in range(TOPK): cnt.scatter_add_(-1, ti[:, :, k:k + 1], one)
    mask_topk = cnt.masked_fill(cnt > 1, 0).float()
    mask_pos = mask_topk * mask_in.float() * mask_gt
    fg = mask_pos.sum(-2)
    if fg.max() > 1:
        multi = (fg[:, None] > 1).expand(-1, M, -1); mi = ov.argmax(1)
        ismax = torch.zeros_like(mask_pos); ismax.scatter_(1, mi.unsqueeze(1), 1)
        mask_pos = torch.where(multi, ismax, mask_pos); fg = mask_pos.sum(-2)
    tgi = mask_pos.argmax(-2)
    bind = torch.arange(B)[..., None]; fi = tgi + bind * M
    tl = gt_labels.long().flatten()[fi]
    tb = gt_bboxes.view(-1, 4)[fi]
    ts = F.one_hot(tl.clamp(0), NC).float()
    ts = torch.where(fg[:, :, None] > 0, ts, torch.zeros_like(ts))
    am = am * mask_pos
    pa = am.amax(-1, keepdim=True); po = (ov * mask_pos).amax(-1, keepdim=True)
    norm = (am * po / (pa + EPS)).amax(-2).unsqueeze(-1)
    return tb, ts * norm, fg.bool()

def dfl_loss(pred_dist, target):
    target = target.clamp(0, REG_MAX - 1 - 0.01)
    tl = target.long(); tr = tl + 1; wl = tr - target; wr = 1 - wl
    p = pred_dist.view(-1, REG_MAX)
    ce_l = F.cross_entropy(p, tl.view(-1), reduction="none").view(tl.shape)
    ce_r = F.cross_entropy(p, tr.view(-1), reduction="none").view(tr.shape)
    return (ce_l * wl + ce_r * wr).mean(-1)          # (n,)

# ---- forward ----
pred_bboxes = bbox_decode(anc, pred_distri)           # grid units
tb, ts, fg = assign(pred_scores.detach().sigmoid(),
                    pred_bboxes.detach() * stride, anc * stride,
                    gt_labels, gt_bboxes, mask_gt)
tss = ts.sum().clamp(min=1)

loss_cls = F.binary_cross_entropy_with_logits(pred_scores, ts, reduction="none").sum() / tss

tb = tb / stride                                       # to grid units
weight = ts.sum(-1)[fg]                                # (n,)
iou = bbox_ciou(pred_bboxes[fg], tb[fg])               # (n,)
loss_box = ((1 - iou) * weight).sum() / tss

def bbox2dist(anc, bbox):
    lt = bbox[..., :2]; rb = bbox[..., 2:]
    return torch.cat([anc - lt, rb - anc], -1).clamp(0, REG_MAX - 1 - 0.01)
tltrb = bbox2dist(anc, tb)
ldfl = dfl_loss(pred_distri[fg].view(-1, REG_MAX), tltrb[fg]) * weight
loss_dfl = ldfl.sum() / tss

total = G_BOX * loss_box + G_CLS * loss_cls + G_DFL * loss_dfl

torch.tensor([loss_box.item(), loss_cls.item(), loss_dfl.item()]).numpy().tofile(os.path.join(OUT, "ref_losses.bin"))
torch.tensor([total.item()]).numpy().tofile(os.path.join(OUT, "ref_total.bin"))

# ---- backward (for part 3) ----
total.backward()
pred_distri.grad.detach().contiguous().numpy().tofile(os.path.join(OUT, "ref_grad_distri.bin"))
pred_scores.grad.detach().contiguous().numpy().tofile(os.path.join(OUT, "ref_grad_scores.bin"))

print(f"box={loss_box.item():.6f} cls={loss_cls.item():.6f} dfl={loss_dfl.item():.6f} total={total.item():.6f}")
print(f"fg positives per batch = {fg.sum(-1).tolist()}")
print(f"grad_distri |max|={pred_distri.grad.abs().max():.4e}  grad_scores |max|={pred_scores.grad.abs().max():.4e}")
