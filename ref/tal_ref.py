"""Faithful, self-contained reimplementation of Ultralytics TaskAlignedAssigner.
Generates deterministic inputs + reference outputs as raw float32 .bin for C++ parity.
Mirrors ultralytics/utils/tal.py (topk=10, alpha=0.5, beta=6.0)."""
import os, math
import torch
import torch.nn.functional as F

torch.manual_seed(0)
HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data")
OUT = os.path.join(HERE, "..", "out")
os.makedirs(DATA, exist_ok=True); os.makedirs(OUT, exist_ok=True)

# ---- config / dims ----
B, NC, M, TOPK = 2, 4, 3, 10
ALPHA, BETA, EPS = 0.5, 6.0, 1e-9
GRID = 5            # 5x5 feature map -> A=25 anchors
A = GRID * GRID

def save(name, t):
    t.detach().contiguous().to(torch.float32).cpu().numpy().tofile(os.path.join(DATA, name))
def save_out(name, t):
    t.detach().contiguous().to(torch.float32).cpu().numpy().tofile(os.path.join(OUT, name))

# ---- inputs ----
ys, xs = torch.meshgrid(torch.arange(GRID), torch.arange(GRID), indexing="ij")
anc_points = torch.stack([xs.flatten(), ys.flatten()], 1).float() * 8 + 4   # (A,2)

r = torch.rand(B, A, 1) * 7 + 3                          # random half-size
pd_bboxes = torch.cat([anc_points.unsqueeze(0).expand(B, -1, -1) - r,
                       anc_points.unsqueeze(0).expand(B, -1, -1) + r], 2)    # (B,A,4) xyxy
pd_scores = torch.rand(B, A, NC)                          # (B,A,nc) in [0,1]

gt_bboxes = torch.tensor([[[6, 6, 26, 26], [20, 10, 38, 30], [10, 20, 30, 38]],
                          [[4, 4, 24, 34], [18, 18, 39, 39], [2, 22, 22, 39]]],
                         dtype=torch.float32)             # (B,M,4) overlapping
gt_labels = torch.tensor([[[1], [2], [0]], [[3], [0], [2]]], dtype=torch.float32)  # (B,M,1)
mask_gt = torch.ones(B, M, 1)                             # all gts valid

for n, t in [("t_pd_scores.bin", pd_scores), ("t_pd_bboxes.bin", pd_bboxes),
             ("t_anc.bin", anc_points), ("t_gt_labels.bin", gt_labels),
             ("t_gt_bboxes.bin", gt_bboxes), ("t_mask_gt.bin", mask_gt)]:
    save(n, t)
with open(os.path.join(DATA, "tal_meta.txt"), "w") as f:
    f.write(f"{B} {NC} {M} {A} {TOPK} {ALPHA} {BETA}\n")

# ---- CIoU (broadcasting, xyxy) ----
def bbox_ciou(b1, b2, eps=1e-7):
    x1, y1, x2, y2 = b1.unbind(-1)
    X1, Y1, X2, Y2 = b2.unbind(-1)
    w1, h1 = (x2 - x1), (y2 - y1)
    w2, h2 = (X2 - X1), (Y2 - Y1)
    inter = (torch.min(x2, X2) - torch.max(x1, X1)).clamp(0) * \
            (torch.min(y2, Y2) - torch.max(y1, Y1)).clamp(0)
    uni = w1 * h1 + w2 * h2 - inter + eps
    iou = inter / uni
    cw = torch.max(x2, X2) - torch.min(x1, X1)
    ch = torch.max(y2, Y2) - torch.min(y1, Y1)
    c2 = cw * cw + ch * ch + eps
    rho2 = ((x1 + x2 - X1 - X2) ** 2 + (y1 + y2 - Y1 - Y2) ** 2) / 4
    v = (4 / math.pi ** 2) * (torch.atan(w2 / (h2 + eps)) - torch.atan(w1 / (h1 + eps))) ** 2
    with torch.no_grad():
        alpha = v / (v - iou + (1 + eps))
    return iou - (rho2 / c2 + v * alpha)

# ---- TaskAlignedAssigner ----
@torch.no_grad()
def assign():
    # candidates in gts: (B,M,A)
    lt = gt_bboxes[..., None, :2]                    # (B,M,1,2)
    rb = gt_bboxes[..., None, 2:]
    deltas = torch.cat([anc_points[None, None] - lt, rb - anc_points[None, None]], -1)  # (B,M,A,4)
    mask_in_gts = deltas.amin(-1).gt(EPS)            # (B,M,A) bool

    mask_comb = (mask_in_gts * mask_gt.bool())       # (B,M,A) bool

    # box metrics
    gt_e = gt_bboxes[:, :, None, :].expand(B, M, A, 4)
    pd_e = pd_bboxes[:, None, :, :].expand(B, M, A, 4)
    overlaps = bbox_ciou(gt_e, pd_e).clamp(0) * mask_comb        # (B,M,A)
    scores_t = pd_scores.permute(0, 2, 1)            # (B,nc,A)
    cls_idx = gt_labels.long().expand(B, M, A)       # (B,M,A) class id per gt
    bbox_scores = torch.gather(scores_t, 1, cls_idx) * mask_comb  # (B,M,A)
    align_metric = bbox_scores.pow(ALPHA) * overlaps.pow(BETA)    # (B,M,A)

    # topk over anchors
    topk_metrics, topk_idxs = torch.topk(align_metric, TOPK, dim=-1)
    topk_mask = mask_gt.expand(-1, -1, TOPK).bool()
    topk_idxs = topk_idxs.masked_fill(~topk_mask, 0)
    count = torch.zeros_like(align_metric, dtype=torch.int8)
    ones = torch.ones_like(topk_idxs[:, :, :1], dtype=torch.int8)
    for k in range(TOPK):
        count.scatter_add_(-1, topk_idxs[:, :, k:k + 1], ones)
    count = count.masked_fill(count > 1, 0)
    mask_topk = count.to(align_metric.dtype)

    mask_pos = mask_topk * mask_in_gts.float() * mask_gt          # (B,M,A)

    # resolve multi-gt anchors by highest overlap
    fg_mask = mask_pos.sum(-2)                                    # (B,A)
    if fg_mask.max() > 1:
        multi = (fg_mask[:, None] > 1).expand(-1, M, -1)
        max_idx = overlaps.argmax(1)                             # (B,A)
        is_max = torch.zeros_like(mask_pos)
        is_max.scatter_(1, max_idx.unsqueeze(1), 1)
        mask_pos = torch.where(multi, is_max, mask_pos)
        fg_mask = mask_pos.sum(-2)
    target_gt_idx = mask_pos.argmax(-2)                          # (B,A)

    # gather targets
    batch_ind = torch.arange(B, dtype=torch.long)[..., None]
    flat_idx = target_gt_idx + batch_ind * M                     # (B,A)
    target_labels = gt_labels.long().flatten()[flat_idx]        # (B,A)
    target_bboxes = gt_bboxes.view(-1, 4)[flat_idx]             # (B,A,4)
    target_labels = target_labels.clamp(0)
    target_scores = F.one_hot(target_labels, NC).float()       # (B,A,nc)
    fg_scores_mask = fg_mask[:, :, None].expand(-1, -1, NC)
    target_scores = torch.where(fg_scores_mask > 0, target_scores, torch.zeros_like(target_scores))

    # normalize
    align_metric = align_metric * mask_pos
    pos_am = align_metric.amax(-1, keepdim=True)
    pos_ov = (overlaps * mask_pos).amax(-1, keepdim=True)
    norm = (align_metric * pos_ov / (pos_am + EPS)).amax(-2).unsqueeze(-1)
    target_scores = target_scores * norm
    return target_labels.float(), target_bboxes, target_scores, fg_mask, target_gt_idx.float()

tl, tb, ts, fg, tgi = assign()
save_out("ref_target_labels.bin", tl)
save_out("ref_target_bboxes.bin", tb)
save_out("ref_target_scores.bin", ts)
save_out("ref_fg_mask.bin", fg)
save_out("ref_target_gt_idx.bin", tgi)
print("TAL reference generated.")
print("fg_mask sum per batch:", fg.sum(-1).tolist())
print("target_gt_idx[0]:", tgi[0].int().tolist())
print("target_labels[0]:", tl[0].int().tolist())
