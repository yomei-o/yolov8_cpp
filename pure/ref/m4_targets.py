"""M4: dump inputs, TAL targets, and reference loss/grads (torch autograd) for the
pure-engine loss parity. Same math/seed as ../ref/loss_ref.py."""
import os, math, torch
import torch.nn.functional as F
HERE = os.path.dirname(os.path.abspath(__file__))
D = os.path.join(HERE, "data_m4"); os.makedirs(D, exist_ok=True)
def save(n, t): t.detach().contiguous().float().cpu().numpy().tofile(os.path.join(D, n))

torch.manual_seed(2)
B, NC, M, GRID, RM, TOPK = 2, 4, 3, 5, 16, 10
A = GRID * GRID; ALPHA, BETA, EPS = 0.5, 6.0, 1e-9
STRIDE = 8.0
ys, xs = torch.meshgrid(torch.arange(GRID), torch.arange(GRID), indexing="ij")
anc = torch.stack([xs.flatten(), ys.flatten()], 1).float() + 0.5
stride = torch.full((A, 1), STRIDE)
pred_distri = torch.randn(B, A, 4 * RM, requires_grad=True)
pred_scores = torch.randn(B, A, NC, requires_grad=True)
gt_bboxes = torch.tensor([[[8, 8, 32, 32], [20, 6, 39, 30], [6, 18, 30, 39]],
                          [[4, 4, 28, 36], [18, 16, 39, 39], [2, 20, 24, 39]]], dtype=torch.float32)
gt_labels = torch.tensor([[[1], [2], [0]], [[3], [0], [2]]], dtype=torch.float32)
mask_gt = torch.ones(B, M, 1)
proj = torch.arange(RM, dtype=torch.float32)

def ciou(b1, b2, eps=1e-7):
    x1, y1, x2, y2 = b1.unbind(-1); X1, Y1, X2, Y2 = b2.unbind(-1)
    w1, h1, w2, h2 = x2 - x1, y2 - y1, X2 - X1, Y2 - Y1
    inter = (torch.min(x2, X2) - torch.max(x1, X1)).clamp(0) * (torch.min(y2, Y2) - torch.max(y1, Y1)).clamp(0)
    uni = w1 * h1 + w2 * h2 - inter + eps; iou = inter / uni
    cw = torch.max(x2, X2) - torch.min(x1, X1); ch = torch.max(y2, Y2) - torch.min(y1, Y1)
    c2 = cw * cw + ch * ch + eps
    rho2 = ((x1 + x2 - X1 - X2) ** 2 + (y1 + y2 - Y1 - Y2) ** 2) / 4
    v = (4 / math.pi ** 2) * (torch.atan(w2 / (h2 + eps)) - torch.atan(w1 / (h1 + eps))) ** 2
    with torch.no_grad(): alpha = v / (v - iou + (1 + eps))
    return iou - (rho2 / c2 + v * alpha)

def bbox_decode(anc, pd):
    b, a, _ = pd.shape
    d = pd.view(b, a, 4, RM).softmax(3).matmul(proj)
    return torch.cat([anc - d[..., :2], anc + d[..., 2:]], -1)

@torch.no_grad()
def assign(ps, pb, ap):
    lt = gt_bboxes[..., None, :2]; rb = gt_bboxes[..., None, 2:]
    deltas = torch.cat([ap[None, None] - lt, rb - ap[None, None]], -1)
    mask_in = deltas.amin(-1).gt(EPS); mc = mask_in * mask_gt.bool()
    ov = ciou(gt_bboxes[:, :, None], pb[:, None]).clamp(0) * mc
    st = ps.permute(0, 2, 1); ci = gt_labels.long().expand(B, M, A)
    bs = torch.gather(st, 1, ci) * mc
    am = bs.pow(ALPHA) * ov.pow(BETA)
    _, ti = torch.topk(am, TOPK, -1)
    tm = mask_gt.expand(-1, -1, TOPK).bool(); ti = ti.masked_fill(~tm, 0)
    cnt = torch.zeros_like(am, dtype=torch.int8); one = torch.ones_like(ti[:, :, :1], dtype=torch.int8)
    for k in range(TOPK): cnt.scatter_add_(-1, ti[:, :, k:k + 1], one)
    mp = cnt.masked_fill(cnt > 1, 0).float() * mask_in.float() * mask_gt
    fg = mp.sum(-2)
    if fg.max() > 1:
        multi = (fg[:, None] > 1).expand(-1, M, -1); mi = ov.argmax(1)
        ismax = torch.zeros_like(mp); ismax.scatter_(1, mi.unsqueeze(1), 1)
        mp = torch.where(multi, ismax, mp); fg = mp.sum(-2)
    tgi = mp.argmax(-2); bind = torch.arange(B)[..., None]; fi = tgi + bind * M
    tb = gt_bboxes.view(-1, 4)[fi]
    tl = gt_labels.long().flatten()[fi].clamp(0)
    ts = F.one_hot(tl, NC).float()
    ts = torch.where(fg[:, :, None] > 0, ts, torch.zeros_like(ts))
    am = am * mp; pa = am.amax(-1, keepdim=True); po = (ov * mp).amax(-1, keepdim=True)
    norm = (am * po / (pa + EPS)).amax(-2).unsqueeze(-1)
    return tb, ts * norm, fg.bool()

pb = bbox_decode(anc, pred_distri)
tb, ts, fg = assign(pred_scores.detach().sigmoid(), pb.detach() * stride, anc * stride)
tss = ts.sum().clamp(min=1)
loss_cls = F.binary_cross_entropy_with_logits(pred_scores, ts, reduction="none").sum() / tss
tbg = tb / stride
weight = ts.sum(-1)[fg]
loss_box = ((1 - ciou(pb[fg], tbg[fg])) * weight).sum() / tss
tltrb = torch.cat([anc - tbg[..., :2], tbg[..., 2:] - anc], -1).clamp(0, RM - 1 - 0.01)
def dfl_loss(pd, tgt):
    tgt = tgt.clamp(0, RM - 1 - 0.01); tl = tgt.long(); tr = tl + 1; wl = tr - tgt; wr = 1 - wl
    p = pd.view(-1, RM)
    return (F.cross_entropy(p, tl.view(-1), reduction="none").view(tl.shape) * wl +
            F.cross_entropy(p, tr.view(-1), reduction="none").view(tr.shape) * wr).mean(-1)
loss_dfl = (dfl_loss(pred_distri[fg].view(-1, RM), tltrb[fg]) * weight).sum() / tss
total = 7.5 * loss_box + 0.5 * loss_cls + 1.5 * loss_dfl

save("pred_distri.bin", pred_distri); save("pred_scores.bin", pred_scores)
save("anc.bin", anc); save("stride.bin", stride)
save("tb.bin", tb); save("ts.bin", ts)              # targets (tb in image units)
torch.tensor([loss_box.item(), loss_cls.item(), loss_dfl.item()]).numpy().tofile(os.path.join(D, "ref_losses.bin"))
torch.tensor([total.item()]).numpy().tofile(os.path.join(D, "ref_total.bin"))
total.backward()
save("ref_grad_distri.bin", pred_distri.grad); save("ref_grad_scores.bin", pred_scores.grad)
open(os.path.join(D, "meta.txt"), "w").write(f"{B} {A} {NC} {RM}\n")
print(f"box={loss_box.item():.6f} cls={loss_cls.item():.6f} dfl={loss_dfl.item():.6f} total={total.item():.6f}")
