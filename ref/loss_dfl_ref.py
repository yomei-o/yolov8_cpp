"""DFL (Distribution Focal Loss) forward reference, faithful to Ultralytics DFLoss.
per-anchor loss = mean over 4 sides of  CE(pred, tl)*wl + CE(pred, tr)*wr ."""
import os, torch
import torch.nn.functional as F

torch.manual_seed(1)
HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data"); OUT = os.path.join(HERE, "..", "out")
os.makedirs(DATA, exist_ok=True); os.makedirs(OUT, exist_ok=True)

REG_MAX, N = 16, 12
pred_dist = torch.randn(N, 4, REG_MAX)              # logits per side
target_ltrb = torch.rand(N, 4) * (REG_MAX - 1)     # continuous distances in [0, 15]

pred_dist.numpy().tofile(os.path.join(DATA, "d_pred_dist.bin"))
target_ltrb.numpy().tofile(os.path.join(DATA, "d_target_ltrb.bin"))
with open(os.path.join(DATA, "dfl_meta.txt"), "w") as f:
    f.write(f"{N} {REG_MAX}\n")

def dfl_loss(pred_dist, target, reg_max=REG_MAX):
    target = target.clamp(0, reg_max - 1 - 0.01)
    tl = target.long()          # left bin
    tr = tl + 1                 # right bin
    wl = tr - target            # weight to left
    wr = 1 - wl                 # weight to right
    p = pred_dist.view(-1, reg_max)      # (N*4, reg_max)
    ce_l = F.cross_entropy(p, tl.view(-1), reduction="none").view(tl.shape)
    ce_r = F.cross_entropy(p, tr.view(-1), reduction="none").view(tr.shape)
    return (ce_l * wl + ce_r * wr).mean(-1, keepdim=True)   # (N,1)

loss = dfl_loss(pred_dist, target_ltrb)
loss.contiguous().numpy().tofile(os.path.join(OUT, "ref_dfl.bin"))
print("DFL ref generated. loss[:3] =", loss.view(-1)[:3].tolist())
print("total mean =", loss.mean().item())
