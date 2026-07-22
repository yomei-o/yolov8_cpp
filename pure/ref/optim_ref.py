"""Reference for the pure optimizers. Runs each optimizer config on a deterministic
quadratic loss f(w)=0.5*||w-target||^2 (grad = w-target) for K steps and dumps the
final params, so the C++ update rules can be checked exactly against torch.optim."""
import os, numpy as np, torch

HERE = os.path.dirname(os.path.abspath(__file__))
D = os.path.join(HERE, "data_optim"); os.makedirs(D, exist_ok=True)
torch.manual_seed(0)

N, K = 20, 50
w0 = torch.randn(N, dtype=torch.float64)
target = torch.randn(N, dtype=torch.float64)
w0.numpy().astype(np.float32).tofile(os.path.join(D, "w0.bin"))
target.numpy().astype(np.float32).tofile(os.path.join(D, "target.bin"))

def run(make_opt):
    w = w0.clone().requires_grad_(True)
    opt = make_opt([w])
    for _ in range(K):
        opt.zero_grad()
        loss = 0.5 * ((w - target) ** 2).sum()
        loss.backward()
        opt.step()
    return w.detach().numpy().astype(np.float32)

cfgs = {
    "sgd":       lambda p: torch.optim.SGD(p, lr=0.1, momentum=0.9, weight_decay=1e-4),
    "sgd_nest":  lambda p: torch.optim.SGD(p, lr=0.1, momentum=0.9, weight_decay=1e-4, nesterov=True),
    "adam":      lambda p: torch.optim.Adam(p, lr=0.05, weight_decay=1e-2),
    "adamw":     lambda p: torch.optim.AdamW(p, lr=0.05, weight_decay=1e-2),
}
for name, mk in cfgs.items():
    run(mk).tofile(os.path.join(D, f"w_{name}.bin"))
open(os.path.join(D, "meta.txt"), "w").write(f"{N} {K}\n")
print(f"optim ref: N={N} K={K} configs={list(cfgs)}")
