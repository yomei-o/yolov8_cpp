"""Reference for the pure BatchNorm2d op. Dumps a fixed input, affine params and
running buffers, then torch.nn.BatchNorm2d forward + backward (given an upstream
grad) and the updated running stats, for both train and eval modes."""
import os, numpy as np, torch

HERE = os.path.dirname(os.path.abspath(__file__))
D = os.path.join(HERE, "data_bn"); os.makedirs(D, exist_ok=True)
torch.manual_seed(0)

N, C, H, W = 2, 4, 5, 6
EPS, MOM = 1e-5, 0.1
def save(n, t): (t.detach().cpu().numpy() if torch.is_tensor(t) else np.asarray(t)).astype(np.float32).tofile(os.path.join(D, n))

x = torch.randn(N, C, H, W, dtype=torch.float64, requires_grad=True)
gamma = torch.randn(C, dtype=torch.float64, requires_grad=True)
beta = torch.randn(C, dtype=torch.float64, requires_grad=True)
rm0 = torch.randn(C, dtype=torch.float64)
rv0 = torch.rand(C, dtype=torch.float64) + 0.5
gy = torch.randn(N, C, H, W, dtype=torch.float64)

save("x.bin", x); save("gamma.bin", gamma); save("beta.bin", beta)
save("rm.bin", rm0); save("rv.bin", rv0); save("gy.bin", gy)
open(os.path.join(D, "meta.txt"), "w").write(f"{N} {C} {H} {W} {EPS} {MOM}\n")

for mode in ("train", "eval"):
    rm = rm0.clone(); rv = rv0.clone()
    bn = torch.nn.BatchNorm2d(C, eps=EPS, momentum=MOM).double()
    with torch.no_grad():
        bn.weight.copy_(gamma); bn.bias.copy_(beta)
        bn.running_mean.copy_(rm); bn.running_var.copy_(rv)
    bn.train(mode == "train")
    xin = x.detach().clone().requires_grad_(True)
    y = bn(xin)
    y.backward(gy)
    save(f"y_{mode}.bin", y)
    save(f"dx_{mode}.bin", xin.grad)
    save(f"dgamma_{mode}.bin", bn.weight.grad)
    save(f"dbeta_{mode}.bin", bn.bias.grad)
    save(f"rm_{mode}.bin", bn.running_mean)
    save(f"rv_{mode}.bin", bn.running_var)

print(f"BN ref: N={N} C={C} H={H} W={W}")
