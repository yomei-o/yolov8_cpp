"""M3a: export yolov8n's first Conv module (conv->BN->SiLU) with BN folded into
the conv weights, plus a reference output, so the pure C++ engine can match it."""
import os, torch
from ultralytics import YOLO

HERE = os.path.dirname(os.path.abspath(__file__))
D = os.path.join(HERE, "data"); os.makedirs(D, exist_ok=True)
def save(n, t): t.detach().contiguous().float().cpu().numpy().tofile(os.path.join(D, n))

torch.manual_seed(0)
m = YOLO("yolov8n.pt").model.model[0].eval()   # Conv(3,16,k3,s2,p1)
conv, bn = m.conv, m.bn
k = conv.kernel_size[0]; stride = conv.stride[0]; pad = conv.padding[0]
Cout, Cin = conv.weight.shape[:2]

# fold BN into conv
std = torch.sqrt(bn.running_var + bn.eps)
w_fold = conv.weight * (bn.weight / std).reshape(-1, 1, 1, 1)
b_fold = bn.bias - bn.weight * bn.running_mean / std

x = torch.randn(1, Cin, 64, 64)
with torch.no_grad():
    ref = m(x)                                 # silu(bn(conv(x)))

save("x.bin", x); save("wfold.bin", w_fold); save("bfold.bin", b_fold); save("ref.bin", ref)
with open(os.path.join(D, "meta.txt"), "w") as f:
    f.write(f"{Cin} {Cout} 64 64 {k} {stride} {pad}\n")
print(f"Conv({Cin},{Cout},k{k},s{stride},p{pad}) -> ref {tuple(ref.shape)}")
print("ref mean/std:", ref.mean().item(), ref.std().item())
