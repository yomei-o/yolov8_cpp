"""Reference for the pure matmul/transpose ops. Dumps A, B and, for a scalar loss
sum((A@B) * G) + sum(A^T * H), the forward A@B and the grads dA, dB from torch autograd."""
import os, numpy as np, torch

HERE = os.path.dirname(os.path.abspath(__file__))
D = os.path.join(HERE, "data_matmul"); os.makedirs(D, exist_ok=True)
torch.manual_seed(0)

M, K, N = 5, 7, 4
A = torch.randn(M, K, dtype=torch.float64, requires_grad=True)
B = torch.randn(K, N, dtype=torch.float64, requires_grad=True)
G = torch.randn(M, N, dtype=torch.float64)     # upstream grad for A@B
H = torch.randn(K, M, dtype=torch.float64)     # upstream grad for A^T

O = A @ B
loss = (O * G).sum() + (A.t() * H).sum()
loss.backward()

def save(n, t): (t.detach().cpu().numpy() if torch.is_tensor(t) else np.asarray(t)).astype(np.float32).tofile(os.path.join(D, n))
save("A.bin", A); save("B.bin", B); save("G.bin", G); save("H.bin", H)
save("O.bin", O); save("dA.bin", A.grad); save("dB.bin", B.grad)
open(os.path.join(D, "meta.txt"), "w").write(f"{M} {K} {N}\n")
print(f"matmul ref: M={M} K={K} N={N}")
