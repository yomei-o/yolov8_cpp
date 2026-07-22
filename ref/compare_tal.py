"""Compare C++ TAL outputs vs Python reference."""
import os, numpy as np
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "out")
def load(n): return np.fromfile(os.path.join(OUT, n), dtype=np.float32)

pairs = [("target_labels", "exact"), ("target_gt_idx", "exact"),
         ("fg_mask", "exact"), ("target_bboxes", "close"), ("target_scores", "close")]
ok = True
for name, kind in pairs:
    a, b = load(f"ref_{name}.bin"), load(f"cpp_{name}.bin")
    if a.shape != b.shape:
        print(f"[FAIL] {name}: shape {a.shape} vs {b.shape}"); ok = False; continue
    if kind == "exact":
        good = np.array_equal(a, b); d = int(np.sum(a != b))
        print(f"[{'OK ' if good else 'FAIL'}] {name}: mismatched elems={d}")
    else:
        d = float(np.max(np.abs(a - b))) if a.size else 0.0
        good = np.allclose(a, b, atol=1e-5, rtol=1e-4)
        print(f"[{'OK ' if good else 'FAIL'}] {name}: max|diff|={d:.3e}")
    ok = ok and good
print("\nALL PASS" if ok else "\nMISMATCH")
raise SystemExit(0 if ok else 1)
