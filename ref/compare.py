"""Compare C++ outputs against the Python reference (allclose)."""
import os, numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "out")

def load(name):
    return np.fromfile(os.path.join(OUT, name), dtype=np.float32)

ok = True
for ref, cpp, label in [("ref_boxes.bin", "cpp_boxes.bin", "DFL boxes"),
                        ("ref_ciou.bin", "cpp_ciou.bin", "CIoU")]:
    a, b = load(ref), load(cpp)
    if a.shape != b.shape:
        print(f"[FAIL] {label}: shape {a.shape} vs {b.shape}"); ok = False; continue
    maxdiff = float(np.max(np.abs(a - b))) if a.size else 0.0
    close = np.allclose(a, b, atol=1e-5, rtol=1e-4)
    print(f"[{'OK ' if close else 'FAIL'}] {label}: max|diff|={maxdiff:.3e}")
    ok = ok and close

print("\nALL PASS" if ok else "\nMISMATCH")
raise SystemExit(0 if ok else 1)
