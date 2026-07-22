"""Verify the C++-written .onnx: structural check + run in onnxruntime and compare the
packed boxes/scores to the reference forward. Usage: python onnx_verify.py [model]"""
import os, sys, numpy as np, onnx, onnxruntime as ort

HERE = os.path.dirname(os.path.abspath(__file__))
model = sys.argv[1] if len(sys.argv) > 1 else "yolov8n"
D = os.path.join(HERE, f"data_arch_{model}")
onnx_path = f"{model}.onnx"

m = onnx.load(onnx_path); onnx.checker.check_model(m); print("onnx.checker: OK")

IMG, NB, NS, RM, Atot = open(os.path.join(D, "io.txt")).read().split()
IMG, NB, NS, Atot = int(IMG), int(NB), int(NS), int(Atot)
x = np.fromfile(os.path.join(D, "x.bin"), np.float32).reshape(1, 3, IMG, IMG)

sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
outs = {o.name: v for o, v in zip(sess.get_outputs(), sess.run(None, {"images": x}))}

# pack box0..box2 / cls0..cls2 into (NB,Atot)/(NS,Atot), level-major, channel-major
def pack(prefix, C):
    dst = np.zeros((C, Atot), np.float32); off = 0
    for l in range(3):
        t = outs[f"{prefix}{l}"][0]                 # (C,h,w)
        hw = t.shape[1] * t.shape[2]
        dst[:, off:off+hw] = t.reshape(C, hw); off += hw
    return dst
boxes, scores = pack("box", NB), pack("cls", NS)

rb = np.fromfile(os.path.join(D, "ref_boxes.bin"), np.float32).reshape(NB, Atot)
rs = np.fromfile(os.path.join(D, "ref_scores.bin"), np.float32).reshape(NS, Atot)
db, ds = np.abs(boxes - rb).max(), np.abs(scores - rs).max()
print(f"onnxruntime boxes max|diff| = {db:.3e}  {'OK' if db < 1e-3 else 'FAIL'}")
print(f"onnxruntime scores max|diff| = {ds:.3e}  {'OK' if ds < 1e-3 else 'FAIL'}")
print("\nA-2(write): C++ .onnx runs in onnxruntime == yolov8 forward"
      if db < 1e-3 and ds < 1e-3 else "MISMATCH")
