"""Export real yolov8n as a TorchScript module returning the RAW detection-head
feature maps (3 tensors) for training in C++. We wrap _predict_once so the traced
output is a homogeneous tuple of tensors (avoids DetectionModel.forward's dict output)."""
import os, torch, torch.nn as nn
from ultralytics import YOLO

HERE = os.path.dirname(os.path.abspath(__file__))
OUTP = os.path.join(HERE, "..", "yolov8n_train.torchscript")

dm = YOLO("yolov8n.pt").model.float()
dm.train()
det = dm.model[-1]
print("nc =", det.nc, " reg_max =", det.reg_max, " stride =", dm.stride.tolist())

class Wrap(nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m
    def forward(self, x):
        y = self.m._predict_once(x)   # training dict: boxes(b,4*rm,A), scores(b,nc,A), feats
        return y["boxes"], y["scores"]

w = Wrap(dm).train()
for p in w.parameters():
    p.requires_grad_(True)

ex = torch.randn(2, 3, 64, 64)
ts = torch.jit.trace(w, ex, strict=False, check_trace=False)
ts.save(OUTP)

out = ts(ex)
print("levels =", len(out), " shapes =", [tuple(o.shape) for o in out])
print("saved:", OUTP)
