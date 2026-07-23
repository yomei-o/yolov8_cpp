"""Dump the architecture description (manifest_unfused.txt + names.txt) for EVERY size
n/s/m/l/x, built from the .yaml config (random init, NO pretrained download). These tiny
text files let the pure-C++ make_init_pt generate initial weights for any size with zero
Python. Writes pure/ref/arch/<model>/{manifest_unfused.txt,names.txt}.
Usage: python export_arch_all.py"""
import os, sys
from ultralytics import YOLO
from yolo_walk import walk

HERE = os.path.dirname(os.path.abspath(__file__))
SIZES = sys.argv[1:] or ["yolov8n", "yolov8s", "yolov8m", "yolov8l", "yolov8x"]

for MODEL in SIZES:
    ym = YOLO(MODEL + ".yaml")          # architecture only, random init, no download
    ym.model.eval()
    mods = walk(ym.model)
    qn = {id(mm): nm for nm, mm in ym.model.named_modules()}
    lines = [str(len(mods))]; names = []
    for m in mods:
        p = qn[id(m)]
        if hasattr(m, "bn"):
            conv, bn = m.conv, m.bn
            Co, Ci, k, s = conv.weight.shape[0], conv.weight.shape[1], conv.kernel_size[0], conv.stride[0]
            lines.append(f"1 {Co} {Ci} {k} {s} {bn.eps}")
            names += [f"{p}.conv.weight", f"{p}.bn.weight", f"{p}.bn.bias", f"{p}.bn.running_mean", f"{p}.bn.running_var"]
        else:
            Co, Ci, k, s = m.weight.shape[0], m.weight.shape[1], m.kernel_size[0], m.stride[0]
            lines.append(f"0 {Co} {Ci} {k} {s} 0")
            names += [f"{p}.weight", f"{p}.bias"]
    D = os.path.join(HERE, "arch", MODEL); os.makedirs(D, exist_ok=True)
    open(os.path.join(D, "manifest_unfused.txt"), "w").write("\n".join(lines) + "\n")
    open(os.path.join(D, "names.txt"), "w").write("\n".join(names) + "\n")
    print(f"{MODEL}: {len(mods)} layers, {len(names)} tensors -> arch/{MODEL}/")
