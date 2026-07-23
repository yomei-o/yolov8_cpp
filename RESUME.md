# RESUME — remaining work

Status of the pure-C++ training toolchain and what's left to make it a full replacement
for Ultralytics-quality training. Verified items live in [README.md](README.md); this file
is the forward-looking TODO.

## Done (pure C++, no Python at run time)
- Engine + all 4 YOLOs (v5/v8/v11/x), all sizes (n/s/m/l/x), forward/loss/train/infer/mAP/ONNX/`.pt`.
- Real training CLI (`pure/train_cli.cpp`): dataset scan → shuffled mini-batches → epochs →
  assignment → loss → Adam(warmup+cosine+wd) → per-epoch val mAP@0.5 → `best.pt`/`last.pt`.
- Initial-weight `.pt` generated in C++ (`pure/make_init_pt.cpp`, `rand`/`from`), all sizes,
  zero-Python bootstrap; checkpoints load back into Ultralytics/reference (0 unexpected).
- **Standard-YOLO dataset ingestion** — directory scan (`images/`↔`labels/`), normalised
  `cls xc yc w h` labels, arbitrary-size images letterboxed (`pure/dataset.hpp`
  `read_yolo_dataset` / `load_boxes_orig`). Verified: val mAP → 0.97 on the synthetic set.
- **Mosaic augmentation** (`make_mosaic`) + horizontal flip + brightness. `train_cli … <imgsz> <mosaic>`.
- GPU/CUDA seam (`pure/backend.hpp`) present in all four; conv/matmul route through `bk::`.

## Remaining (roughly in priority order)
1. **Real-dataset convergence parity** — train on COCO128 (or similar) and compare final
   mAP@0.5:0.95 against Ultralytics. This is the key "results, not just pipeline" validation.
   Nothing beyond synthetic data has been checked for convergence quality yet.
2. **Richer augmentation** — HSV colour jitter, random affine (scale/translate/rotate/shear),
   mixup, and "close mosaic for the last N epochs" (Ultralytics default). Only flip +
   brightness + mosaic exist today.
3. **`data.yaml` + unified CLI** — parse `data.yaml` (train/val paths, `nc`, `names`) and add
   `train`/`val`/`detect`/`export` subcommands so it reads a dataset the way Ultralytics does.
   Today `train_cli` takes an images dir / list file and a fixed `nc`.
4. **Training-quality features** — EMA weights, resume-from-checkpoint, multi-scale training,
   rectangular val, label smoothing, separate bias/BN LR + warmup-bias-lr, mAP@0.5:0.95 in
   the val loop (only mAP@0.5 is printed now).
5. **Speed** — yolox uses a per-image forward summed per minibatch (batch it like v8); verify
   the GPU path on real hardware for v5/v11/x (only v8 was seen running on a Colab T4).

## Notes / gotchas
- Label coords: internally everything is xyxy in the **letterboxed SxS pixel** space; GT and
  decoded detections share it, so val mAP is apples-to-apples. `load_boxes_orig` reads either
  format into original pixels, then `lb_map` applies the letterbox transform.
- `make_init_pt … rand` trains but won't converge on tiny data; use `from <pretrained.pt>`
  for transfer learning.
- Build: MSVC via `C:/prog/claude/cc.sh`; `scratch/` must pre-exist; vcvars hangs here.
