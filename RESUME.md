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
- **Augmentation** — mosaic + mixup + random-affine (rotate/scale/shear/translate) + HSV +
  flip, with **close-mosaic** (disable for last N epochs). Toggle via `AugCfg` / CLI flags.
- **Unified `yolo` CLI** (`pure/yolo.cpp`) reading `data.yaml`: `train` / `val` / `detect`
  (`export` still delegates to the standalone `onnx_export` — see remaining #3).
  Val reports **mAP@0.5 and mAP@0.5:0.95**.
- GPU/CUDA seam (`pure/backend.hpp`) present in all four; conv/matmul route through `bk::`.
  GPU training done for all four (per the parallel session).

## Remaining (roughly in priority order)
1. **Real-dataset convergence parity** — train on COCO128 (or similar) and compare final
   mAP@0.5:0.95 against Ultralytics. This is the key "results, not just pipeline" validation.
   Nothing beyond synthetic data has been checked for convergence quality yet.
2. **Custom `nc`** — the head is fixed at 80 classes; training on a dataset with `nc != 80`
   needs the cls head resized + re-initialised (make_init_pt could emit an `nc`-sized head).
   Today class ids must be < 80.
3. **`export` in the unified CLI** — fold BN from the `.pt` and emit ONNX in-CLI (today
   `yolo export` points at the standalone, onnxruntime-verified `onnx_export`).
4. **Training-quality features** — EMA weights, resume-from-checkpoint, multi-scale, rect val,
   label smoothing, separate bias/BN LR + warmup-bias-lr. (mAP@0.5:0.95 in val — done.)
5. **Speed** — yolox uses a per-image forward summed per minibatch (batch it like v8).
6. **CPU speed on Apple Silicon** — the CUDA seam doesn't help on Mac (Metal≠CUDA). Add a
   BLAS path to `bk::gemm_hosted` (Apple Accelerate / OpenBLAS) for a big CPU speedup without
   a GPU; a full Metal backend is a much larger, lower-priority effort.
7. **[DONE] Unified `yolo.cpp` verified under a CUDA build on free-Colab T4** (2026-07-23):
   `nvcc -x cu -std=c++17 --extended-lambda -DUSE_CUDA` builds clean; COCO128 runs end-to-end.
   Pretrained `init.pt` vals 0.59 mAP@0.5 / 0.44 @0.5:0.95 (matches yolov8n) — inference+mAP
   correct on real data. Fine-tune (`--lr 2e-4`) no longer collapses; close-mosaic recovery
   0.45→0.52 over 3 epochs. See `colab/coco128_train.ipynb`.
8. **[TOP SPEED PRIORITY] Device-resident buffers + cuBLAS** — measured ~4.6 min/epoch on T4
   (COCO128/640/batch4) => ~8 h for 100 epochs, only ~3× CPU. The `bk::gemm_hosted` backend
   copies host↔device per op and uses a naive kernel, capping GPU use far below a T4's
   potential (Ultralytics does COCO128 in seconds/epoch). Keep activations device-resident
   across ops and route GEMM through cuBLAS. (On CPU, 640px is ~a day for 100ep — measured
   ~5.7 s/image fwd+bwd; smaller imgsz or the Apple-Silicon BLAS path (#6) helps.)

## Notes / gotchas
- Label coords: internally everything is xyxy in the **letterboxed SxS pixel** space; GT and
  decoded detections share it, so val mAP is apples-to-apples. `load_boxes_orig` reads either
  format into original pixels, then `lb_map` applies the letterbox transform.
- `make_init_pt … rand` trains but won't converge on tiny data; use `from <pretrained.pt>`
  for transfer learning.
- Build: MSVC via `C:/prog/claude/cc.sh`; `scratch/` must pre-exist; vcvars hangs here.
