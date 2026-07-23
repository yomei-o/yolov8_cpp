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
7. **Verify the unified `train_cli`/`yolo.cpp` under a CUDA build** — compile with
   `nvcc -DUSE_CUDA` and run COCO128 end-to-end on a (free-Colab) T4. The CUDA seam
   (`backend.hpp`) + a training loop were verified on T4, but the new dataset-ingestion +
   augmentation CLI path hasn't been built/run under nvcc yet (aug/dataset are host-side, so
   it should work; conv/matmul auto-route to `bk::` on GPU). Est. COCO128/640px/100ep on a
   T4 ≈ 7–20 min. On CPU it is impractical: measured ~5.7 s/image fwd+bwd at 640px
   (~4.6 GFLOP/s effective, naive GEMM) => COCO128/640px/100ep ≈ ~a day (20–30 h) on
   M2-class CPU. Smaller imgsz (320 ≈ 4× faster) or a BLAS backend (#6) helps; GPU is the fix.

## Notes / gotchas
- Label coords: internally everything is xyxy in the **letterboxed SxS pixel** space; GT and
  decoded detections share it, so val mAP is apples-to-apples. `load_boxes_orig` reads either
  format into original pixels, then `lb_map` applies the letterbox transform.
- `make_init_pt … rand` trains but won't converge on tiny data; use `from <pretrained.pt>`
  for transfer learning.
- Build: MSVC via `C:/prog/claude/cc.sh`; `scratch/` must pre-exist; vcvars hangs here.
