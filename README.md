# yolov8_cpp

Training **YOLOv8** in C++ — first with LibTorch, then with **zero external
dependencies** (a from-scratch autograd engine, C++ standard library only).

日本語: 本物の YOLOv8 を C++ で学習する実験。まず LibTorch で損失の順伝播・逆伝播が
本家と一致することを確かめ、実物の yolov8n を C++ から学習。さらに **依存ライブラリを
完全に外して**、自作 autograd エンジン（標準ライブラリのみ）で yolov8n の順伝播・損失・
学習を再現する。CPU / OpenMP は `-fopenmp` の有無だけで切り替え。

Every step is **verified numerically against PyTorch / Ultralytics** (bit-level on
forward, ~1e-8 on gradients).

**→ [PORTING_GUIDE.md](PORTING_GUIDE.md): how to port a Python YOLO's training to C++**
(the methodology, gotchas, and how to adapt it to yolov5 / yolov11).

## Two tracks

### 1. LibTorch track — port the v8 loss, then train the real yolov8n
The hard, non-obvious parts of `v8DetectionLoss` are hand-ported to C++ and checked
against Ultralytics; LibTorch's autograd handles the backward pass.

| file | what it verifies | result |
|------|------------------|--------|
| `step1_dfl_ciou.cpp` | DFL decode + CIoU | bit-identical |
| `step2_tal.cpp` | TaskAlignedAssigner (full) | bit-identical |
| `step3a/b/c` | DFL loss / 3-loss forward / **backward grads** | ~1e-8 |
| `step4_train.cpp` | tiny CNN training loop | loss ↓ |
| `step5_train_yolov8n.cpp` | **load real yolov8n (TorchScript) and train** | loss 13.8 → 0.9 |

`v8loss.h` is the reusable loss module. `ref/*.py` generate the references.

### 2. Pure track (`pure/`) — no LibTorch, no CMake, just `g++`
A minimal reverse-mode autograd engine over dense tensors, then all of yolov8n on top.

| file | milestone | result |
|------|-----------|--------|
| `pure/autograd.hpp` | engine core + spatial ops (conv/pool/upsample/concat) | gradcheck OK |
| `pure/gradcheck*.cpp` | finite-difference gradient checks | max err ~1e-4 |
| `pure/net.hpp` + `pure/m3c_forward.cpp` | **full yolov8n forward** | matches real net ~3e-5 |
| `pure/v8pure.hpp` + `pure/m4bc_loss.cpp` | v8 loss forward + **backward** | grads match torch ~1e-9 |

BatchNorm is folded into the preceding conv; `pure/ops2d.hpp` holds the extra
differentiable ops used by the loss.

## Build

Pure track (nothing but a compiler):
```sh
g++ -std=c++20 -O2            pure/m3c_forward.cpp -o m3c.exe   # CPU
g++ -std=c++20 -O2 -fopenmp   pure/m3c_forward.cpp -o m3c.exe   # OpenMP (same result)
```

LibTorch track (CMake + LibTorch 2.13 CPU, C++20):
```sh
cmake -B build -S . -A x64
cmake --build build --config Release
```

References (needs `pip install ultralytics torch`):
```sh
python pure/ref/m3c_export.py     # dump yolov8n weights + reference output
python ref/loss_ref.py            # dump loss/grad reference
```

## Status
DFL decode, CIoU, TAL, the full v8 loss (forward + backward), the full yolov8n
forward, and training all reproduced and verified. Remaining: port TAL into the
pure engine and close the pure end-to-end training loop; optional CUDA backend
behind the conv seam.
