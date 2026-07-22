# Python の YOLO 学習を C++ に移植する手順 (PORTING GUIDE)

このリポジトリで yolov8n を C++ 学習に移植したときの**方法論**をまとめる。
yolov5 / yolov11 など他モデルでも同じ手順で進められるように書いてある。
(A methodology to port a Python YOLO's *training* to C++, reproducible for other models.)

---

## 0. 中心となる考え方 (the core ideas)

1. **PyTorch は「答え合わせ」にしか使わない。** 各ステップで Python が入力と正解出力を
   出し、C++ がそれに一致するかを数値照合する。forward はビットレベル、勾配は ~1e-8 で合う。
2. **逆伝播は書かない。** autograd (LibTorch でも自作エンジンでも) が backward をやる。
   人間が移植するのは **損失の順伝播だけ**。
3. **微分が要る所と要らない所を分ける。** ラベル割り当て (YOLOv8 の TAL) は
   `@torch.no_grad()`。ここは素の配列処理で書く (autograd に乗せない)。
4. **2 トラックで進める。**
   - **Track A (LibTorch 併用):** 最短で「C++ で学習が回る」に到達。autograd は LibTorch。
   - **Track B (依存ゼロ):** 自作 autograd + 全 op を手書き。標準ライブラリのみ。
     Track A で正しさを固めてから移すと安全。

---

## 1. 検証ハーネス (verification harness)

全ステップ共通のパターン。シリアライズ形式やライブラリ版に依存しない一番堅い方法:

- Python が入力と参照出力を **raw float32 の .bin** で書き出す (`tensor.numpy().tofile`)。
- C++ が同じ .bin を読み、同じ計算をし、`max|diff|` を出す。
- 判定: forward は `< 1e-4`、勾配は `< 1e-5`。
- 自作エンジンの op 単体は **有限差分による gradcheck** で検証 (外部依存ゼロで自己完結、
  `pure/gradcheck*.cpp`)。判定は PyTorch 流の `atol + rtol*|ana|` (float32 の有限差分は
  勾配が小さい要素で相対誤差が発散するため、純粋な相対誤差では誤判定する)。

---

## 2. Track A: LibTorch 併用で「本物を学習」まで

順番と、各ステップの対応ファイル (yolov8):

1. **DFL デコード + CIoU** (`step1`) — 箱の分布 (reg_max=16 bins) を softmax→期待値→ltrb→
   `dist2bbox`。CIoU は箱回帰の損失。
2. **TAL** (`step2`) — `TaskAlignedAssigner`。候補選択→アライメント指標(cls^α·CIoU^β)→
   topk→重複解決→ターゲット収集→正規化。`@torch.no_grad()` なので `NoGradGuard` で囲む。
3. **v8 loss forward + backward** (`step3a/b/c`, `v8loss.h`) — cls=BCEWithLogits,
   box=CIoU, dfl=Distribution Focal Loss。`loss = 7.5*box + 0.5*cls + 1.5*dfl`,
   `target_scores_sum = max(target_scores.sum(), 1)`。**勾配一致まで**確認する。
4. **学習ループ** (`step4` 合成CNN → `step5` 本物 yolov8n)。

### 本物モデルを C++ に持ち込む方法 (重要)
- `model.export(format=torchscript)` は**使わない** (推論用にデコード済み出力を吐く)。
- 代わりに **DetectionModel を training モードのまま `torch.jit.trace`** して、
  Detect ヘッドの生出力を得る。ラッパーで返り値を**同種型の tuple** に整える:
  ```python
  class Wrap(nn.Module):
      def forward(self, x):
          y = self.m._predict_once(x)   # training: dict{boxes,scores,feats}
          return y["boxes"], y["scores"]
  ts = torch.jit.trace(Wrap(dm).train(), example, strict=False, check_trace=False)
  ```
- C++ 側: `torch::jit::load` → `mod.train()` → `mod.parameters()` を optimizer へ →
  forward の tuple から boxes/scores を取り出して `v8loss.h` に渡す。

---

## 3. Track B: 依存ゼロ (自作 autograd)

`pure/` 以下。ビルドは **`g++ -std=c++20 -O2 [-fopenmp]` 一発**。CMake も LibTorch も不要。

マイルストーン:
- **M1 エンジン核** (`autograd.hpp`) — tape 方式の reverse-mode autograd。
  Node = data + grad + backward closure + parents。`backward()` は topological 逆順。
- **M2 空間 op** — conv2d/maxpool/upsample/concat/slice の forward + **backward を手書き**。
  conv は入力・重み・バイアスの 3 方向。gradcheck で検証。
- **M3 全順伝播** (`blocks.hpp`, `net.hpp`) — Conv/C2f/Bottleneck/SPPF/neck/Detect を
  組む。重みは Python から**融合済み** (下記) でフラット .bin に出し、C++ が順に読む。
- **M4 損失** (`ops2d.hpp`, `v8pure.hpp`) — 微分 op を追加 (sub/div/min2/max2/atan/exp/
  log/softmax_rows/log_softmax_rows/gather_row 等)。CIoU は**プリミティブ op から構成**
  すれば backward は autograd 任せ。TAL のターゲットは定数として注入。
- **M5 TAL + 学習** (`tal.hpp`, `m5_train.cpp`) — TAL を素の C++ 配列で移植 (M5a)、
  forward→pack→TAL→loss→backward→SGD の学習ループ (M5b)。

### CPU / OpenMP / GPU の切替
- 重いループに `#pragma omp parallel for` を付けておく。**`-fopenmp` の有無だけで**
  CPU⇔OpenMP が切り替わる (プラグマは無効時ただ無視される、ソース不変)。
- GPU は本質的に CUDA 依存。conv 等を `#ifdef USE_CUDA` のバックエンド境界の裏に隠して後差し。

---

## 4. 落とし穴リスト (gotchas) — ここで詰まった

1. **BN の融合とモード。** 推論では BatchNorm を conv に融合できる:
   `w' = w·γ/√(var+ε)`, `b' = β - γ·mean/√(var+ε)`。
   **参照出力を作るとき**は Detect を training に保ちつつ **BN だけ eval** にする
   (`for m in model.modules(): if isinstance(m, BatchNorm2d): m.eval()`)。でないと
   融合 (running 統計) と BN (バッチ統計) が食い違い、出力が**全く合わない**。
2. **CIoU の α 項は `torch.no_grad()`。** forward は同じでも `.detach()` しないと
   backward がズレる。
3. **`torch::tensor` のネスト初期化子は同一スカラー型必須** (`8.` は double, `8` は int で
   落ちる)。混在リテラルは `torch::from_blob` で作る。
4. **LibTorch 2.13+ は C++20 必須** (designated-init / bitfield-init)。
   さらに TorchScript を読むなら **LibTorch と、script した Python torch の版を揃える**
   (bytecode version の不整合で load 失敗)。
5. **ultralytics の training 出力は版で変わる。** `_predict_once` が list→dict
   (`{boxes,scores,feats}`) に変化していた。ラッパーで固定するのが安全。
6. **アンカー順序。** `make_anchors` は +0.5 offset、レベル連結、各レベル内は行優先 (y→x)。
   これが `(b,C,h,w)→flatten` の順と一致していないと全部ズレる。
7. **チャンネル順。** boxes は `[4*reg_max(分布)][nc(クラス)]`。DFL は `(...,4,reg_max)` に
   reshape してから bin 方向 softmax。
8. **学習率。** 事前学習済み重み + SGD では大きい lr で即 NaN。`1e-4` 前後から。
   (LibTorch 版は Adam 1e-3 で安定)

---

## 5. yolov5 / yolov11 への展開 (adapting to other models)

移植は **「ネット構造」と「損失」を分けて**考える。検証ハーネス (§1) と方法論 (§0) は共通。

### yolov11 (v8 に近い — 楽)
- **損失は v8 と同系** (TAL + DFL + CIoU + BCE)。`v8loss.h` / `pure/v8pure.hpp` /
  `pure/tal.hpp` はほぼそのまま流用できる。
- **変わるのはネット構造だけ。** ブロックが `C3k2`, `C2PSA` (attention) 等に変わる。
  `pure/net.hpp` に対応ブロックを追加すればよい。C2PSA の attention は softmax/matmul が
  要るが、`ops2d.hpp` の部品で組める (matmul は追加が必要)。
- 手順: (a) `_predict_once` の training 出力形を確認 → (b) 融合重みを canonical 順で
  ダンプ → (c) net を組んで forward 一致 (M3c 相当) → (d) 損失は流用して forward/backward
  一致 → (e) 学習ループ。

### yolov5 (旧世代 — 損失が別系統)
- **anchor ベース**。各セル 3 anchor、出力は `(b, 3*(5+nc), h, w)` で **objectness** を含む。
- **損失が違う:** `ComputeLoss` = box(CIoU) + obj(BCE) + cls(BCE)。**DFL なし・TAL なし**。
  代わりに **`build_targets` (wh 比によるアンカーマッチング)** を no-grad で移植する。
- **デコードが違う:** `xy = (sigmoid*2-0.5+grid)*stride`, `wh = (sigmoid*2)^2 * anchor`。
- ネットは Focus(旧)/Conv/C3/SPPF。ブロックを net.hpp に足す。
- つまり **損失部分は書き直し**、検証ハーネスと autograd エンジンは再利用。

---

## 6. まとめ (TL;DR)
1. Python を oracle にして .bin で数値照合。forward→勾配の順に一致させる。
2. 逆伝播は autograd 任せ。移植するのは損失の forward と (no-grad の) ラベル割り当てだけ。
3. まず LibTorch 併用で学習を通し、次に op を手書きして依存ゼロへ。
4. モデルを替えるときは「ネット構造」と「損失」を分けて考える
   (v11≈v8 は損失流用、v5 は損失書き直し)。
