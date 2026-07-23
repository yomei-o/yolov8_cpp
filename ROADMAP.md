# ROADMAP — 今後の実装予定 (planned work)

現状: yolov8n の学習を C++ で再現済み (LibTorch 版 + 依存ゼロ版、`README.md` 参照)。
以下は明日以降やりたいこと。各項目に着手用の技術メモを添える。
(Status: yolov8n training reproduced in C++. Items below are next steps, with notes.)

---

## A. モデル入出力 (model I/O)

### A-1. `.pt` ファイルの読み書き
- `.pt` は **ZIP アーカイブ + pickle + tensor storages**。純 C++ で直接読むには zip 展開 +
  pickle パーサが要る (中規模)。まずは **Python ブリッジ**で実用化する:
  - 読み: `state_dict()` を走査して融合重み or 生重みを flat `.bin` + manifest に出す
    (現状 `pure/ref/m3c_export.py` がこれ。汎用化する)。
  - ✅ **書き戻しは実装済み** (`pure/m9_train_writeback.cpp` + `pure/ref/writeback_pt.py`):
    C++ が unfused な重み (conv + BN γ/β/running stats) を flat `.bin` に出す → Python が
    `yolo_walk.py` の正準順で state_dict に流し込み `.pt` 化 → Ultralytics がそのまま推論。
    シリアライズは完全一致 (0.0)、未学習重みでは forward も ~2e-5 で一致、C++ 学習後の重みでも
    Ultralytics がロード・推論できることを確認済み。
- **重要 (BN の扱い):** 現在は推論用に BN を conv に**融合**している。学習した重みを `.pt` に
  戻して再学習/微調整するには、**BN を融合せず独立の学習可能 op として持つ**必要がある
  (→ エンジンに BatchNorm op を追加、下記 C-1)。融合版は「推論専用の書き出し」に留める。
- 将来: zip + pickle の最小 reader/writer を C++ に実装して Python 非依存の `.pt` I/O。

### A-2. ONNX 入出力 ✅ 実装済み
- **エクスポート:** ✅ 自前の protobuf writer (`pure/onnx.hpp`) + `pure/onnx_export.cpp` で
  順伝播を標準 ONNX (opset 13: Conv / Sigmoid+Mul / MaxPool / Resize / Concat / Add / Slice)
  として出力。**onnxruntime で実行して本家 forward と一致 (~2e-5)**、`onnx.checker` も通過
  (n=223, m=289 ノード)。外部ライブラリ不使用。
- **インポート (本命):** ✅ `pure/onnx.hpp` の protobuf reader + `pure/onnx_run.hpp` の
  **グラフ駆動インタプリタ**で、`.onnx` をパースして pure ops で実行 (`pure/m13_onnx_run.cpp`)。
  アーキ manifest も重み Provider も不要で `.onnx` だけから forward を再現 (~2e-5)。
  onnx ライブラリで再シリアライズした (packed) ファイルも読める。
  - 残り: 対応 op は yolov8 backbone+head の範囲。Ultralytics が in-graph デコード込みで
    出力する ONNX (Reshape/Transpose/Softmax/Sub/Div/Gather 等) を読むには op を追加。
- 注意: ONNX は推論/交換用、**学習は自作エンジン**のまま。

---

## B. s / m / l / x モデル対応 (scaling)

- yolov8 の n/s/m/l/x は **depth/width 乗数**と max_channels が違うだけで構造は同系
  (depth 0.33/0.33/0.67/1.0/1.0, width 0.25/0.5/0.75/1.0/1.25)。C2f の繰り返し数 (n_bott) と
  各層チャンネルが変わる。
- ✅ **データ駆動化 実装済み** (`pure/net_dyn.hpp`, `pure/m12_dyn.cpp`,
  `pure/ref/export_arch.py`): エクスポータが層グラフ (type / from / C2f depth など) を
  `arch.txt` に書き出し、汎用ビルダがそれを読んで組み立てる。**yolov8n / s / m** の順伝播が
  本家と一致 (~1e-4) することを確認 (n=63 conv, m=83 conv, 同一コード)。l/x も `export_arch.py`
  で重みを出すだけで動くはず。
  - 残り: A-2 (ONNX import) と統合すれば任意モデルが読み込みで動く。学習・推論デモを各サイズで。

---

## C. エンジン拡張 (engine)

- ✅ **C-1. BatchNorm を学習可能 op に** — 実装済み (`pure/bn.hpp`, `pure/m7_bn.cpp`)。
  forward/backward + running stats 更新、train/eval 両モードで torch と一致 (~1e-7)。
  unfused な conv+BN+SiLU 版フォワード (`pure/net_unfused.hpp`) も本家と一致 (~2e-5)。
- ✅ **C-2. optimizer** — 実装済み (`pure/optim.hpp`, `pure/m10_optim.cpp`)。SGD
  (momentum / weight decay / Nesterov) と Adam / AdamW を torch.optim と一致 (~1e-7) で検証。
  残り: LR スケジュール (cosine/warmup) と学習ループへの本格組み込み。
- ✅ **C-3. matmul op** — 実装済み (`pure/linalg.hpp`: matmul + transpose2d, `pure/m11_matmul.cpp`)。
  forward/backward を torch と一致 (~1e-7) で検証。v11 の C2PSA (attention, 下記 D-1) の土台。
- **C-4. CUDA バックエンド**: ✅ **シム土台 実装済み** (`pure/backend.hpp`, `pure/m17_gpu.cpp`)。
  単一ヘッダで `bk::Buffer<T>` (device メモリ) + `bk::parallel_for(n, [=] BK_HD (i){...})` +
  `bk::gemm` を提供。**同一ソースが nvcc(-DUSE_CUDA) で実 CUDA、g++/MSVC で CPU** にコンパイル
  (CPU パスが GPU 無し時の「エミュ」)。CPU で vecadd/gemm を検証 (diff 0)。
  ビルド: `nvcc -x cu -std=c++17 --extended-lambda -DUSE_CUDA -O2 pure/m17_gpu.cpp`。
  - 残り (要 GPU 実機/クラウド): (1) 実 GPU で m17 実行確認、(2) conv の im2col+GEMM と
    elementwise を `bk::` 経由に差し替え (Tensor データを `bk::Buffer` に載せる)、(3) 学習
    ループを device 常駐化 (毎 iter の H2D/D2H を避ける)。GPU が無いこのマシンでは CPU パスの
    正しさと CUDA パスのコンパイル整合まで。実行/性能は Colab 等の GPU で。
- **C-5. 速度**: ✅ **im2col + GEMM 化 実装済み** (`pure/autograd.hpp` conv2d)。順・逆とも
  パッチを (K,P) 行列に集約し、連続メモリの内ループを自動ベクトル化＋ `parallel_for` で
  並列化。順伝播 640 が 11.9s → 2.2s (約5.4x)、学習 iter(160) が 0.99s → 0.32s (約3x)。
  残り: キャッシュブロッキング、バッファ再利用 (malloc 削減)、fp16、複数画像バッチ。

---

## D. 他モデル (other models) — 詳細は `PORTING_GUIDE.md` §5

- **D-1. yolov11**: 損失は v8 系そのまま流用 (`v8loss.h`/`v8pure.hpp`/`tal.hpp`)。
  `net.hpp` に C3k2 / C2PSA ブロックを追加 (C2PSA は matmul op が要る → C-3)。
- **D-2. yolov5**: anchor ベース + objectness。**損失を書き直し**
  (box CIoU + obj BCE + cls BCE、DFL/TAL なし、`build_targets` の wh 比マッチング、
  デコードも別式)。検証ハーネスと autograd エンジンは再利用。

---

## E. 学習・推論パイプライン (pipeline)

- **E-1. 実データローダ**: 画像読み込み (letterbox リサイズ) + ラベル読み込み +
  gt 正規化 (画像座標 ↔ stride 単位)。現状は合成バッチ。
  - ✅ **実装済み** (`pure/dataset.hpp`, `pure/m14_train_real.cpp`): stb_image で画像を読み
    letterbox、YOLO ラベルを読んで gt を letterbox 画像座標に変換。実画像＋実ラベル
    (Ultralytics 検出を疑似 GT に) で unfused BN 学習 → TAL → v8 loss → Adam + cosine LR、
    loss が低下。**ミニバッチ対応済み** (`Batch load_batch()` が複数画像を letterbox して
    GT をパディング＋マスク; bus+zidane の 2 枚バッチで loss 8.4 → 1.2)。推論描画は
    `pure/m6_demo.cpp`。残り: データ拡張、チェックポイント運用。
- **E-2. 推論経路**: ✅ **実装済み** (`pure/infer.hpp` + `pure/m6_infer.cpp`):
  DFL デコード + アンカー/stride + sigmoid + クラス別 NMS。Ultralytics の eval 出力と
  NMS 結果に一致 (box ~8e-5, 同一クラス・件数)。デコードは xyxy 出力
  (本バージョンの Ultralytics ヘッドに合わせた)。
- ✅ **E-3. 評価**: 実装済み (`pure/metrics.hpp`, `pure/m15_map.cpp`)。COCO 方式 mAP
  (area=all, maxDets=100, IoU 0.50:0.05:0.95, 101点補間) を **pycocotools と一致 (~3e-7)** で検証。
- **E-4. 複数バッチ / 学習の実運用化** (チェックポイント保存 = A-1 と連動)。

---

## 優先順位の目安 (suggested order)
1. **C-1 (BN op)** → A-1 (.pt 書き戻し) — 学習した重みを実際に使えるようにする
2. **B (データ駆動 net) + A-2 (ONNX import)** — s/m/l/x と他モデルの土台
3. **E-1/E-2 (データローダ + NMS)** — 合成から実データへ
4. **D-1 (yolov11)** → D-2 (yolov5)
5. C-4 (CUDA), C-5 (速度) は随時

> メモ: 粒度の細かいタスクは GitHub Issues に切ってもよい (`gh issue create`)。
