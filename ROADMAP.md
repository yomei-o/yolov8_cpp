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
  - 書き戻し: C++ が学習した重みを flat `.bin` に出す → Python が `model.state_dict()` に
    流し込んで `torch.save` で `.pt` 化 → Ultralytics でそのまま推論できるようにする。
- **重要 (BN の扱い):** 現在は推論用に BN を conv に**融合**している。学習した重みを `.pt` に
  戻して再学習/微調整するには、**BN を融合せず独立の学習可能 op として持つ**必要がある
  (→ エンジンに BatchNorm op を追加、下記 C-1)。融合版は「推論専用の書き出し」に留める。
- 将来: zip + pickle の最小 reader/writer を C++ に実装して Python 非依存の `.pt` I/O。

### A-2. ONNX 入出力
- **エクスポート (推論相互運用):** 順伝播を ONNX グラフ (protobuf) として出力。まずは
  Python 側 (`torch.onnx.export` / `ultralytics export`) を使い、C++ 推論結果と照合。
- **インポート (これが本命):** ONNX を C++ でパースして**グラフ駆動で forward を構築**すれば、
  アーキをハードコードせずに済む → s/m/l/x や v11 が「読むだけ」で動く (下記 B と直結)。
  protobuf パーサ + 対応 op のディスパッチテーブルを用意する。
- 注意: ONNX の学習サポートは限定的。ONNX は推論/交換用、**学習は自作エンジン**のまま。

---

## B. s / m / l / x モデル対応 (scaling)

- yolov8 の n/s/m/l/x は **depth/width 乗数**と max_channels が違うだけで構造は同系
  (depth 0.33/0.33/0.67/1.0/1.0, width 0.25/0.5/0.75/1.0/1.25)。C2f の繰り返し数 (n_bott) と
  各層チャンネルが変わる。
- 現状 `pure/net.hpp` は n 用に **接続と n_bott をハードコード**している。対応方針:
  - **データ駆動化**: エクスポータが「各層の型・チャンネル・n_bott・接続 (from)」を manifest に
    書き出し、`net.hpp` がそれを読んで汎用に組み立てる。yaml (`ultralytics/cfg/models`) を
    パースしても同じ。
  - これができると s/m/l/x は manifest 差し替えだけ、さらに A-2 (ONNX import) と統合すれば
    任意モデルが読み込みで動く。
- 検証は各サイズで M3c (順伝播一致) → 学習、と同じ刻みで。

---

## C. エンジン拡張 (engine)

- **C-1. BatchNorm を学習可能 op に** (forward/backward + running stats 更新)。融合をやめて
  conv+BN+act を別々に持てば `.pt` 学習ラウンドトリップ (A-1) が可能になる。
- **C-2. optimizer**: 現状 SGD 直書き。Adam / momentum / weight decay / LR スケジュール。
- **C-3. matmul op**: v11 の C2PSA (attention) に必要 (下記 D-1)。
- **C-4. CUDA バックエンド**: conv 等を `#ifdef USE_CUDA` 境界の裏に隠して差し込む
  (CPU/OpenMP はフラグ切替で維持)。
- **C-5. 速度**: 現状 naive conv。im2col + GEMM、キャッシュブロッキング、fp16。

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
  - ✅ **画像側は実装済み** (`pure/m6_demo.cpp`): stb_image で JPEG/PNG を読み、
    letterbox + bilinear リサイズして推論、結果を元画像に描画して stb_image_write で出力。
    残り: ラベル読み込みと gt 正規化 (学習用データローダ)。
- **E-2. 推論経路**: ✅ **実装済み** (`pure/infer.hpp` + `pure/m6_infer.cpp`):
  DFL デコード + アンカー/stride + sigmoid + クラス別 NMS。Ultralytics の eval 出力と
  NMS 結果に一致 (box ~8e-5, 同一クラス・件数)。デコードは xyxy 出力
  (本バージョンの Ultralytics ヘッドに合わせた)。
- **E-3. 評価**: mAP 計算。
- **E-4. 複数バッチ / 学習の実運用化** (チェックポイント保存 = A-1 と連動)。

---

## 優先順位の目安 (suggested order)
1. **C-1 (BN op)** → A-1 (.pt 書き戻し) — 学習した重みを実際に使えるようにする
2. **B (データ駆動 net) + A-2 (ONNX import)** — s/m/l/x と他モデルの土台
3. **E-1/E-2 (データローダ + NMS)** — 合成から実データへ
4. **D-1 (yolov11)** → D-2 (yolov5)
5. C-4 (CUDA), C-5 (速度) は随時

> メモ: 粒度の細かいタスクは GitHub Issues に切ってもよい (`gh issue create`)。
