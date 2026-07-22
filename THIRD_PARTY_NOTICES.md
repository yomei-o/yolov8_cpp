# Third-party notices / サードパーティ ライセンス表記

This repository bundles third-party components. Each retains its own license and
copyright; the notices below apply to those files only.

このリポジトリには第三者の成果物を同梱しています。各ファイルはそれぞれの
ライセンス・著作権に従います。以下の表記はそれらのファイルにのみ適用されます。

---

## 1. Ultralytics YOLOv8 — model weights & sample images

- **Files:**
  - `yolov8n.pt` — original Ultralytics weights
  - `weights/yolov8n/weights.bin`, `weights/yolov8n/manifest.txt` — BN-folded repack of
    `yolov8n.pt`'s parameters (a derivative of the weights above)
  - `assets/bus.jpg`, `assets/zidane.jpg` — standard Ultralytics test images
  - reference tensors generated from the model under `pure/ref/` (git-ignored)
- **Copyright:** © Ultralytics
- **Source:** https://github.com/ultralytics/ultralytics
- **License:** **GNU Affero General Public License v3.0 (AGPL-3.0-or-later)**
  — full text: https://www.gnu.org/licenses/agpl-3.0.html

> Note: because these AGPL-3.0 artifacts are redistributed here, any redistribution of
> this repository (or a network service built from it) must comply with the AGPL-3.0
> terms for them. `weights/yolov8n/weights.bin` is a derived form of `yolov8n.pt` and is
> covered by the same license.
>
> 注: 上記は AGPL-3.0 です。本リポジトリを再配布・ネットワーク公開する場合、これらの
> 成果物について AGPL-3.0 の義務（ソース開示等）を満たす必要があります。`weights.bin`
> は `yolov8n.pt` の派生物であり同ライセンスの対象です。

## 2. stb — single-header image libraries

- **Files:** `pure/third_party/stb_image.h`, `pure/third_party/stb_image_write.h`
- **Author:** Sean Barrett and contributors
- **Source:** https://github.com/nothings/stb
- **License:** dual-licensed — **Public Domain (Unlicense)** OR **MIT**, at your option.
  The full license text is included at the bottom of each header file.

---

The project's own source code (the `pure/` engine, `ref/` scripts, `step*.cpp`, and the
docs) was written for this repository. It does not yet declare a top-level license — add
a `LICENSE` file to state the terms you want for that code. Note that distributing it
together with the AGPL-3.0 artifacts above carries the AGPL obligations for those files.
