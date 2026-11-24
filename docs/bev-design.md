# 床の上の複数カメラの追跡の設計（本体との違い・with-mopf との違い）

[English](bev-design.en.md)

2026-11-25。フォーク `kawanoKen/twinsim`、ブランチ `feature/bev`。土台はブランチ `feature/camera-groups`（`upstream/main` の v1.2.0 に `camera_groups` だけを足したもの。main への取り込みの候補）。

視野の重なるカメラの群ごとに、人を床の上（真上から見た図）で 1 つの ID で追い、シミュレータの正解と比べる機能のまとめ。
本体に何を足したか、何を読んで何を書くか、描画からどう動かすか、ブランチ `with-mopf` の MOPF の連携と何が違うかを書く。
追跡のしかた（アルゴリズム）の中身はここには書かない（使い方と設定は [bev-tracking.md](bev-tracking.md)、計画と進み具合は [bev-tracking-plan.md](bev-tracking-plan.md)、ファイルの形は [formats.md の 12](formats.md#bev)）。

## 1. 全体の流れ

```
シナリオ（camera_groups を足す）─→ twinsim run（CPU）─→ 正解（ground_truth・mot・aicity・tasks・graph）
                                        │
                                        └→ twinsim render（GPU、Isaac Sim）─→ video/<カメラ>.mp4
                                                                                │
  twinsim bev-detect ─→ bev_dets/<検出器>/          正解の枠から（CPU）か PeopleNet（GPU、DeepStream）
         │
  twinsim bev-track ──→ bev_tracks/<名前>/<群>/      A（CPU）か MV3DT（GPU、DeepStream）
         │                 bev.csv（床の上の軌跡）と、正解と同じ形の mot/・aicity/・graph/・tasks.json
         │
  twinsim bev-eval ───→ bev_eval/                    床の上の指標・カメラごとの指標・人数・集計を正解と比べる
  twinsim bev-overlay → bev_overlay/                 上面図とカメラを並べた確かめの動画
```

- 本体の流れ（`run` → `render`）はそのまま。床の上の追跡は、本体の結果のフォルダを読んで、同じフォルダの `bev_*` に書く。
- 検出と追跡を分けた。検出は一度ファイルに残すので、同じ検出のまま追跡器だけを比べられる。
- 追跡の出力は、正解と同じ形（`mot/`・`aicity/`・`graph/`・`tasks.json`）でも書く。正解を読む道具がそのまま推定も読める。

## 2. 本体（main）に足したもの

### 2.1 本体を変えた所

変えたのは設定の 1 項目だけで、ブランチ `feature/camera-groups` に分けてある（このブランチの土台）。シミュレータの計算と、ほかの出力は変えていない。

| 場所 | 変えたこと |
|---|---|
| `twinsim/scenario.py` | 設定に `camera_groups` を足す（`{群: {cameras: [...], area: [[x, y], ...], note}}`）。群の名前、知らないキー、`cameras[]` にないカメラ、評価領域の多角形を確かめる（`camera_group_errors`） |
| `twinsim/outputs.py` | `ground_truth.json` に `camera_groups` を書く（なければ `{}`） |
| `docs/config-reference.md`・`formats.md` | `camera_groups` の説明（日英） |
| `tests/test_smoke.py` | 正解に残ること・人の動きが変わらないこと・検査 |

- 前の版の設定はそのまま読め、同じ設定と種なら結果も同じ。`ground_truth.json` に `camera_groups` のキーが増えるだけ。
- `camera_groups` を書かなければ、床の上の追跡は全部のカメラを 1 つの群 `all` とし、評価領域はカメラから床が見える所にする。

### 2.2 足した部分（`twinsim/bev/`）

本体のモジュールは読むだけで、`twinsim/bev/` の中で閉じている。

| 場所 | 役割 | 主な入口 | 要るもの |
|---|---|---|---|
| `calib.py` | 正解のカメラ → 3×4 の投影行列 | `calib`・`projection` | |
| `groups.py` | カメラの群と評価領域 | `groups`・`Group` | |
| `io.py` | 結果の場所を開く、正解の位置、検出と軌跡の読み書き | `open_run`・`gt_positions` | |
| `detect.py` | 検出: 正解の枠から（`gt_boxes`、見落とし・ずれ・誤検出を足せる）、PeopleNet（`peoplenet`）、外の検出（`external`） | `detect` | `peoplenet` は GPU |
| `deepstream.py` | DeepStream のコンテナを動かす共通の部分 | `peoplenet`・`run_container` | docker、DeepStream のコンテナ、モデル |
| `trackers/` | 追跡器（`a`・`mv3dt`）と入口 | `track` | `mv3dt` は GPU と mosquitto |
| `appearance.py` | A の窓最適化に使う胴の色 | `ensure` | OpenCV（任意の依存） |
| `presets/` | 追跡器の設定（場面ごと） | `--preset` | |
| `evaluate.py` | 床の上の指標・カメラごとの指標・人数・集計の比べ | `evaluate` | |
| `graph.py` | ゾーンの人数と移動（`graph/` と同じ形） | `graph` | |
| `tasks.py` | ゲートの通過・止まった区間・混み具合・EV の待ち時間（`tasks.json` と同じ形） | `predict`・`compare` | |
| `overlay.py` | 確かめの動画 | `overlay` | ffmpeg、描画した動画 |
| `synthetic.py` | 正解から作る追跡（評価の確かめ用） | `synthetic_tracks` | |

ほかに足したもの: コマンド 4 つ（`twinsim/__main__.py`）、テスト（`tests/test_bev.py`）、例（`examples/09_bev_tracking.py`）、文書（`docs/bev-*.md`、`formats.md` の 12、日英）、
任意の依存 `bev`（`pyproject.toml`。OpenCV。`uv sync --extra bev`）。

### 2.3 コマンド

| コマンド | すること | GPU |
|---|---|---|
| `twinsim bev-detect <結果>` | 検出 → `bev_dets/<名前>/` | `--detector peoplenet` だけ（`--gpu`） |
| `twinsim bev-track <結果>` | 群ごとの追跡 → `bev_tracks/<名前>/<群>/`。`--repeats`（繰り返し）・`--perturb`（設定を少し揺らした 15 通り） | `--tracker mv3dt` だけ（`--gpu`） |
| `twinsim bev-eval <結果> --tracker <名前>` | 正解と比べる → `bev_eval/` | 使わない |
| `twinsim bev-overlay <結果> --tracker <名前>` | 確かめの動画 → `bev_overlay/` | 使わない |

Python からは `twinsim.bev` の `detect`・`track`・`evaluate`・`overlay`・`synthetic_tracks`。

### 2.4 外のもの（PeopleNet・MV3DT を使うときだけ）

| 環境変数 | 中身 | 既定 |
|---|---|---|
| `TWINSIM_DEEPSTREAM_IMAGE` | DeepStream のコンテナ | `ghcr.io/nvidia/deepstream:9.2-triton-multiarch` |
| `TWINSIM_DEEPSTREAM_MODELS` | モデルの置き場所（`PeopleNet2.6.3/`、MV3DT は `BodyPose3DNet/` も。TensorRT のエンジンを置くので書ける所） | なし（`--models` でも渡せる） |
| `TWINSIM_MOSQUITTO` | MV3DT の MQTT の受け口（mosquitto の実行ファイル）。実行のたびに 127.0.0.1 の空いた番号で立てて止める | PATH の `mosquitto` |
| `TWINSIM_MQTT` | すでに動いている受け口を使うとき（`host:port`） | なし |

## 3. 入力と出力

### 3.1 本体の結果から読むもの

| ファイル | 使い方 |
|---|---|
| `ground_truth.json` | `cameras`（投影行列を作る）、`camera_groups`（群と評価領域）、`people`・`tracks`（正解の位置。部屋の中の人は除く）、`gates`・`cars`・`gate_events`・`people[].events`（ゲート・EV の正解） |
| `scenario_resolved.json` | 空間（ゾーン・歩ける所）。設定に書いた空間も読めるように、`ground_truth.json` の `space` より先に見る |
| `render_log.json` | 動画の 0 コマ目がシミュレータのどのコマか（なければ描画していないとみなし、シミュレータのコマのまま） |
| `video/<カメラ>.mp4` | PeopleNet・MV3DT の入力、胴の色、確かめの動画 |
| `mot/<カメラ>.txt` | 見えている割合（判定しない人を決める）、正解の枠からの検出、カメラごとの採点の正解 |
| `aicity/cameras.json` | カメラの番号（追跡の `aicity/` を正解と同じ番号にする） |
| `tasks.json` | 集計の正解（止まった区間・混み具合）。混み具合の段階の境い |

### 3.2 書くもの

```
runs/<名前>/
  （本体の出力: ground_truth.json・mot/・aicity/・tasks.json・graph/・video/ …）
  bev_dets/<検出器>/          <カメラ>.txt（検出の枠）・run.json・app_<カメラ>.npy（胴の色。使うときだけ）
  bev_tracks/<名前>/<群>/
    bev.csv                   床の上の軌跡（twinsim 独自の形: コマ・ID・位置・誤差・見えたカメラの数）
    mot/<カメラ>.txt          ┐
    aicity/tracks.txt         │ 正解と同じ形（下の表）
    aicity/cameras.json       │
    graph/zone_counts.csv     │
    graph/zone_flows.csv      │
    tasks.json                ┘
    run.json                  追跡器・設定・検出・かかった時間・twinsim の版と git のハッシュ
  bev_eval/                   <名前>_<群>.json・summary.csv・per_camera.csv
  bev_overlay/                <名前>_<群>.mp4
```

正解と同じ形で書くもの:

| 正解（本体） | 推定（追跡） | 違い |
|---|---|---|
| `mot/<カメラ>.txt` | `bev_tracks/…/mot/<カメラ>.txt` | ID は追跡器の ID。A の枠は床の位置から作った円柱の枠 |
| `aicity/gt.txt`・`cameras.json` | `bev_tracks/…/aicity/tracks.txt`・`cameras.json` | カメラの番号は同じ。ID は追跡器の ID（全カメラで同じ） |
| `graph/zone_counts.csv`・`zone_flows.csv` | `bev_tracks/…/graph/` 同じ名前 | 評価領域の中の点だけ |
| `tasks.json` | `bev_tracks/…/tasks.json` | 評価領域に入る所だけ（`observed`）。状態・グループ・行き先は推定しない（`not_estimated`） |

`bev-eval` は、床の上の指標（IDF1・HOTA・MOTA など）のほかに、上の推定を正解と比べた値（カメラごとの指標、ゾーンの人数、ゲートの通過、EV の待ち時間、止まった区間、混み具合）を `bev_eval/<名前>_<群>.json` に書く。

## 4. 動かし方

### 4.1 描画なしで（CPU だけ）

正解の枠に見落としなどを足した検出で、追跡と採点の流れを確かめる。

```bash
twinsim run config.json --out runs/test                    # outputs に mot と aicity を入れる。camera_groups を書く
twinsim bev-detect runs/test --name noisy --miss 0.1 --jitter-px 3 --fp-per-frame 0.3
twinsim bev-track runs/test --dets noisy --label a_noisy
twinsim bev-eval runs/test --tracker a_noisy
```

例: `uv run python examples/09_bev_tracking.py`（60 秒・5 台、30 秒ほど）。

### 4.2 描画した動画から

```bash
twinsim run config.json --out runs/test
./isaac_up.sh && ./run.sh render runs/test                  # 描画（GPU。所要時間と GPU を確かめてから）
export TWINSIM_DEEPSTREAM_MODELS=<モデルの置き場所>

twinsim bev-detect runs/test --detector peoplenet --gpu 3                        # PeopleNet の検出（群ごとに 1 回）
twinsim bev-track runs/test --dets peoplenet --preset office_gate_ev-gate --group gate --label a_gate   # A
twinsim bev-track runs/test --tracker mv3dt --gpu 3 --repeats 3                  # MV3DT（中で検出する。3 回）

twinsim bev-eval runs/test --tracker a_gate --frames 2000 3000                   # 動画のコマ 2000〜2999 を採点
twinsim bev-eval runs/test --tracker mv3dt --frames 2000 3000                    # 3 回の平均と標準偏差
twinsim bev-overlay runs/test --tracker a_gate --group gate --frames 2000 2300   # 確かめの動画
```

- GPU は `--gpu` で 1 枚を決める（既定はない）。`nvidia-smi` で空きを見てから。
- A の窓最適化は `--param be_mode=post`（後処理）か `closed`（閉ループ）。胴の色を使うときは OpenCV（`uv sync --extra bev`）と動画が要る。

### 4.3 かかる時間（`office_gate_ev`、在館 1,200 人・10 fps・300 秒・5 台 × 2 群、GPU 1 枚）

| 段 | 時間 |
|---|---|
| PeopleNet の検出（10 台） | 40 秒ほど |
| A（1 群） | 1 コマ 4〜8 ms（3,000 コマで 10〜25 秒、CPU） |
| MV3DT（3 回 × 2 群） | 10 分ほど |
| 採点（1 群） | 数秒〜数十秒 |

## 5. with-mopf との違い

ブランチ `with-mopf` の `twinsim/mopf/` は、外のアプリ MOPF（`~/MOPF_ALL_video`）で 1 台ずつ追跡し、その結果を正解と比べる連携。
床の上の追跡（`twinsim/bev/`）とは、追跡のしかたも、比べる場所も違う。

### 5.1 何が違うか

| | `twinsim/mopf/`（with-mopf） | `twinsim/bev/`（このブランチ） |
|---|---|---|
| 土台 | 2026-11-02 に main から分けた（今の main より 59 コミット前） | 今の main（v1.2.0）の上 |
| 追跡 | 1 台ずつ（MOPF。ID はカメラごと） | 視野の重なる群ごとにまとめて床の上で（A・MV3DT。ID は群で 1 つ） |
| 視野の重なるカメラ | 同じ人が同時に映ると別の人になる（つながない） | 1 人にまとめる |
| 視野の重ならないカメラの間 | `linking.py` がつなぐ（時間・向き・見た目） | まだない（段階 5 の予定） |
| 採点の場所 | 画面の上（IoU 0.3） | 床の上（1 m・0.5 m）。画面の上（IoU 0.3）も副に |
| 検出・追跡の実行 | MOPF のサーバーを起動して API で送る | DeepStream のコンテナをその場で動かす（A は CPU だけ） |
| 人流モニターの線とエリア | ある（`monitor.py`・`crossings.py`） | ない（ゲートの通過は `tasks.json` の形で推定） |
| 出力 | `mopf_tracks/`・`mopf_eval/`・`mopf_graph/` | `bev_dets/`・`bev_tracks/`・`bev_eval/`・`bev_overlay/` |
| GUI | 「MOPF」のタブ | ない（コマンドと Python だけ） |

### 5.2 重なっている機能

| 機能 | `twinsim/mopf/` | `twinsim/bev/` | 決まり |
|---|---|---|---|
| カメラごとの採点 | `evaluate.py` | `evaluate.per_camera` | 同じ（IoU 0.3、見えている割合 0.4 未満は判定しない） |
| ゾーンの人数・移動 | `graph.py`（`mopf_graph/`） | `graph.py`（`bev_tracks/…/graph/`） | どちらも本体の `graph/` と同じ形 |
| 重ねた動画 | `overlay.py`（1 台） | `overlay.py`（上面図と群のカメラ） | 見せ方が違う |
| 正解から作る追跡 | `synthetic.py` | `synthetic.py` | 同じ考え方 |

一緒にするなら、この 4 つを共通にできる。追跡（`mopf/process.py`・`bev/trackers/`）と、カメラの間をつなぐ所（`mopf/linking.py`）は役割が違うので分けたままにする。

### 5.3 組み合わせ方

視野の重なる群の中は床の上の追跡でまとめ、群と群の間（視野が重ならない）は `linking.py` の考え方でつなぐ、が本来の形（段階 5）。
`linking.py` はカメラの追跡（1 台の ID のひと続き）をつなぐので、群の追跡（`bev.csv` の ID のひと続き）を同じように渡せるようにする。

## 6. 既知の制限

- ゲートを出た人は多く数える: シミュレータでは、ゲートを使わずにレーンを通り抜ける人がいて、正解の `gate_events` に入らない（映像からは区別できない）。
- EV の待ち時間は、正解の位置から推定しても長めに出る（待つ場所に着く前の、人混みの中のゆっくりした歩きも止まっているとみなす）。
- 座っている人（1.2 の座る動作）は、枠が低くなるので、人の円柱に合わせる A の位置がずれる。まだ手を入れていない。
- MV3DT は同じ設定でも実行ごとに結果がぶれる（`--repeats` で平均を見る）。
- PeopleNet・MV3DT は DeepStream のコンテナとモデルが要る（CI では動かさず、テストはコンテナの呼び出しを差し替えて確かめる）。
