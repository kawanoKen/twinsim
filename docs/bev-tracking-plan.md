# 複数カメラの床の上の追跡（BEV 追跡）を足す計画

[English](bev-tracking-plan.en.md)

2026-11-16。フォーク `kawanoKen/twinsim`、ブランチ `feature/bev`（`feature/camera-groups` の上。前のブランチ `feature/bev-tracking` は `upstream/main` の `7dfe9a6`、1.1.0 から）。

視野の重なる複数のカメラで、同じ人を床の上（BEV）の 1 つの ID で追う追跡を、twinsim の中で「描画 → 検出 → 追跡 → 評価」と通しで動かせるようにする。
作者のロードマップの「1.3 2 台以上のカメラ」（視野が重なるカメラの組と重ならない組、カメラをまたいで同じ人を対応づける正解と評価）に沿う。
組み込み方は、ブランチ `with-mopf` の MOPF との連携（`twinsim/mopf/`、`mopf-process`・`mopf-eval`、`docs/integration.md`）と同じ形にそろえる。

追跡器は 2 つ入れる。

- **A**：全カメラの検出を床の上の 1 組のトラックへ直接割り当てる集中型のオンライン追跡（学習なし、CPU）。後付けの窓最適化・閉ループも選べる。
  元は `experiments/closed_loop/code/cltrack.py`（このリポジトリの外）。
- **公式 MV3DT**：NVIDIA DeepStream 9.2 の多視点 3D 追跡。カメラごとに追跡して、MQTT で ID をそろえる分散型。MOPF と同じく外のアプリとして呼ぶ（GPU）。

## 0. 進み具合

| 日付 | 段階 | 内容 |
|---|---|---|
| 2026-11-16 | 1 済み | シナリオの `camera_groups`・`twinsim/bev/`（`calib`・`groups`・`io`・`detect`（`gt_boxes`・`external`）・`trackers/a`・`evaluate`・`synthetic`）・`bev-detect`・`bev-track`・`bev-eval`・`tests/test_bev.py`・文書（[bev-tracking.md](bev-tracking.md)、`formats`・`config-reference` の日英）。全部のテストと ruff が通る |
| 2026-11-16 | 2 の一部 | 場面 2 の PeopleNet の検出を `external` で読み込み、A が外の実験の実装と全部のコマで同じ ID・位置の差 0.0005 m 以下になった（評価も motmetrics と同じ値）。残りは `detect peoplenet` |
| 2026-11-24 | 2 済み | `detect peoplenet`（`twinsim/bev/deepstream.py`。群ごとに DeepStream の `deepstream-app` で検出だけを書き出す）。場面 2 で外の実験の検出と全部の枠が同じ値、A がゲート 0.761・EV 前 0.637（判定しない人の決まりありで 0.768・0.699）を再現（GPU 1 枚で 40 秒）。検出の枠は小数 6 桁で書く（2 桁に丸めると A の値がわずかに変わった） |
| 2026-11-24 | 3 済み | `trackers/mv3dt`（設定の生成・この計算だけの MQTT の受け口・`deepstream-test5-app`・出力の変換）、`bev-track --repeats`、`bev-eval` の平均と標準偏差。場面 2 で 3 回の平均がゲート 0.640 ± 0.021、EV 前 0.390 ± 0.011（外の実験 0.639・0.395）。GPU 1 枚で 3 回 × 2 群が 10 分 |
| 2026-11-24 | 4 の一部 | カメラごとの評価（`per_camera.csv`。評価領域の外の人は判定しない）、`bev_graph/`（ゾーンの人数・移動と正解の比べ）、`bev-overlay`、A の 15 通りの揺らし（`--perturb`。ゲート 0.764 ± 0.006・EV 前 0.635 ± 0.011、外の実験 0.763・0.635）、`examples/09_bev_tracking.py`。残りは A の窓最適化・閉ループ（`finish`）、ゲートの通過・EV の待ち時間の比べ |
| 2026-11-24 | 4 の続き | A の窓最適化（`trackers/backend.py`、`be_mode` = `post`・`closed`）と見え方（`appearance.py`、OpenCV は任意の依存 `bev`）。場面 2 の値が外の実験の今の実装と同じ（ゲート 後処理 0.783、閉ループ 0.774・訂正 0.789、EV 前 0.586、0.579・0.596）。外の実験の記録（`robust.json`）の C の値（0.789・0.809）は、今の実装では出ない（記録の後に設定が変わったため） |
| 2026-11-24 | 本体 1.2.0 を取り込み | `upstream/main`（v1.2.0）の上に載せ直した。設定に書いた空間を `scenario_resolved.json` から読む、部屋の中の人（`in_room`）を正解から除く。全部のテスト（241）と ruff が通る |
| 2026-11-25 | 本体の出力の形に合わせる | 追跡が、正解と同じ形の `aicity/`（カメラの番号も正解と同じ）・`graph/`・`tasks.json` を `bev_tracks/<追跡器>/<群>/` に書く（`bev_graph/` はやめた）。`tasks.json` はゲートの通過・止まった区間・混み具合・EV の待ち時間を推定し、`bev-eval` が正解と比べる（`twinsim/bev/tasks.py`）。場面 2 の後ろの 100 秒（A・PeopleNet ／ 正解から作った追跡）: ゲートを入った人 正解 61 に推定 62（2 秒以内で対応 60）／ 61（61）、EV の待ち時間の平均 正解 23.8 秒に 19.7 秒（73 人）／ 29.7 秒（47 人）。ゲートを使わずにレーンを通る人は正解の `gate_events` に入らないので、出る向きは多く数える（正解 2、正解から作った追跡でも 7） |
| 2026-11-25 | with-mopf との比べ | `twinsim/bev/compare.py`・`bev-mopf`（`tracks`・`import`・`compare`）。オフィスの場面で 8 台の A と with-mopf の 2 台（CamGate・CamEV）を、服の色あり・なしの描画で比べた。服の色ありでは mopf、なしでは A が良く、カメラの間の移りはどちらでも A が強い。群に `area` は書かない（A が範囲の外の検出を捨てるため）。結果・図・やり方は [報告（2026-11-25）](report/2026-11-25/README.md) |

## 1. これまでに分かっていること（外の実験の結果）

twinsim の場面（Isaac Sim で描画、PeopleNet v2.6.3 の検出は両者で同じ）で比べた。数値は評価区間の IDF1（床の上で 1 m 以内を対応とする）。

| 場面 | カメラ | 人（1 時刻の平均） | 公式 MV3DT | A |
|---|---|---|---|---|
| オフィスのロビー（19 × 19 m） | 7 台、視野が重なる | 28 人 | 0.36〜0.41 | 0.670 |
| `office_gate_ev`（在館 1,200 人）のゲートの廊下 | 5 台（G1〜G4 ＋ CamGate） | 5.7 人 | 0.639 | 0.763（揺らした 15 通りの平均。揺らさない設定で 0.760） |
| 同、EV 前の広間 | 5 台（E1〜E4 ＋ CamEV） | 27.9 人 | 0.395 | 0.635（同、0.637） |

- MV3DT は、カメラ間で ID がそろわず、同じ人に視点ごとに別の ID が付く（EV 前で ID の切り替わり 1,323 回、A は 352 回）。
- A の後付け（窓最適化）は、検出が時間的・視点間でそろっているときだけ効く（ゲートの群で +0.017、EV 前では −0.029）。

## 2. 全体の流れ

```
シナリオ（camera_groups を足す）─→ twinsim run（CPU）─→ 正解（ground_truth・mot・aicity・frames）・中間層のグラフ（graph/）
                                         │
                                         └→ twinsim render（GPU）─→ video/<カメラ>.mp4
                                                                      │
              twinsim bev-detect（GPU。または正解の枠から作る）─→ bev_dets/<検出器>/
                                                                      │
              twinsim bev-track（A は CPU、MV3DT は GPU）──────────→ bev_tracks/<追跡器>/<群>/
                                                                      │
              twinsim bev-eval ─→ bev_eval/（床の上の指標・カメラごとの指標・人数）
                               └→ bev_tracks/<追跡器>/<群>/graph/・tasks.json（正解と同じ形）─→ GNN・最適化（MOPF の mopf_graph/ と同じ使い方）
```

- 検出と追跡を分け、検出は一度ファイルに残す。同じ検出のまま追跡器だけを比べられる（A と MV3DT は同じ検出で比べる）。
- MV3DT は検出器を中に持つ。検出を外から渡す形にはせず、DeepStream の中の検出の設定を同じにそろえる（上位 100 件の NMS など）。
  検出の結果も DeepStream から書き出して `bev_dets/` に置き、A の入力にもできる（外の実験と同じやり方）。

## 3. コマンド

MOPF の `mopf-process`・`mopf-eval`・`mopf-overlay`・`mopf-link` に合わせて、頭に `bev-` を付ける。リポジトリの中では `./run.sh <コマンド>`。

| コマンド | 中身 | GPU |
|---|---|---|
| `twinsim bev-detect <結果> --detector {gt_boxes,peoplenet,external} [--group 群]` | 各カメラの映像から人の箱を出して `bev_dets/<検出器>/` に置く | `peoplenet` は使う |
| `twinsim bev-track <結果> --tracker {a,mv3dt} [--group 群] [--dets 検出器] [--preset 名前]` | 床の上の軌跡を `bev_tracks/<追跡器>/<群>/` に置く | `mv3dt` は使う |
| `twinsim bev-eval <結果> [--tracker ...]` | 正解と比べる（`bev_eval/`。床の上・カメラごと・人数・集計） | 使わない |
| `twinsim bev-overlay <結果> --tracker ... --group ...` | 上面図と各カメラを並べ、正解・予測・誤りの印を重ねた動画 | 使わない |
| `twinsim bev-link <結果>`（段階 5） | 視野の重ならない群の間（ゲート → EV 前）で、同じ人をつなぐ推論と採点 | 使わない |

- GPU を使うコマンドは、MOPF と同じく、使う前に所要時間と GPU を示して確かめる（`--confirm` がなければ見積もりだけ出す）。
- Python からは `twinsim.bev.detect`・`track`・`evaluate`・`overlay`。

## 4. 置き場所

`twinsim/mopf/` と同じ並びで、`twinsim/bev/` に置く。

```
twinsim/bev/
  __init__.py   入口の一覧（mopf/__init__.py と同じ書き方）
  calib.py      正解のカメラ（4×4 の行列と画角）→ 3×4 の投影行列（世界の m → 画素）。twinsim の投影との一致を確かめる
  groups.py     camera_groups の読み込み・評価領域・各群の床の見え方（何台から見えるか）
  detect.py     検出器の包み: gt_boxes（正解の枠に、ずれ・見落とし・誤検出を加える）、peoplenet（DeepStream）、external（読み込み）
  trackers/
    base.py     追跡器の約束（下の 6）と登録
    a.py        A（円柱で床の位置・オンライン追跡・窓最適化・閉ループ）
    mv3dt.py    公式 MV3DT の包み（設定を作る・コンテナと MQTT の受け口を起動・出力を床の上の軌跡に直す）
  evaluate.py   床の上の指標・カメラごとの指標・人数・判定しない人の決まり
  graph.py      追跡から中間層のグラフ（graph/ と同じ形）
  overlay.py    上面図と各カメラを並べた動画
  synthetic.py  確かめ用の追跡（正解から作る。誤りを決まった割合で入れられる）
  io.py         共通の読み込み（正解・検出・追跡）
```

- A は numpy と scipy だけで動く。twinsim の依存は増やさない。
- MV3DT は、DeepStream のコンテナ（`ghcr.io/nvidia/deepstream:9.2-triton-multiarch`）と、その中で使うモデルの置き場所を、使う人の環境で指定する。
  NVIDIA のファイルや重みはリポジトリに入れない。MQTT の受け口（mosquitto）も、その環境にあるものを起動する。

## 5. 設定に足すもの（シナリオの `camera_groups`）

`camera_groups` は、`cameras[]` と同じくシナリオの一番上のキーにする（`bev-*` 用の別の設定ファイルは作らない）。

```json
"camera_groups": {
  "gate": {"cameras": ["G1", "G2", "G3", "G4", "CamGate"], "area": [[-4.5, 13.0], [-1.5, 13.0], [-1.5, 21.0], [-4.5, 21.0]]},
  "ev":   {"cameras": ["E1", "E2", "E3", "E4", "CamEV"],   "area": [[-14.5, -7.0], [-9.5, -7.0], [-9.5, 2.0], [-14.5, 2.0]]}
}
```

- `area` は評価領域の多角形（世界の m）。書かなければ、群のカメラのどれかから床が見える所。
- 追跡器の設定（A のパラメータ）は場面で最適値が変わる（消失保持の長さは、静止して近い人の場面では短く、歩いて通り過ぎる場面では長く）。
  名前付きの設定（`twinsim/bev/presets/<名前>.json`）として持ち、`--preset` で選ぶ。外の実験で調整した値を、最初の設定として入れる。
- 計算の前の検査（`scenario.validate`）で確かめる：知らないキー（一番上と `camera_groups.<群>` の中）、`cameras[]` にないカメラ名、群の名前（`cameras[]` と同じく文字・数字・_・- だけ）、`area` が 3 点以上の多角形か。今の twinsim の決まりと同じく誤りにする。
- 書かなければ「群なし」。`bev-*` は `cameras[]` 全部を 1 つの群（名前 `all`）として扱う。シミュレーション（`twinsim run`）の結果は変わらない。
- 解決した値を `scenario_resolved.{yaml,json}` と `ground_truth.json`（`cameras` の隣に `camera_groups`）に残す。`bev-*` は結果のフォルダの `ground_truth.json` から群を読むので、シナリオのファイルを別に渡さなくてよい。
- `docs/config-reference.md`（日英）のキーの一覧と検査の表、`docs/formats.md`（日英）の `ground_truth.json` の説明に足す。

## 6. 追跡器の約束

```python
class Tracker:                                   # twinsim/bev/trackers/base.py
    name: str
    def __init__(self, calib, area, fps, params, floor_z=0.0): ...
    def step(self, frame, detections) -> list[TrackPoint]    # 毎コマ。その時点で出せる結果（オンライン）
    def finish(self) -> list[TrackPoint]                      # 終わりに、遅れて訂正した結果（窓最適化など）
```

- `detections`：カメラ・箱（x1, y1, x2, y2）・確信度。床の位置を直接出す検出器のための床の位置は任意。
- `TrackPoint`：コマ・ID・床の位置（x, y）・誤差（2×2）・見えたカメラの数・種類（`online` か `corrected`）・遅れのコマ数。
- 外の手法（MV3DT、のちに UMPN）は、`step` を使わず、映像やファイルをまとめて渡して外のコマンドを動かし、出力を同じ形に直す（`run_offline`）。

## 7. 出力の形（`docs/formats.md` の日英に足す）

| ファイル | 形 | 中身 |
|---|---|---|
| `calib/<カメラ>.json` | JSON | `P`（3×4）・`K`・`width`・`height`・`fps`・`eye` |
| `bev_dets/<検出器>/<カメラ>.txt` | CSV | `frame, x1, y1, x2, y2, conf`（任意で `xw, yw`）。コマは動画の 0 始まり（`render_log.json` の `frames[0]` からの数） |
| `bev_dets/<検出器>/run.json` | JSON | 検出器・設定・かかった時間・twinsim の版（git のハッシュ） |
| `bev_tracks/<追跡器>/<群>/bev.csv` | CSV | `frame, t, id, x, y, cov_xx, cov_xy, cov_yy, n_views, kind, delay` |
| `bev_tracks/<追跡器>/<群>/mot/<カメラ>.txt` | MOTChallenge | 床の位置から作ったカメラごとの枠（身長 1.7 m・半径 0.3 m の円柱を写した箱） |
| `bev_tracks/<追跡器>/<群>/aicity.txt` | AI City の MTMC | 正解の `aicity/gt.txt` と同じ形（`camera_id obj_id frame_id xmin ymin width height xworld yworld`） |
| `bev_tracks/<追跡器>/<群>/run.json` | JSON | 追跡器・設定・使った検出・1 コマあたりの時間・twinsim の版 |
| `bev_eval/<追跡器>_<群>.json`・`bev_eval/summary.csv` | JSON・CSV | 下の 8 の指標 |
| `bev_tracks/<追跡器>/<群>/graph/zone_counts.csv`・`zone_flows.csv`・`tasks.json` | CSV・JSON | `graph/`・`tasks.json` と同じ形（2026-11-25 に `bev_graph/` から移した） |

## 8. 評価の決まり

- **床の上（主）**：IDF1・HOTA・MOTA・MODA・ID の切り替わり・適合率・再現率・位置の誤差を、床の上の距離 1 m（と 0.5 m）で群ごとに出す。
  - 判定しない人（見逃しにも誤検出にも数えない。その近くの予測は捨てる）:
    - 群のどのカメラでも、見えている割合が 0.4 未満の人（MOPF の比べ方と同じしきい値）
    - 扉・EV・湧き出し点で現れる・消える瞬間の人（ベンチマークの採点ツールの `portal` と同じ考え方）
  - 評価領域の外の予測は捨てる。
- **カメラごと（副）**：`bev_tracks/.../mot/` を正解の `mot/` と比べる（IoU 0.3、見えている割合 0.4 未満は判定しない）。`mopf_eval/` の値と並べられる。
- **人数・業務の指標**：ゾーンごとの人数の誤差、ゲートの通過、EV の待ち時間を、`tasks.json`・`graph/` と比べる。
- **複数回の実行**：MV3DT は同じ設定でも実行ごとに結果がぶれるので、既定で 3 回走らせて平均と標準偏差を出す。
  A は小さな揺らし（主なパラメータを ±3% など 15 通り）で平均と差を出す（外の実験と同じ）。

## 9. 確かめ（テスト）

| 確かめ | やり方 |
|---|---|
| 投影 | `calib.py` の行列が twinsim の `Camera.project` と 1e−6 画素以内で一致する |
| 評価 | 正解から作った追跡（`synthetic.py`）で IDF1・MOTA が 1。見逃し 20% なら再現率 0.8、ID の取り違えを入れたら ID の切り替わりの数が合う（MOPF の `synthetic_tracks` と同じ確かめ） |
| A | 小さな場面（`office_smoke` に視野の重なるカメラ 3 台）で、`gt_boxes`（誤りなし）の検出から IDF1 0.95 以上。GPU なしで CI で動く |
| 外の実験との一致 | 段階 2・3 で、場面 2（`office_gate_ev`、在館 1,200 人）の値が再現する（A は同じ値、MV3DT は実行ごとのぶれの範囲） |
| 書き方 | `uv run --group dev ruff check`・`uv run --group dev pytest -q` |

## 10. 段階

| 段階 | 中身 | 終わりの確かめ |
|---|---|---|
| 1 | シナリオの `camera_groups`（検査・`ground_truth.json` に残す）・`calib`・`groups`・ファイルの形・`synthetic`・`evaluate`（床の上）・`detect gt_boxes`・`trackers/a`（オンライン）・`bev-detect`・`bev-track`・`bev-eval`・テスト | GPU なしで、小さな場面が通しで動き、テストが通る |
| 2 | `detect peoplenet`（DeepStream の中の PeopleNet で検出だけを書き出す）、群ごとの A、A の名前付きの設定 | 場面 2 の A の値（揺らさない設定でゲート 0.760、EV 前 0.637）が再現する |
| 3 | `trackers/mv3dt`（設定の生成・コンテナと MQTT・出力の変換・複数回の実行） | 場面 2 の MV3DT の値（0.639、0.395）が、実行ごとのぶれの範囲で再現する |
| 4 | A の窓最適化・閉ループ（`finish` で遅れて訂正した結果）、カメラごとの評価、人数・業務の指標、`bev-overlay`、`bev_graph/`、文書（日英）、`examples/` | 文書と例がそろい、全部のテストが通る |
| 5（任意） | `bev-link`（群の間のつなぎ）、UMPN の包み、作者の描画ファーム（`tools/render_farm.sh`）で複数の GPU に分けて描く手順、GUI・REST API | ロードマップ 1.3 の「離れた場所のカメラ間の対応の評価」 |

## 11. 守ること

- 文書は日本語と英語の 2 つ（`docs/*.md`・`docs/*.en.md`）。片方を直したら、もう片方も同じように直す。この計画も、実装に入る段階で英語版を作る。
- 出力の形を変えたら `docs/formats.md`（と英語版）と README の「出力の形」を直す。
- データ（結果・映像・検出・追跡）は git に入れない。
- GPU は共有。使う前に `nvidia-smi` で空きを確かめ、検出などと同じマシンで動かすときは同じ GPU 1 枚に載せるのが既定、分けるときも 2 枚まで（`CLAUDE.md`）。描画など長い計算は所要時間と GPU を示して確認を取ってから始める。
- 特定の企業や建物の名前は、文書とコードに書かない（`CLAUDE.md`）。
- フォークへの送信は段階ごとに確認を取ってから。作者のリポジトリへの提案（プルリクエスト）は、フォークの持ち主が決める。
- 作者は `main` を更新し続けているので、段階ごとに `upstream/main` を取り込む。
- 結果には、使った twinsim の版（git のハッシュ）を必ず残す（外の実験で、場面ごとにシミュレータの版が違っていたため）。

## 12. まだ決めていないこと

- 判定しない人の「見えている割合」に、近似の枠（`mot/`）を使うか、描画から取る正確な枠（`render(..., labels=True)` の `labels/`）を使うか。後者は描画が 2〜3 倍遅くなる。
- MV3DT の検出の設定を、既定（上位 20 件）と A と同じ上位 100 件のどちらにそろえるか（外の実験では、場面で良いほうが変わった）。
