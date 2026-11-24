# 床の上の複数カメラの追跡（`twinsim bev-*`）

[English](bev-tracking.en.md)

視野の重なるカメラの群ごとに、人を床の上（真上から見た図）で追跡し、シミュレータの正解と比べます。
検出と追跡を分けてあるので、同じ検出のまま追跡器だけを比べられます。計画と段階は [bev-tracking-plan.md](bev-tracking-plan.md)、本体と with-mopf との違いは [bev-design.md](bev-design.md)。

## 手順

1. シナリオに `camera_groups` を書きます（[設定の説明](config-reference.md#camera-groups)）。書かなければ、全部のカメラを 1 つの群 `all` とします。`area` を書かなければ、評価領域は群のカメラのどれかから床が見える歩ける所です。
   ```json
   "camera_groups": {"gate": {"cameras": ["G1", "G2", "G3", "G4", "CamGate"], "area": [[-4.5, 13.0], [-1.5, 13.0], [-1.5, 21.0], [-4.5, 21.0]]}}
   ```
2. 計算します（`outputs` に `mot` を入れる。正解の枠と、見えている割合に使う）。描画は要りません（描画したなら、コマは動画に合わせます）。
   ```bash
   twinsim run config.json --out runs/test
   ```
3. 検出（`bev_dets/<名前>/`）。
   ```bash
   twinsim bev-detect runs/test                                   # 正解の枠から（GPU なし）
   twinsim bev-detect runs/test --name noisy --miss 0.1 --jitter-px 3 --fp-per-frame 0.5   # 見落とし・ずれ・誤検出を足す
   twinsim bev-detect runs/test --detector peoplenet --gpu 3   # 描画した動画に PeopleNet（DeepStream のコンテナ、GPU。群ごとに 1 回）
   twinsim bev-detect runs/test --detector external --name mydet --source 検出のフォルダ     # 外で作った検出（同じ形の CSV）
   ```
4. 追跡（`bev_tracks/<名前>/<群>/`。本体と同じ形の `mot/`・`aicity/`・`graph/`・`tasks.json` も書く）。
   ```bash
   twinsim bev-track runs/test --dets noisy --preset default --param max_age_s=1.5 --label a_noisy
   ```
5. 評価（`bev_eval/`）。
   ```bash
   twinsim bev-eval runs/test --tracker a_noisy                    # 全部のコマ
   twinsim bev-eval runs/test --tracker a_noisy --frames 2000 3000 # 後ろの 100 秒（10 fps）
   ```

Python からは `twinsim.bev` の `detect`・`track`・`evaluate`・`overlay`・`synthetic_tracks`・`calib`・`groups` を呼びます（例: [examples/09_bev_tracking.py](../examples/09_bev_tracking.py)）。ファイルの形は [formats.md の 12](formats.md#bev)。

## PeopleNet の検出（`--detector peoplenet`）

描画した動画（`video/<カメラ>.mp4`）を、群ごとに 1 回 DeepStream の `deepstream-app` に通し、検出器の出力だけを書き出します（追跡器は動かさない）。
設定は外の実験と同じです: PeopleNet v2.6.3（ResNet-34、FP16）、人だけ、NMS（上位 100 件、IoU 0.5）、入力は動画の画素の大きさのまま。

- 要るもの: docker と NVIDIA のコンテナの実行環境、DeepStream のコンテナ（`TWINSIM_DEEPSTREAM_IMAGE`、既定 `ghcr.io/nvidia/deepstream:9.2-triton-multiarch`）、
  モデルの置き場所（`TWINSIM_DEEPSTREAM_MODELS` か `--models`。`PeopleNet2.6.3/resnet34_peoplenet.onnx` と `detector_labels.txt` がある所。TensorRT のエンジンを初めに作って同じ所に置くので、書ける場所）。
- GPU は `--gpu` で 1 枚を決めます（既定はない。`nvidia-smi` で空きを見てから）。在館 1,200 人・10 fps・300 秒・10 台（5 台 × 2 群）で 40 秒ほど。
- 全部の確信度の枠を書きます。しきい値は追跡器の設定（A の `det_conf`）で決めます。

## 追跡器

| 名前 | 中身 | 必要なもの |
|---|---|---|
| `a` | 集中型のオンライン追跡。枠を人の円柱（背 1.7 m・半径 0.3 m）に合わせて床の位置を出し、カメラごとのハンガリアン法で軌跡に割り当て、カメラ間で融合したカルマンフィルタで追う。2 台以上で見えたときだけ新しい人にする | numpy・scipy（CPU、1 コマ数 ms） |
| `mv3dt` | 公式 MV3DT（NVIDIA DeepStream 9.2 の多視点 3D 追跡）。カメラごとに追跡し、MQTT で ID と床の位置をカメラ間でそろえる分散型。検出（PeopleNet）は中で行い、`bev_dets/` を使わない。床の位置は、コマと ID ごとにカメラの足の位置を平均する | GPU、DeepStream のコンテナ、mosquitto（5 台・3000 コマで 1〜2 分） |

設定は `twinsim/bev/presets/<名前>.json`（`--preset`）。時間の設定は秒で書き、動画の fps でコマに直します。

| 設定 | 調整した場面 |
|---|---|
| `default` | ロビー（7 台、止まって話す人が多い） |
| `office_gate_ev-gate` | `office_gate_ev`（在館 1,200 人）のゲートの群（5 台、歩いて通り過ぎる） |
| `office_gate_ev-ev` | 同じ場面の EV 前の群（5 台、待って乗る） |
| `mv3dt`（MV3DT の既定） | 公式の見本の設定そのまま（`presets/mv3dt_tracker.yml`）。検出は DeepStream の既定 |
| `mv3dt_nms100` | MV3DT の検出を A と同じ設定（NMS 上位 100 件・IoU 0.5）にそろえたもの |

### A の窓最適化（`be_mode`）

A は、毎コマの追跡（前の段）のあとに、直近の窓の観測をつなぎ直す後ろの段を足せます（外の実験と同じ計算）。
窓の小片を、窓の始まりまでに決めた ID と最小費用の経路被覆でつなぎます。費用は位置（速度で補正した変位）と見え方（胴の色）です。

```bash
twinsim bev-track runs/test --dets peoplenet --preset office_gate_ev-gate --label a_closed \
  --param be_mode=closed --param be_window_s=4.5 --param be_every_s=1.5 --param be_link_gain=0.01 --param be_merge=false --param be_min_len_s=0
twinsim bev-eval runs/test --tracker a_closed --kind online       # 毎コマの出力
twinsim bev-eval runs/test --tracker a_closed --kind corrected    # 遅れて訂正した出力
```

- `be_mode`: `none`（既定）・`post`（後処理。訂正した出力だけが変わる）・`closed`（閉ループ。訂正した軌跡で前の段を置き換えて続ける。毎コマの出力も変わる）。
- `be_window_s`（窓の長さ）・`be_every_s`（何秒ごとに解くか）。ほかの `be_*` は `twinsim/bev/trackers/backend.py` の `BE_DEFAULTS`。上の例の `link_gain` などは、外の実験の「保守」の設定。
- 見え方（胴の HSV の色のヒストグラム）は、初めの 1 回に描画した動画から作り、`bev_dets/<検出器>/app_<カメラ>.npy` に置きます。OpenCV が要ります（任意の依存: `uv sync --extra bev`）。
  使わないなら `--param be_w_app=0`（位置だけ。OpenCV も動画も要らない）。
- 場面 2（後ろの 100 秒、IDF1）: ゲートの群は後処理の訂正した出力 0.783（A 0.761）、閉ループ 0.774・訂正 0.789。EV 前の群は後処理 0.586（A 0.637）、閉ループ 0.579・訂正 0.596。
  密集した EV 前では悪くなります（外の実験と同じ傾向）。値は外の実験の今の実装と同じです（見え方のヒストグラムも全部の枠で同じ値）。

### MV3DT（`--tracker mv3dt`）

```bash
twinsim bev-track runs/test --tracker mv3dt --gpu 3 --repeats 3          # bev_tracks/mv3dt_r1, _r2, _r3
twinsim bev-eval runs/test --tracker mv3dt --frames 2000 3000             # 3 回の平均と標準偏差
twinsim bev-track runs/test --tracker mv3dt --gpu 3 --param tracker.MultiViewAssociator.minPeerTrackletMatchScore=0.3 --param pgie.topk=100
```

- 群ごとに、動画を `deepstream-test5-app`（PeopleNet → MV3DT）に通します。実行のたびに、その計算だけの MQTT の受け口（mosquitto、127.0.0.1 の空いた番号）を立てて、終わりに止めます。
  mosquitto は `TWINSIM_MOSQUITTO`（実行ファイル。既定は PATH の `mosquitto`）、すでに動いている受け口を使うなら `TWINSIM_MQTT`（`host:port`）。
  モデルの置き場所には PeopleNet のほかに `BodyPose3DNet/bodypose3dnet_accuracy.onnx` も要ります。
- 設定は `pgie`（検出の上書き）と `tracker`（`presets/mv3dt_tracker.yml` の上書き。`"節.キー": 値`、または節を丸ごと `{"節": {キー: 値}}`）。`--param pgie.<キー>=値`・`--param tracker.<節>.<キー>=値` でも上書きできます。
- MV3DT は同じ設定でも実行ごとに結果がぶれます（カメラ間で ID がそろう割合が変わる）。`--repeats` で何回か動かし、`bev-eval` で平均と標準偏差を見ます。
- カメラごとの枠（`mot/<カメラ>.txt`）は MV3DT が出した枠そのままです（A は床の位置から作った円柱の枠）。

## 評価の決まり

- 床の上の距離 1 m と 0.5 m 以内を同じ人とみなし、IDF1・HOTA・MOTA・MODA・ID の切り替わり・適合率・再現率を群ごとに出します（CLEAR MOT と IDF1 は motmetrics、HOTA は TrackEval と同じ数え方）。
- 評価領域の外の正解と予測は数えません。部屋の中にいる人（状態 `in_room`。映らない）も正解に入れません。
- 判定しない人（見逃しにも誤検出にも数えず、その近くの対応しなかった予測は捨てる）:
  - 群のどのカメラでも、見えている割合（`mot/` の近似）が 0.4 未満の人（`--vis-min`、MOPF の比べ方と同じしきい値）
  - 評価領域の中で現れた・消えた人の、前後 1 秒（`--portal-s`。EV の扉・出入口）
- カメラごと（副。`bev_eval/per_camera.csv`）: 追跡のカメラごとの枠（`mot/<カメラ>.txt`）を正解の `mot/` と比べます（IoU 0.3 以上を対応、MOPF の `mopf-eval` と同じ決まり）。
  見えている割合 0.4 未満の人と、床の位置が評価領域の外の人は判定しません。A の枠は床の位置から作った円柱の枠なので、検出の枠より粗くなります。
- 人数（`bev_tracks/<追跡器>/<群>/graph/`）: ゾーンの人数（秒ごと）とゾーン間の移動（1 分ごと）を、シミュレータの `graph/` と同じ形で書き、正解と比べます
  （床の 6 割以上が評価領域に入るゾーンだけ。`bev_eval/<名前>_<群>.json` の `graph`）。
- 集計（`bev_tracks/<追跡器>/<群>/tasks.json`）: 正解の `tasks.json` と同じ形で、ゲートの通過・止まった区間・混み具合・EV の待ち時間を追跡から推定し、
  正解と比べます（`bev_eval/<名前>_<群>.json` の `tasks`。推定のしかたは [formats.md](formats.md#bev)）。
  推定のしかたそのものの誤差もあるので、正解から作った追跡（`twinsim.bev.synthetic_tracks`）の値と並べて見ます。
- 設定の小さな違いでの変わり方（A）: `bev-track --perturb` で主な設定を ±3%（と消すまでの時間を ±1 コマ）揺らした 15 通りを `<名前>_p0`〜`_p14` に作り、`bev-eval --tracker <名前>` で平均と標準偏差を出します（外の実験と同じ揺らし）。

## 確かめの動画（`bev-overlay`）

```bash
twinsim bev-overlay runs/test --tracker a_noisy --group gate --frames 2000 2300   # bev_overlay/a_noisy_gate.mp4
```

左に上面図（白い丸 = 正解、色の点と番号 = 予測、赤い丸 = 見逃し、赤い × = 誤検出、黄色の輪 = ID の切り替わり、灰色 = 判定しない人）、
右に群のカメラ（緑の枠 = 正解、灰色 = ほとんど隠れた正解、色の枠 = 追跡の枠）を並べます。描画した動画（`video/`）と ffmpeg が要ります。

## with-mopf と比べる（`bev-mopf`）

ブランチ with-mopf の MOPF の結果（`mopf_tracks/`・`mopf_eval/links.json`）を読み込み、同じカメラの画面の上で、with-mopf と同じ決まりで比べます。
評価領域の外の人と、見えている割合 0.4 未満の人は判定しません。決まりと手順は [bev-design.md の 5.4](bev-design.md)。

```bash
twinsim bev-mopf import runs/test --group mopf2                                         # bev_tracks/mopf/mopf2/
twinsim bev-mopf compare runs/test --tracker A=a/s7 --tracker mopf=mopf/mopf2 --area-group s7   # bev_eval/compare_CamGate_CamEV.json
```

## 確かめ

- `tests/test_bev.py`: 投影が twinsim のカメラと 1e−6 画素以内で一致、正解から作った追跡で IDF1・MOTA が 1、見落とし 20% で再現率 0.8、ID の取り違えの数が合う、小さな場面で A が正解の枠から IDF1 0.95 以上。
- 外の実験との一致（2026-11-16、11-24）: `office_gate_ev`（在館 1,200 人、10 fps、300 秒、群ごとに 5 台）の描画で、`--detector peoplenet` の検出が外の実験の検出と全部の枠で同じ値になりました。その検出で、A の結果が外の実験の実装と全部のコマで同じ ID・位置の差 0.0005 m 以下（出力の丸め）になりました。
  後ろの 100 秒の IDF1（1 m、判定しない人なし）はゲートの群 0.761、EV 前の群 0.637、上の決まりで判定しない人を除くと 0.768・0.699 です。
- A の 15 通りの揺らし（2026-11-24）: ゲートの群 0.764 ± 0.006、EV 前の群 0.635 ± 0.011（外の実験の平均 0.763・0.635。どの設定も差は 0.0002 以下で、`bev.csv` の位置を mm に丸めるため）。
- MV3DT（2026-11-24）: 同じ場面で 3 回の平均と標準偏差が、ゲートの群 0.640 ± 0.021、EV 前の群 0.390 ± 0.011（1 m、判定しない人なし。外の実験は 0.639・0.395）。決まりありで 0.643 ± 0.020・0.406 ± 0.011。
