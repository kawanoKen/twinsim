# 床の上の複数カメラの追跡（`twinsim bev-*`）

[English](bev-tracking.en.md)

視野の重なるカメラの群ごとに、人を床の上（真上から見た図）で追跡し、シミュレータの正解と比べます。
検出と追跡を分けてあるので、同じ検出のまま追跡器だけを比べられます。計画と段階は [bev-tracking-plan.md](bev-tracking-plan.md)。

## 手順

1. シナリオに `camera_groups` を書きます（[設定の説明](config-reference.md#camera-groups)）。書かなければ、全部のカメラを 1 つの群 `all` とします。
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
   twinsim bev-detect runs/test --detector external --name mydet --source 検出のフォルダ     # 外で作った検出（同じ形の CSV）
   ```
4. 追跡（`bev_tracks/<名前>/<群>/`）。
   ```bash
   twinsim bev-track runs/test --dets noisy --preset default --param max_age_s=1.5 --label a_noisy
   ```
5. 評価（`bev_eval/`）。
   ```bash
   twinsim bev-eval runs/test --tracker a_noisy                    # 全部のコマ
   twinsim bev-eval runs/test --tracker a_noisy --frames 2000 3000 # 後ろの 100 秒（10 fps）
   ```

Python からは `twinsim.bev` の `detect`・`track`・`evaluate`・`synthetic_tracks`・`calib`・`groups` を呼びます。ファイルの形は [formats.md の 12](formats.md#bev)。

## 追跡器

| 名前 | 中身 | 必要なもの |
|---|---|---|
| `a` | 集中型のオンライン追跡。枠を人の円柱（背 1.7 m・半径 0.3 m）に合わせて床の位置を出し、カメラごとのハンガリアン法で軌跡に割り当て、カメラ間で融合したカルマンフィルタで追う。2 台以上で見えたときだけ新しい人にする | numpy・scipy（CPU、1 コマ数 ms） |

設定は `twinsim/bev/presets/<名前>.json`（`--preset`）。時間の設定は秒で書き、動画の fps でコマに直します。

| 設定 | 調整した場面 |
|---|---|
| `default` | ロビー（7 台、止まって話す人が多い） |
| `office_gate_ev-gate` | `office_gate_ev`（在館 1,200 人）のゲートの群（5 台、歩いて通り過ぎる） |
| `office_gate_ev-ev` | 同じ場面の EV 前の群（5 台、待って乗る） |

## 評価の決まり

- 床の上の距離 1 m と 0.5 m 以内を同じ人とみなし、IDF1・HOTA・MOTA・MODA・ID の切り替わり・適合率・再現率を群ごとに出します（CLEAR MOT と IDF1 は motmetrics、HOTA は TrackEval と同じ数え方）。
- 評価領域の外の正解と予測は数えません。
- 判定しない人（見逃しにも誤検出にも数えず、その近くの対応しなかった予測は捨てる）:
  - 群のどのカメラでも、見えている割合（`mot/` の近似）が 0.4 未満の人（`--vis-min`、MOPF の比べ方と同じしきい値）
  - 評価領域の中で現れた・消えた人の、前後 1 秒（`--portal-s`。EV の扉・出入口）

## 確かめ

- `tests/test_bev.py`: 投影が twinsim のカメラと 1e−6 画素以内で一致、正解から作った追跡で IDF1・MOTA が 1、見落とし 20% で再現率 0.8、ID の取り違えの数が合う、小さな場面で A が正解の枠から IDF1 0.95 以上。
- 外の実験との一致（2026-11-16）: `office_gate_ev`（在館 1,200 人、10 fps、300 秒、群ごとに 5 台）の描画に PeopleNet をかけた検出で、A の結果が外の実験の実装と全部のコマで同じ ID・位置の差 0.0005 m 以下（出力の丸め）になりました。
  後ろの 100 秒の IDF1（1 m、判定しない人なし）はゲートの群 0.761、EV 前の群 0.637、上の決まりで判定しない人を除くと 0.768・0.699 です。
