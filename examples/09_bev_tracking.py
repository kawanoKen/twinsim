"""床の上の複数カメラの追跡を作って採点する（CPU だけ、30 秒ほど。描画も GPU も要らない）。

視野の重なるカメラの群（camera_groups）を設定に書いて計算し、正解の枠から作った検出（見落とし・ずれ・誤検出を足す）で
追跡器 A を動かし、床の上の指標（IDF1・HOTA・MOTA）、カメラごとの指標、ゾーンの人数の誤差を出す。
描画した動画があれば、検出を --detector peoplenet、追跡器を mv3dt にでき、bev-overlay で確かめの動画も作れる（docs/bev-tracking.md）。

実行（リポジトリの最上段のフォルダで）:
    uv run python examples/09_bev_tracking.py

結果は runs/tutorial_bev/ に書く（bev_dets/・bev_tracks/・bev_eval/）。追跡は正解と同じ形の mot/・aicity/・graph/・tasks.json も書く。
"""

from pathlib import Path

import twinsim
from twinsim.bev import detect, evaluate, track

config = {
    "name": "tutorial_bev", "space": "office", "seed": 1, "duration_s": 60, "fps": 10,
    "streams": [{"name": "arrivals", "origin": "entrance", "rate": {"type": "constant", "per_min": 20},
                 "destinations": [{"to": "south_exit", "via": ["gate"]}]}],
    # ゲートの廊下を前後から見る 4 台と、ロビーから見る 1 台（視野が重なる）
    "cameras": [{"name": "G1", "eye": [-4.3, 22.5, 2.8], "target": [-3.0, 15.5, 0.8], "focal_mm": 8.0},
                {"name": "G2", "eye": [-1.7, 22.5, 2.8], "target": [-3.0, 15.5, 0.8], "focal_mm": 8.0},
                {"name": "G3", "eye": [-4.3, 13.0, 2.8], "target": [-3.0, 17.5, 0.8], "focal_mm": 8.0},
                {"name": "G4", "eye": [-1.7, 13.0, 2.8], "target": [-3.0, 17.5, 0.8], "focal_mm": 8.0},
                {"name": "CamGate"}],
    # 群: 一緒に追跡するカメラと、評価する床の範囲（世界の m の多角形）
    "camera_groups": {"gate": {"cameras": ["G1", "G2", "G3", "G4", "CamGate"],
                               "area": [[-4.5, 13.0], [-1.5, 13.0], [-1.5, 21.0], [-4.5, 21.0]]}},
    "outputs": ["ground_truth", "frames", "tasks", "graph", "mot", "aicity"],
}

run = Path("runs/tutorial_bev")
twinsim.simulate(config, out_dir=run)

# 検出: 正解の枠に、見落とし 10%・ずれ 3 画素・1 コマに 0.3 個の誤検出を足す
detect(run, "gt_boxes", name="noisy", miss=0.1, jitter_px=3.0, fp_per_frame=0.3)

# 追跡: A（設定は presets/default.json。--param と同じく params で上書きできる）
track(run, "a", dets="noisy", params={"max_age_s": 1.5}, label="a_noisy")

# 採点: 床の上で 1 m（と 0.5 m）以内を同じ人とみなす
res = evaluate(run, "a_noisy")["gate"]
m = res["metrics"]["1m"]
print(f"床の上   IDF1 {m['IDF1']:.3f}  HOTA {m['HOTA']:.3f}  MOTA {m['MOTA']:.3f}  ID の切り替わり {m['IDSW']}")
for cam, c in res["per_camera"].items():
    print(f"{cam:8s} IDF1 {c['IDF1']:.3f}  MOTA {c['MOTA']:.3f}  人数の誤差 {c['count_mae']:.2f}")
print("評価領域の人数の誤差（秒ごとの平均）:", res["graph"]["area_count"]["mae"])
g = res["tasks"]["gates"]["gate:in"]
print(f"ゲートを入った人: 正解 {g['truth']}、推定 {g['pred']}（2 秒以内で対応 {g['matched']}）")
print("結果:", run / "bev_eval", run / "bev_tracks/a_noisy/gate")
