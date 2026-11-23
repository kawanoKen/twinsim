"""確かめ用: 正解から床の上の追跡（bev_tracks/<label>/<群>/bev.csv）を作る（揺れ・見逃し・ID の取り違えを足せる）。
評価のテストに使う（mopf/synthetic.py と同じ考え方）。"""

import json

import numpy as np

from .groups import groups as load_groups
from .io import gt_positions, open_run, run_space, version, write_bev
from .trackers.base import TrackPoint


def synthetic_tracks(run_dir, group_names=None, label="synthetic", jitter_m=0.0, miss=0.0, switch_every_s=0.0, seed=0):
    """評価領域の中の正解の人を、そのまま（または崩して）追跡の形で書く。
    jitter_m: 位置のずれ（標準偏差 m）、miss: 1 点ずつ落とす割合、switch_every_s: この秒ごとに全員の ID を新しくする（ID の切り替わり）。"""
    run, gt, f0, n_frames = open_run(run_dir)
    rng = np.random.default_rng(seed)
    pos = gt_positions(gt, f0, n_frames)
    num = {n: i for i, n in enumerate(sorted(gt["people"]), start=1)}
    every = int(round(switch_every_s * gt["fps"]))
    out = {}
    for gname, group in load_groups(gt, group_names, run_space(run, gt)).items():
        pts = []
        for f in range(n_frames):
            items = pos.get(f, {})
            names = list(items)
            if not names:
                continue
            inside = group.contains([items[n] for n in names])
            for n, ok in zip(names, inside):
                if not ok or (miss and rng.random() < miss):
                    continue
                x, y = items[n]
                if jitter_m:
                    x, y = x + rng.normal(0, jitter_m), y + rng.normal(0, jitter_m)
                tid = num[n] + (100000 * (f // every) if every else 0)
                pts.append(TrackPoint(f, tid, float(x), float(y), None, 1))
        folder = run / "bev_tracks" / label / gname
        write_bev(folder / "bev.csv", pts, gt["fps"], f0)
        info = {"tracker": "synthetic", "group": gname, "synthetic": {"jitter_m": jitter_m, "miss": miss, "switch_every_s": switch_every_s, "seed": seed},
                "frame_offset": f0, "frames": n_frames, "points": len(pts), "version": version()}
        (folder / "run.json").write_text(json.dumps(info, ensure_ascii=False, indent=1))
        out[gname] = folder
    return out
