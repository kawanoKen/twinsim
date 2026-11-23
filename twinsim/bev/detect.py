"""検出: 各カメラの人の枠を bev_dets/<検出器>/<カメラ>.txt に置く（形は io.DET_COLUMNS）。

  gt_boxes  正解の枠（mot/<カメラ>.txt。人を半径 0.25 m の箱とみなした近似）に、ずれ・見落とし・誤検出を加える。描画も GPU も要らない
  external  外で作った検出を読み込む（同じ形の CSV を <検出器>/<カメラ>.txt に置いたフォルダ）
（PeopleNet（DeepStream）は段階 2 で足す。）
"""

import json
import shutil
import time
from pathlib import Path

import numpy as np

from .groups import groups as load_groups
from .io import load_dets, open_run, run_space, version, write_dets

DETECTORS = ("gt_boxes", "external")


def gt_boxes(run, gt, f0, n_frames, cam, vis_min=0.4, jitter_px=0.0, miss=0.0, fp_per_frame=0.0, rng=None):
    """正解の枠 → 検出（画面の中に切った枠、確信度 1）。見えている割合が vis_min 未満の人は出さない。"""
    rng = rng or np.random.default_rng(0)
    c = gt["cameras"][cam]
    W, H = c["width"], c["height"]
    rows = []
    path = run / "mot" / f"{cam}.txt"
    if not path.exists():
        raise FileNotFoundError(f"正解の枠がない: {path}（シナリオの outputs に mot を入れて計算し直す）")
    for line in path.read_text().splitlines():
        q = line.split(",")
        f = int(q[0]) - 1 - f0
        if not 0 <= f < n_frames or float(q[8]) < vis_min or (miss and rng.random() < miss):
            continue
        x, y, w, h = map(float, q[2:6])
        b = np.array([x, y, x + w, y + h]) + (rng.normal(0, jitter_px, 4) if jitter_px else 0)
        b = np.clip(b, [0, 0, 0, 0], [W, H, W, H])
        if b[2] - b[0] > 2 and b[3] - b[1] > 2:
            rows.append((f, *b, 1.0))
    if fp_per_frame:  # 誤検出: 画面のどこかに、人くらいの大きさの枠
        for f in range(n_frames):
            for _ in range(rng.poisson(fp_per_frame)):
                h = rng.uniform(0.15, 0.5) * H
                x0, y0 = rng.uniform(0, W - h * 0.4), rng.uniform(0, H - h)
                rows.append((f, x0, y0, x0 + h * 0.4, y0 + h, float(rng.uniform(0.3, 1.0))))
    rows.sort(key=lambda r: r[0])
    return rows


def detect(run_dir, detector="gt_boxes", group_names=None, cameras=None, name=None, source=None, seed=0, **opts):
    """検出を bev_dets/<name>/ に書く（name の既定は検出器の名前）。cameras を省くと、群のカメラ全部。
    gt_boxes の opts: vis_min・jitter_px・miss・fp_per_frame。external: source（同じ形の CSV のフォルダ）。"""
    if detector not in DETECTORS:
        raise ValueError(f"検出器 {detector} はない（あるのは {', '.join(DETECTORS)}）")
    run, gt, f0, n_frames = open_run(run_dir)
    if cameras is None:
        cameras = sorted({c for g in load_groups(gt, group_names, run_space(run, gt)).values() for c in g.cameras})
    out = run / "bev_dets" / (name or detector)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    rng = np.random.default_rng(seed)
    n = 0
    for cam in cameras:
        if detector == "gt_boxes":
            rows = gt_boxes(run, gt, f0, n_frames, cam, rng=rng, **opts)
            write_dets(out / f"{cam}.txt", rows)
            n += len(rows)
        else:
            src = Path(source or "").expanduser() / f"{cam}.txt"
            if not src.exists():
                raise FileNotFoundError(f"外の検出がない: {src}")
            if src.resolve() != (out / f"{cam}.txt").resolve():
                shutil.copyfile(src, out / f"{cam}.txt")
            n += sum(len(v) for v in load_dets(out, [cam]).values())
    info = {"detector": detector, "cameras": cameras, "options": {**opts, "seed": seed, **({"source": str(source)} if source else {})},
            "frame_offset": f0, "frames": n_frames, "boxes": n, "elapsed_s": round(time.time() - t0, 2), "version": version()}
    (out / "run.json").write_text(json.dumps(info, ensure_ascii=False, indent=1))
    return out, info
