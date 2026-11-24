"""追跡器（base.py の約束）と、群ごとに検出から床の上の軌跡を作る入口 track。"""

import json
import time
from pathlib import Path

import numpy as np

from ..calib import calib as make_calib
from ..groups import groups as load_groups
from ..io import load_dets, open_run, run_space, version, write_bev
from .a import PERSON_H, PERSON_R, cylinder_box
from .base import TRACKERS, Tracker, TrackPoint, get, register

PRESETS = Path(__file__).resolve().parent.parent / "presets"


def presets():
    return sorted(p.stem for p in PRESETS.glob("*.json"))


def load_preset(name, tracker):
    """presets/<名前>.json の params（その追跡器のものだけ）。"""
    p = PRESETS / f"{name}.json"
    if not p.exists():
        raise ValueError(f"設定 {name} はない（あるのは {', '.join(presets())}）")
    d = json.loads(p.read_text())
    if d.get("tracker") != tracker:
        raise ValueError(f"設定 {name} は追跡器 {d.get('tracker')} のもの（今は {tracker}）")
    return d["params"]


def track(run_dir, tracker="a", group_names=None, dets="gt_boxes", preset=None, params=None, label=None, repeats=1, **offline_opts):
    """群ごとに bev_tracks/<label>/<群>/ を作る（label の既定は追跡器の名前）。設定は 追跡器の既定 → presets/<preset>.json
    （既定は追跡器ごと: A は default、MV3DT は mv3dt）→ params の順に重ねる。
    A は bev_dets/<dets>/ の検出から。MV3DT（offline）は動画から中で検出して動かす（offline_opts: gpu・image・models・keep_raw）。
    repeats が 2 以上なら <label>_r1, <label>_r2, … に分けて書く（実行ごとにぶれる追跡器を比べるため）。"""
    run, gt, f0, n_frames = open_run(run_dir)
    cls = get(tracker)
    preset = preset or cls.default_preset
    p = dict(load_preset(preset, tracker) if preset else {}, **(params or {}))
    if offline_opts and not cls.offline:
        raise ValueError(f"追跡器 {tracker} には {', '.join(offline_opts)} を使わない")
    space = run_space(run, gt)
    floor_z = space.z
    label = label or tracker
    labels = [label] if repeats <= 1 else [f"{label}_r{k}" for k in range(1, repeats + 1)]
    out = {}
    for gname, group in load_groups(gt, group_names, space).items():
        cal = make_calib(gt, group.cameras)
        D = None if cls.offline else load_dets(run / "bev_dets" / dets, group.cameras)
        for lab in labels:
            folder = run / "bev_tracks" / lab / gname
            trk = cls(cal, group, gt["fps"], p, floor_z=floor_z)
            t0 = time.perf_counter()
            extra = {}
            if cls.offline:
                pts, obs, extra = trk.run_offline(run, gt, n_frames, folder / f"_{tracker}", **offline_opts)
            else:
                pts = []
                for f in range(n_frames):
                    pts += trk.step(f, D.get(f, []))
                pts += trk.finish()
            dt = time.perf_counter() - t0
            write_bev(folder / "bev.csv", pts, gt["fps"], f0)
            if cls.offline:
                write_native_views(folder, obs, cal, f0, n_frames)
            else:
                write_views(folder, pts, cal, gt, f0, floor_z)
            info = {"tracker": tracker, "group": gname, "group_def": group.to_json(),
                    "dets": "internal" if cls.offline else dets, "preset": preset,
                    "params": p, "frame_offset": f0, "frames": n_frames, "points": len(pts), "ids": len({q.id for q in pts}),
                    "ms_per_frame": round(1000 * dt / max(n_frames, 1), 3), "version": version(), **({"run": extra} if extra else {})}
            (folder / "run.json").write_text(json.dumps(info, ensure_ascii=False, indent=1))
            out[gname if repeats <= 1 else f"{gname}/{lab}"] = info
    return out


def write_native_views(folder, obs, cal, f0, n_frames):
    """追跡器が出したカメラごとの枠（MV3DT）: mot/<カメラ>.txt と aicity.txt（write_views と同じ形）。"""
    (folder / "mot").mkdir(parents=True, exist_ok=True)
    rows, lines = {c: [] for c in cal}, []
    idx = {c: i for i, c in enumerate(cal, start=1)}
    for f in sorted(obs):
        if f >= n_frames:
            continue
        for cam, i, (x0, y0, x1, y1), (x, y), conf in obs[f]:
            rows[cam].append(f"{f + f0 + 1},{i},{x0:.1f},{y0:.1f},{x1 - x0:.1f},{y1 - y0:.1f},{conf:.4f},-1,-1\n")
            if np.isfinite([x, y]).all():
                lines.append(f"{idx[cam]} {i} {f + f0 + 1} {x0:.1f} {y0:.1f} {x1 - x0:.1f} {y1 - y0:.1f} {x:.3f} {y:.3f}\n")
    for cam, r in rows.items():
        (folder / "mot" / f"{cam}.txt").write_text("".join(r))
    (folder / "aicity.txt").write_text("".join(lines))
    (folder / "aicity_cameras.json").write_text(json.dumps(idx))


def write_views(folder, pts, cal, gt, f0, floor_z, kind="online"):
    """床の位置からカメラごとの枠（背 1.7 m・半径 0.3 m の円柱を写した箱）: mot/<カメラ>.txt と aicity.txt（正解の aicity/gt.txt と同じ形）。"""
    (folder / "mot").mkdir(parents=True, exist_ok=True)
    pts = [q for q in pts if q.kind == kind]
    xy = np.array([[q.x, q.y] for q in pts], float).reshape(-1, 2)
    lines = []
    for ci, (cam, c) in enumerate(cal.items(), start=1):
        P = np.asarray(c["P"], float)
        rows = []
        if len(xy):
            # 床の点がカメラの前にあるものだけ（後ろの点の投影は意味がない）
            depth = np.c_[xy, np.full(len(xy), floor_z), np.ones(len(xy))] @ P[2]
            box = cylinder_box(P, xy, floor_z)
            for q, b, z in zip(pts, box, depth):
                if z <= 0.1:
                    continue
                x0, y0, x1, y1 = max(b[0], 0), max(b[1], 0), min(b[2], c["width"]), min(b[3], c["height"])
                if x1 - x0 < 2 or y1 - y0 < 2:
                    continue
                rows.append(f"{q.frame + f0 + 1},{q.id},{x0:.1f},{y0:.1f},{x1 - x0:.1f},{y1 - y0:.1f},1,-1,-1\n")
                lines.append(f"{ci} {q.id} {q.frame + f0 + 1} {x0:.1f} {y0:.1f} {x1 - x0:.1f} {y1 - y0:.1f} {q.x:.3f} {q.y:.3f}\n")
        (folder / "mot" / f"{cam}.txt").write_text("".join(rows))
    (folder / "aicity.txt").write_text("".join(lines))
    (folder / "aicity_cameras.json").write_text(json.dumps({c: i for i, c in enumerate(cal, start=1)}))


__all__ = ["PERSON_H", "PERSON_R", "TRACKERS", "Tracker", "TrackPoint", "get", "load_preset", "presets", "register", "track"]
