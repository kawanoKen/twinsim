"""追跡器（base.py の約束）と、群ごとに検出から床の上の軌跡を作る入口 track。"""

import json
import time
from pathlib import Path

import numpy as np

from ...space import load_space
from ..calib import calib as make_calib
from ..groups import groups as load_groups
from ..io import load_dets, open_run, version, write_bev
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


def track(run_dir, tracker="a", group_names=None, dets="gt_boxes", preset=None, params=None, label=None):
    """bev_dets/<dets>/ の検出から、群ごとに bev_tracks/<label>/<群>/ を作る（label の既定は追跡器の名前）。
    設定は 追跡器の既定 → presets/<preset>.json（既定 default、A だけ）→ params の順に重ねる。"""
    run, gt, f0, n_frames = open_run(run_dir)
    cls = get(tracker)
    if cls.offline:
        raise ValueError(f"追跡器 {tracker} はまとめて動かす（段階 3）")
    p = dict(load_preset(preset or "default", tracker) if (preset or tracker == "a") else {}, **(params or {}))
    det_dir = run / "bev_dets" / dets
    floor_z = load_space(gt["space"]).z
    out = {}
    for gname, group in load_groups(gt, group_names).items():
        cal = make_calib(gt, group.cameras)
        D = load_dets(det_dir, group.cameras)
        trk = cls(cal, group, gt["fps"], p, floor_z=floor_z)
        pts, t0 = [], time.perf_counter()
        for f in range(n_frames):
            pts += trk.step(f, D.get(f, []))
        pts += trk.finish()
        dt = time.perf_counter() - t0
        folder = run / "bev_tracks" / (label or tracker) / gname
        write_bev(folder / "bev.csv", pts, gt["fps"], f0)
        write_views(folder, pts, cal, gt, f0, floor_z)
        info = {"tracker": tracker, "group": gname, "group_def": group.to_json(), "dets": dets, "preset": preset or ("default" if tracker == "a" else None),
                "params": p, "frame_offset": f0, "frames": n_frames, "points": len(pts), "ids": len({q.id for q in pts}),
                "ms_per_frame": round(1000 * dt / max(n_frames, 1), 3), "version": version()}
        (folder / "run.json").write_text(json.dumps(info, ensure_ascii=False, indent=1))
        out[gname] = info
    return out


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
