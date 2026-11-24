"""追跡器（base.py の約束）と、群ごとに検出から床の上の軌跡を作る入口 track。"""

import json
import time
from pathlib import Path

import numpy as np

from ..calib import calib as make_calib
from ..evaluate import crop
from ..graph import graph as write_graph
from ..groups import groups as load_groups
from ..io import load_bev, load_dets, open_run, run_space, version, write_bev
from ..tasks import predict as predict_tasks
from ..tasks import write as write_tasks
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


PERTURB_KEYS = ("sigma_a", "r_suppress", "sig_perp", "sig_ray", "gate_chi2", "v_damp")


def perturbations(p, fps):
    """A の小さな揺らし 15 通り（外の実験と同じ）: そのまま、主な 6 つを ×0.97・×1.03（v_damp は 1 まで）、max_age_s を ±1 コマ。"""
    out = [{}]
    for k in PERTURB_KEYS:
        for m in (0.97, 1.03):
            out.append({k: min(p[k] * m, 1.0) if k == "v_damp" else p[k] * m})
    out += [{"max_age_s": p["max_age_s"] - 1 / fps}, {"max_age_s": p["max_age_s"] + 1 / fps}]
    return out


def track(run_dir, tracker="a", group_names=None, dets="gt_boxes", preset=None, params=None, label=None, repeats=1, perturb=False,
          **offline_opts):
    """群ごとに bev_tracks/<label>/<群>/ を作る（label の既定は追跡器の名前）。設定は 追跡器の既定 → presets/<preset>.json
    （既定は追跡器ごと: A は default、MV3DT は mv3dt）→ params の順に重ねる。
    A は bev_dets/<dets>/ の検出から。MV3DT（offline）は動画から中で検出して動かす（offline_opts: gpu・image・models・keep_raw）。
    repeats が 2 以上なら <label>_r1, <label>_r2, … に分けて書く（実行ごとにぶれる追跡器を比べるため）。
    perturb なら、設定を少し揺らした 15 通りを <label>_p0（揺らさない）〜 _p14 に書く（設定の小さな違いでどれだけ変わるか。A だけ）。"""
    run, gt, f0, n_frames = open_run(run_dir)
    cls = get(tracker)
    preset = preset or cls.default_preset
    p = dict(load_preset(preset, tracker) if preset else {}, **(params or {}))
    if offline_opts and not cls.offline:
        raise ValueError(f"追跡器 {tracker} には {', '.join(offline_opts)} を使わない")
    space = run_space(run, gt)
    floor_z = space.z
    label = label or tracker
    runs = [(label, p)] if repeats <= 1 else [(f"{label}_r{k}", p) for k in range(1, repeats + 1)]
    if perturb:
        if cls.offline or not hasattr(cls, "DEFAULTS"):
            raise ValueError(f"--perturb は A のような設定の決まった追跡器だけ（今は {tracker}）")
        if repeats > 1:
            raise ValueError("--perturb と --repeats は一緒に使わない")
        full = {**cls.DEFAULTS, **p}
        runs = [(f"{label}_p{k}", {**p, **d}) for k, d in enumerate(perturbations(full, gt["fps"]))]
    out = {}
    for gname, group in load_groups(gt, group_names, space).items():
        cal = make_calib(gt, group.cameras)
        use_app = tracker == "a" and {**cls.DEFAULTS, **p}["be_mode"] != "none" and {**cls.DEFAULTS, **p}["be_w_app"] > 0 \
            and {**cls.DEFAULTS, **p}["be_app"] == "hist"
        if use_app:  # A の窓最適化の見え方（胴の色）: 動画から作って bev_dets/ に置く（初めの 1 回）
            from ..appearance import ensure
            ensure(run, gt, run / "bev_dets" / dets, group.cameras)
        D = None if cls.offline else load_dets(run / "bev_dets" / dets, group.cameras, app=use_app)
        for lab, pk in runs:
            folder = run / "bev_tracks" / lab / gname
            trk = cls(cal, group, gt["fps"], pk, floor_z=floor_z)
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
            ids = aicity_ids(run, gt)
            if cls.offline:
                write_native_views(folder, obs, cal, f0, n_frames, ids)
            else:
                write_views(folder, pts, cal, gt, f0, floor_z, ids)
            write_products(run, gt, f0, n_frames, space, group, folder)
            info = {"tracker": tracker, "group": gname, "group_def": group.to_json(),
                    "dets": "internal" if cls.offline else dets, "preset": preset,
                    "params": pk, "frame_offset": f0, "frames": n_frames, "points": len(pts), "ids": len({q.id for q in pts}),
                    "ms_per_frame": round(1000 * dt / max(n_frames, 1), 3), "version": version(), **({"run": extra} if extra else {})}
            (folder / "run.json").write_text(json.dumps(info, ensure_ascii=False, indent=1))
            out[gname if len(runs) == 1 else f"{gname}/{lab}"] = info
    return out


def aicity_ids(run, gt):
    """カメラの番号: 正解の aicity/cameras.json と同じ（なければ全部のカメラの名前の順。outputs.write_aicity と同じ決め方）。"""
    p = run / "aicity" / "cameras.json"
    return json.loads(p.read_text()) if p.exists() else {c: i for i, c in enumerate(sorted(gt["cameras"]), start=1)}


def write_aicity(folder, lines, ids):
    """aicity/tracks.txt（正解の aicity/gt.txt と同じ形）と aicity/cameras.json（正解と同じ番号）。"""
    (folder / "aicity").mkdir(parents=True, exist_ok=True)
    (folder / "aicity" / "tracks.txt").write_text("".join(lines))
    (folder / "aicity" / "cameras.json").write_text(json.dumps(ids, ensure_ascii=False))


def write_products(run, gt, f0, n_frames, space, group, folder, kind="online"):
    """bev.csv（評価領域の中の点）→ 本体と同じ形の graph/（zone_counts.csv・zone_flows.csv）と tasks.json。"""
    pred = crop(load_bev(folder / "bev.csv", kind), group)
    write_graph(gt, f0, n_frames, space, group, pred, folder / "graph")
    tp = run / "tasks.json"
    levels = json.loads(tp.read_text())["5_congestion"]["levels_per_m2"] if tp.exists() else None
    write_tasks(folder / "tasks.json", predict_tasks(gt, f0, n_frames, space, group, pred, levels))


def write_native_views(folder, obs, cal, f0, n_frames, ids):
    """追跡器が出したカメラごとの枠（MV3DT）: mot/<カメラ>.txt と aicity/（write_views と同じ形）。"""
    (folder / "mot").mkdir(parents=True, exist_ok=True)
    rows, lines = {c: [] for c in cal}, []
    idx = {c: ids[c] for c in cal}
    for f in sorted(obs):
        if f >= n_frames:
            continue
        for cam, i, (x0, y0, x1, y1), (x, y), conf in obs[f]:
            rows[cam].append(f"{f + f0 + 1},{i},{x0:.1f},{y0:.1f},{x1 - x0:.1f},{y1 - y0:.1f},{conf:.4f},-1,-1\n")
            if np.isfinite([x, y]).all():
                lines.append(f"{idx[cam]} {i} {f + f0 + 1} {x0:.1f} {y0:.1f} {x1 - x0:.1f} {y1 - y0:.1f} {x:.3f} {y:.3f}\n")
    for cam, r in rows.items():
        (folder / "mot" / f"{cam}.txt").write_text("".join(r))
    write_aicity(folder, lines, ids)


def write_views(folder, pts, cal, gt, f0, floor_z, ids, kind="online"):
    """床の位置からカメラごとの枠（背 1.7 m・半径 0.3 m の円柱を写した箱）: mot/<カメラ>.txt と aicity/（正解の aicity/ と同じ形）。"""
    (folder / "mot").mkdir(parents=True, exist_ok=True)
    pts = [q for q in pts if q.kind == kind]
    xy = np.array([[q.x, q.y] for q in pts], float).reshape(-1, 2)
    lines = []
    for cam, c in cal.items():
        ci = ids[cam]
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
    write_aicity(folder, lines, ids)


__all__ = ["PERSON_H", "aicity_ids", "write_products",
           "PERSON_R", "TRACKERS", "Tracker", "TrackPoint", "get", "load_preset", "perturbations", "presets", "register", "track"]
