"""床の上の追跡の共通の部品: 結果の場所を開く・正解の位置・検出と追跡のファイルの読み書き・版の記録。

コマは動画の 0 始まり（描画したなら render_log.json の frames[0] から数える。描画していなければシミュレータのコマと同じ）。
"""

import csv
import json
import subprocess
from collections import defaultdict
from pathlib import Path

from .. import __version__

DET_COLUMNS = ["frame", "x1", "y1", "x2", "y2", "conf"]
BEV_COLUMNS = ["frame", "t", "id", "x", "y", "cov_xx", "cov_xy", "cov_yy", "n_views", "kind", "delay"]


def open_run(run_dir):
    """(結果の場所, 正解, 動画の 0 コマ目のシミュレータのコマ, 動画のコマの数)。"""
    run = Path(run_dir).expanduser()
    gt = json.loads((run / "ground_truth.json").read_text())
    rl = run / "render_log.json"
    if rl.exists():
        a, b = json.loads(rl.read_text())["frames"]
    else:
        a, b = 0, gt["total_frames"] - 1
    return run, gt, int(a), int(b) - int(a) + 1


def run_space(run, gt):
    """結果の空間。設定に書いた空間（dict・ファイルの場所）も読めるよう、scenario_resolved.json の space を先に見る（labels.py と同じ）。"""
    from ..space import load_space
    resolved = Path(run) / "scenario_resolved.json"
    src = json.loads(resolved.read_text(encoding="utf-8")).get("space", gt["space"]) if resolved.exists() else gt["space"]
    return load_space(src)


def version():
    """twinsim の版と git のハッシュ（git の外なら None）。結果に必ず残す。"""
    try:
        h = subprocess.run(["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parent, capture_output=True, text=True, timeout=10)
        commit = h.stdout.strip() or None
        dirty = bool(subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=Path(__file__).resolve().parent,
                                    capture_output=True, text=True, timeout=10).stdout.strip()) if commit else None
    except (OSError, subprocess.SubprocessError):
        commit, dirty = None, None
    return {"twinsim": __version__, "git": commit, "dirty": dirty}


def gt_positions(gt, f0, n_frames):
    """{動画のコマ: {人: (x, y)}}（正解の全員。評価領域で絞るのは呼ぶ側）。部屋の中（状態 in_room、1.2。映らない）のコマは除く。"""
    out = defaultdict(dict)
    for name, p in gt["people"].items():
        s = p["start_frame"]
        hidden = {g for a, b, st in p.get("states", []) if st == "in_room" for g in range(a, b + 1)}
        for k, (x, y, _) in enumerate(gt["tracks"][name]):
            f = s + k - f0
            if 0 <= f < n_frames and s + k not in hidden:
                out[f][name] = (x, y)
    return out


# ---- 検出 ----

def write_dets(path, rows):
    """rows: [(frame, x1, y1, x2, y2, conf), ...] → CSV（見出しの行つき）。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(DET_COLUMNS)
        for f, x1, y1, x2, y2, c in rows:
            w.writerow([int(f), f"{x1:.2f}", f"{y1:.2f}", f"{x2:.2f}", f"{y2:.2f}", f"{c:.6f}"])


def load_dets(folder, cameras):
    """bev_dets/<検出器>/ → {動画のコマ: [(カメラ, (x1, y1, x2, y2), 確信度), ...]}。"""
    out = defaultdict(list)
    for cam in cameras:
        p = Path(folder) / f"{cam}.txt"
        if not p.exists():
            raise FileNotFoundError(f"検出がない: {p}（先に twinsim bev-detect）")
        with open(p, newline="") as fh:
            for r in csv.DictReader(fh):
                out[int(r["frame"])].append((cam, (float(r["x1"]), float(r["y1"]), float(r["x2"]), float(r["y2"])), float(r["conf"])))
    return out


# ---- 追跡 ----

def write_bev(path, points, fps, f0):
    """points: TrackPoint の並び → bev.csv（t はシミュレータの時刻の秒）。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(BEV_COLUMNS)
        for p in points:
            c = p.cov if p.cov is not None else ((float("nan"),) * 2,) * 2
            w.writerow([p.frame, f"{(p.frame + f0) / fps:.3f}", p.id, f"{p.x:.3f}", f"{p.y:.3f}", f"{c[0][0]:.5f}", f"{c[0][1]:.5f}", f"{c[1][1]:.5f}",
                        p.n_views, p.kind, p.delay])


def load_bev(path, kind="online"):
    """bev.csv → {動画のコマ: [(ID, (x, y)), ...]}（kind の行だけ）。"""
    out = defaultdict(list)
    with open(path, newline="") as fh:
        for r in csv.DictReader(fh):
            if r["kind"] == kind:
                out[int(r["frame"])].append((int(r["id"]), (float(r["x"]), float(r["y"]))))
    return out
