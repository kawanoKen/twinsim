"""中間層のグラフ: 床の上の追跡 → ゾーンの人数（秒ごと）・ゾーン間の移動（1 分ごと）を、シミュレータの graph/ と同じ形で
bev_tracks/<追跡器>/<群>/graph/ に書き（bev-track）、正解と比べる（bev-eval。映像からでもシミュレータからでも同じ形）。

比べるのは、床の coverage_min 以上が群の評価領域に入るゾーンだけ（評価領域の外は追跡しないので）。
正解は同じコマの全員（部屋の中の人を除く）。予測は評価領域の中の点だけ。
"""

import csv
from collections import Counter

import numpy as np

from .io import gt_positions

COVERAGE_MIN = 0.6


def zone_coverage(space, group):
    """{ゾーン: 歩ける床のうち評価領域に入る割合}。"""
    out = {}
    for z in space.zones:
        pts = [space.center(i, j) for i, j in zip(*np.nonzero(space.grid)) if space.zone_of(*space.center(i, j)) == z]
        if pts:
            out[z] = float(group.contains(np.array(pts)).mean())
    return out


def write_counts(path, zones, rows):
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["t"] + zones)
        w.writerows(rows)


def flows_of(frames_xy, space, fps, f0):
    """{コマ: [(ID, (x, y))]} → Counter((分, 元のゾーン, 先のゾーン))。同じ ID のゾーンが変わったら 1 回（シミュレータの graph/ と同じ）。"""
    last, flows = {}, Counter()
    for f in sorted(frames_xy):
        for i, (x, y) in frames_xy[f]:
            z = space.zone_of(x, y)
            if z is None:
                continue
            if i in last and last[i] != z:
                flows[(int((f + f0) / fps // 60), last[i], z)] += 1
            last[i] = z
    return flows


def graph(gt, f0, n_frames, space, group, pred, folder=None, frames=None, coverage_min=COVERAGE_MIN):
    """pred {動画のコマ: [(ID, (x, y))]}（評価領域で切ったもの）と正解の比べを返す。folder を渡せば folder/{zone_counts.csv, zone_flows.csv}
    （全部の秒）も書く。比べるのは frames（動画のコマ。既定は全部）に入る秒と移動だけ。"""
    fps = gt["fps"]
    zones = list(space.zones)
    cov = zone_coverage(space, group)
    seen = [z for z in zones if cov.get(z, 0) >= coverage_min]
    pos = {}
    for f, v in gt_positions(gt, f0, n_frames).items():
        names = list(v)
        ok = group.contains([v[n] for n in names]) if names else []
        pos[f] = {n: v[n] for n, k in zip(names, ok) if k}
    use = set(range(n_frames)) if frames is None else set(frames)
    secs = sorted({(f + f0) // fps for f in range(n_frames)})
    rows_p, rows_g = [], []
    for t in secs:
        f = min(max(t * fps - f0, 0), n_frames - 1)  # その秒の最初のコマ
        cp = Counter(space.zone_of(x, y) for _, (x, y) in pred.get(f, []))
        cg = Counter(space.zone_of(x, y) for x, y in pos.get(f, {}).values())
        rows_p.append([t] + [cp.get(z, 0) for z in zones])
        rows_g.append([t] + [cg.get(z, 0) for z in zones])
    if folder is not None:
        folder.mkdir(parents=True, exist_ok=True)
        write_counts(folder / "zone_counts.csv", zones, rows_p)
        with open(folder / "zone_flows.csv", "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["minute", "source", "target", "count"])
            for (m, a, b), c in sorted(flows_of(pred, space, fps, f0).items()):
                w.writerow([m, a, b, c])
    # 比べるのは frames の中だけ
    fp = flows_of({f: v for f, v in pred.items() if f in use}, space, fps, f0)
    gflow = flows_of({f: list(v.items()) for f, v in pos.items() if f in use}, space, fps, f0)
    sel = [k for k, t in enumerate(secs) if min(max(t * fps - f0, 0), n_frames - 1) in use]
    rows_p, rows_g = [rows_p[k] for k in sel], [rows_g[k] for k in sel]
    zi = {z: k + 1 for k, z in enumerate(zones)}
    cmp = {}
    for z in seen:
        p = np.array([r[zi[z]] for r in rows_p], float)
        g = np.array([r[zi[z]] for r in rows_g], float)
        cmp[z] = {"coverage": round(cov[z], 3), "mean_pred": round(float(p.mean()), 3), "mean_truth": round(float(g.mean()), 3),
                  "mae": round(float(np.abs(p - g).mean()), 3), "bias": round(float((p - g).mean()), 3)}
    tot_p = np.array([sum(r[1:]) for r in rows_p], float)
    tot_g = np.array([sum(r[1:]) for r in rows_g], float)
    keys = set(fp) | set(gflow)
    pairs = {(a, b) for _, a, b in keys if a in seen and b in seen}
    flow_cmp = {f"{a}->{b}": {"pred": sum(c for (_, x, y), c in fp.items() if (x, y) == (a, b)),
                              "truth": sum(c for (_, x, y), c in gflow.items() if (x, y) == (a, b))} for a, b in sorted(pairs)}
    res = {"seconds": len(rows_p), "coverage_min": coverage_min, "zones_compared": cmp,
           "area_count": {"mean_pred": round(float(tot_p.mean()), 3) if len(tot_p) else None,
                          "mean_truth": round(float(tot_g.mean()), 3) if len(tot_g) else None,
                          "mae": round(float(np.abs(tot_p - tot_g).mean()), 3) if len(tot_p) else None},
           "flows_compared": flow_cmp}
    return res
