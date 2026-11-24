"""追跡から、本体の tasks.json と同じ形の推定を作り（bev_tracks/<追跡器>/<群>/tasks.json、bev-track）、正解と比べる（bev-eval）。

推定するもの（評価領域に入る所だけ。どこを見たかは observed に書く）:
  1_count      present（評価領域の人数、秒ごと）、by_zone（歩ける床の COVERAGE_MIN 以上が評価領域に入るゾーン）
  2_direction  gate_per_min・gate_events: レーンの中心を結ぶ線を、レーンの幅の中で横切った時刻。向きは in_dir との内積
               （正解の t はゲートを通り始めた時刻なので、推定は 0.5〜1 秒ほど遅れる）。レーンが全部評価領域に入るゲートだけ
  3_dwell      止まった区間: 前後 0.5 秒の動きの速さが V_STOP 未満のひと続き（MERGE_S より短い動きはつなぐ）が STOP_S 以上。
               状態は分からないので state は stop
  5_congestion by_zone の人数 / (ゾーンの面積 × 評価領域に入る割合) → 段階（境いは正解の tasks.json と同じ）。
               評価領域の外の部分も同じ混み具合とみなす
  elevator     waits: かごの扉から R_DOOR 以内で終わった軌跡（乗ったとみなす）の、初めて止まった（EV_STOP_S 以上）時刻から終わりまで。
               board_s（乗った時刻）も書く。止まらずに乗ったら 0
推定しないもの（not_estimated）: 1_count.by_state（状態）、4_group、6_trend（現れた所・行き先は評価領域の外）。
person は追跡器の ID。時刻はシミュレータの秒、秒ごとの配列は正解と同じ長さで、動画の外の秒は null。
"""

import json
import math

import numpy as np
from scipy.optimize import linear_sum_assignment

from .graph import COVERAGE_MIN, zone_coverage

V_STOP = 0.3     # m/s。これより遅ければ止まっている（追跡の位置の揺れは 0.1 m ほど）
STOP_S = 3.0     # 正解の 3_dwell と同じ
EV_STOP_S = 1.0  # EV の待ちの始まりに使う止まった区間の長さ
R_DOOR = 2.0     # m
MERGE_S = 3.0    # 止まった区間の間の、これより短い動きはつなぐ（待つ人は人混みの中で少しずつ動く）
MATCH_S = 2.0    # ゲートの通過を正解と対応づける時刻の差
LEVELS = [0.10, 0.20]  # outputs.task_aggregates の既定
NOT_ESTIMATED = ["1_count.by_state", "4_group", "6_trend"]


def per_id(pred):
    """{コマ: [(ID, (x, y))]} → {ID: (コマの配列, (N, 2) の位置)}（コマの順）。"""
    tmp = {}
    for f in sorted(pred):
        for i, xy in pred[f]:
            tmp.setdefault(i, ([], []))
            tmp[i][0].append(f)
            tmp[i][1].append(xy)
    return {i: (np.array(a), np.array(b, float).reshape(-1, 2)) for i, (a, b) in tmp.items()}


def speeds(fr, xy, fps):
    """点ごとの前後 0.5 秒の動きの速さ（m/s）。"""
    h = max(1, int(round(0.5 * fps)))
    j0 = np.searchsorted(fr, fr - h)
    j1 = np.searchsorted(fr, fr + h, side="right") - 1
    dt = (fr[j1] - fr[j0]) / fps
    d = np.linalg.norm(xy[j1] - xy[j0], axis=1)
    return np.where(dt > 0, d / np.maximum(dt, 1e-9), 0.0)


def stops(fr, xy, fps, min_s, merge_s=MERGE_S):
    """止まった区間 [(始まりの添字, 終わりの添字)]（軌跡の間が 1 秒より空いたら分ける。merge_s より短い動きの間はつなぐ）。"""
    slow = speeds(fr, xy, fps) < V_STOP
    runs, k = [], 0
    while k < len(fr):
        if not slow[k]:
            k += 1
            continue
        a = k
        while k + 1 < len(fr) and slow[k + 1] and fr[k + 1] - fr[k] <= fps:
            k += 1
        runs.append([a, k])
        k += 1
    merged = []
    for a, b in runs:
        if merged and (fr[a] - fr[merged[-1][1]]) / fps < merge_s and all(np.diff(fr[merged[-1][1]:a + 1]) <= fps):
            merged[-1][1] = b
        else:
            merged.append([a, b])
    return [(a, b) for a, b in merged if (fr[b] - fr[a] + 1) / fps >= min_s]


def gate_lines(gt, group):
    """{ゲート: (レーンの中心, in_dir, 横の向き, 横の範囲, 線の位置)}（レーンが全部評価領域に入るゲートだけ）。"""
    out = {}
    for g, d in (gt.get("gates") or {}).items():
        lanes = np.asarray(d["lanes"], float).reshape(-1, 2)
        if not len(lanes) or not group.contains(lanes).all():
            continue
        n = np.asarray(d["in_dir"], float)
        n = n / np.linalg.norm(n)
        tang = np.array([-n[1], n[0]])
        s = np.sort(lanes @ tang)
        half = float(np.diff(s).mean()) / 2 if len(s) > 1 else 0.5
        out[g] = (lanes, n, tang, (s[0] - half, s[-1] + half), float((lanes @ n).mean()))
    return out


def gate_crossings(tracks, lines, fps, f0):
    """[{t, gate, lane, person, dir}]（時刻の順）。"""
    ev = []
    for i, (fr, xy) in tracks.items():
        for g, (lanes, n, tang, (s0, s1), c) in lines.items():
            d = xy @ n - c
            for k in range(len(fr) - 1):
                if fr[k + 1] - fr[k] > fps or not (d[k] < 0 <= d[k + 1] or d[k] >= 0 > d[k + 1]):
                    continue
                a = d[k] / (d[k] - d[k + 1])
                p = xy[k] + a * (xy[k + 1] - xy[k])
                s = float(p @ tang)
                if not s0 <= s <= s1:
                    continue
                lane = int(np.argmin(np.abs(lanes @ tang - s)))
                t = (fr[k] + a * (fr[k + 1] - fr[k]) + f0) / fps
                ev.append({"t": round(float(t), 2), "gate": g, "lane": lane, "person": int(i), "dir": "in" if d[k + 1] > d[k] else "out"})
    return sorted(ev, key=lambda e: e["t"])


def door_cars(gt, group):
    """{かご: 扉の位置}（扉のまわり 1 m のどこかが評価領域に入るかご）。"""
    out = {}
    ang = np.linspace(0, 2 * np.pi, 16, endpoint=False)
    for c, d in (gt.get("cars") or {}).items():
        door = np.asarray(d["door"], float)
        if group.contains(door + np.c_[np.cos(ang), np.sin(ang)]).any():
            out[c] = door
    return out


def predict(gt, f0, n_frames, space, group, pred, levels=None):
    """pred {動画のコマ: [(ID, (x, y))]}（評価領域で切ったもの）→ tasks.json と同じ形の dict。"""
    fps, total = gt["fps"], gt["total_frames"]
    S = len(range(0, total, fps))
    cov = zone_coverage(space, group)
    seen = [z for z in space.zones if cov.get(z, 0) >= COVERAGE_MIN]
    area = {z: space.zone_area(z) for z in seen}
    present = [None] * S
    by_zone = {z: [None] * S for z in seen}
    for s in range(S):
        f = s * fps - f0
        if not 0 <= f < n_frames:
            continue
        rows = pred.get(f, [])
        present[s] = len(rows)
        zs = [space.zone_of(x, y) for _, (x, y) in rows]
        for z in seen:
            by_zone[z][s] = zs.count(z)
    tracks = per_id(pred)
    lines = gate_lines(gt, group)
    events = gate_crossings(tracks, lines, fps, f0)
    minutes = int(math.ceil(total / fps / 60))
    gate_min = {f"{g}:{d}": [0] * minutes for g in lines for d in ("in", "out")}
    for e in events:
        gate_min[f"{e['gate']}:{e['dir']}"][min(int(e["t"] // 60), minutes - 1)] += 1
    dwell = []
    for i, (fr, xy) in tracks.items():
        for a, b in stops(fr, xy, fps, STOP_S):
            dwell.append({"person": int(i), "state": "stop", "start_s": round((fr[a] + f0) / fps, 2), "end_s": round((fr[b] + f0) / fps, 2),
                          "zone": space.zone_of(*xy[a])})
    dwell.sort(key=lambda d: d["start_s"])
    lv = levels or LEVELS
    eff = {z: area[z] * cov[z] for z in seen}
    density = {z: [None if n is None else round(n / eff[z], 3) if eff[z] else 0.0 for n in by_zone[z]] for z in seen}
    level = {z: [None if d is None else int(sum(d >= t for t in lv)) for d in density[z]] for z in seen}
    cars = door_cars(gt, group)
    waits = []
    for i, (fr, xy) in tracks.items():
        if not cars or fr[-1] >= n_frames - 1:  # 動画の終わりまで映っていた軌跡は乗っていない
            continue
        car, door = min(cars.items(), key=lambda c: np.linalg.norm(xy[-1] - c[1]))
        if np.linalg.norm(xy[-1] - door) > R_DOOR:
            continue
        st = stops(fr, xy, fps, EV_STOP_S)
        start = fr[st[0][0]] if st else fr[-1]
        waits.append({"person": int(i), "car": car, "wait_s": round((fr[-1] - start) / fps, 2), "board_s": round((fr[-1] + f0) / fps, 2)})
    waits.sort(key=lambda w: w["board_s"])
    return {"seconds": S, "zones": seen, "zone_area_m2": {z: round(area[z], 1) for z in seen},
            "observed": {"frames": [f0, f0 + n_frames - 1], "zone_coverage": {z: round(cov[z], 3) for z in seen}, "gates": list(lines),
                         "cars": list(cars)},
            "not_estimated": NOT_ESTIMATED,
            "1_count": {"present": present, "by_zone": by_zone},
            "2_direction": {"gate_per_min": gate_min, "gate_events": events},
            "3_dwell": dwell,
            "5_congestion": {"levels_per_m2": lv, "density": density, "level": level},
            "elevator": {"waits": waits}}


def write(path, tasks):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(tasks, ensure_ascii=False))


def _stats(v):
    v = np.asarray(v, float)
    return {"n": int(len(v)), "mean": round(float(v.mean()), 2) if len(v) else None, "median": round(float(np.median(v)), 2) if len(v) else None}


def compare(pred, truth, gt, f0, frames):
    """推定（predict の dict）を正解と比べる。truth: 正解の tasks.json（なければ None。滞留と混み具合は比べない）。
    比べるのは frames（動画のコマ）の時間だけ。"""
    fps = gt["fps"]
    frames = list(frames)
    t0, t1 = (frames[0] + f0) / fps, (frames[-1] + f0 + 1) / fps
    use = set(frames)
    secs = [s for s in range(pred["seconds"]) if s * fps - f0 in use]
    inside = lambda t: t0 <= t < t1  # noqa: E731
    out = {"window_s": [round(t0, 2), round(t1, 2)]}
    # ゲート: 向きごとの人数と、MATCH_S 以内の 1 対 1 の対応（時刻の近い順）
    gates = {}
    for g in pred["observed"]["gates"]:
        for d in ("in", "out"):
            te = sorted(e["t"] for e in gt.get("gate_events", []) if e["gate"] == g and e["dir"] == d and inside(e["t"]))
            pe = sorted(e["t"] for e in pred["2_direction"]["gate_events"] if e["gate"] == g and e["dir"] == d and inside(e["t"]))
            lag = []
            if te and pe:
                C = np.abs(np.subtract.outer(te, pe))
                for i, j in zip(*linear_sum_assignment(np.where(C <= MATCH_S, C, 1e6))):
                    if C[i, j] <= MATCH_S:
                        lag.append(pe[j] - te[i])
            m = len(lag)
            gates[f"{g}:{d}"] = {"truth": len(te), "pred": len(pe), "matched": m,
                                 "precision": round(m / len(pe), 3) if pe else None, "recall": round(m / len(te), 3) if te else None,
                                 "lag_s": round(float(np.mean(lag)), 2) if lag else None}
    out["gates"] = gates
    # EV: 乗った人の待ち時間の分布（正解は ground_truth.json の events の board − wait_start）
    cars = set(pred["observed"]["cars"])
    if cars:
        tw = [p["events"]["board"] - p["events"]["wait_start"] for p in gt["people"].values()
              if "board" in p["events"] and "wait_start" in p["events"] and p["events"].get("car") in cars and inside(p["events"]["board"])]
        pw = [w["wait_s"] for w in pred["elevator"]["waits"] if inside(w["board_s"])]
        out["elevator_wait_s"] = {"truth": _stats(tw), "pred": _stats(pw)}
    if truth is not None:
        zones = pred["zones"]
        dw = {}
        for z in zones:
            td = [d["end_s"] - d["start_s"] for d in truth["3_dwell"] if d["zone"] == z and inside(d["start_s"])]
            pd = [d["end_s"] - d["start_s"] for d in pred["3_dwell"] if d["zone"] == z and inside(d["start_s"])]
            if td or pd:
                dw[z] = {"truth": len(td), "pred": len(pd), "truth_s": round(float(sum(td)), 1), "pred_s": round(float(sum(pd)), 1)}
        out["dwell"] = dw
        cg = {}
        for z in zones:
            tl = truth["5_congestion"]["level"].get(z)
            if tl is None or not secs:
                continue
            a = np.array([pred["5_congestion"]["level"][z][s] for s in secs], float)
            b = np.array([tl[s] for s in secs], float)
            cg[z] = {"agree": round(float((a == b).mean()), 3), "mean_level_pred": round(float(a.mean()), 3), "mean_level_truth": round(float(b.mean()), 3)}
        out["congestion"] = cg
    return out
