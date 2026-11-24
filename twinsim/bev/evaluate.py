"""床の上の追跡を正解と比べる（群ごと）: IDF1・HOTA・MOTA・MODA・ID の切り替わり・適合率・再現率・位置の誤差。

距離のしきい値（既定 1 m と 0.5 m）以内を同じ人とみなす。CLEAR MOT と IDF1 は motmetrics と同じ数え方
（前のコマの対応を先に保ち、残りをハンガリアン法で対応づける。IDF1 は ID 同士の全体の対応）、HOTA は TrackEval と同じ
（類似度 = max(0, 1 − 距離 / しきい値)、α = 0.05〜0.95）。依存は numpy と scipy だけ。

判定しない人（見逃しにも誤検出にも数えない。対応しなかった予測でその人の近くにあるものは捨てる）:
  vis_min   群のどのカメラでも、見えている割合（mot/ の近似）がこれ未満の人。None で使わない
  portal_s  評価領域の中で現れた・消えた人の、現れてから・消えるまでのこの秒数（扉・EV・湧き出し点）。0 で使わない
評価領域の外の正解と予測は数えない。
"""

import csv
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist

from .graph import graph as bev_graph
from .groups import groups as load_groups
from .io import gt_positions, load_bev, open_run, run_space, version
from .tasks import compare as compare_tasks
from .tasks import predict as predict_tasks

THRESHOLDS = (1.0, 0.5)
VIS_MIN = 0.4      # MOPF の比べ方（mopf/io.py の VIS_MIN）と同じ
PORTAL_S = 1.0
KEYS = ["IDF1", "HOTA", "DetA", "AssA", "MOTA", "MODA", "IDSW", "Prec", "Rec", "BEV_err_m", "n_gt_ids", "n_pred_ids", "n_gt"]


# ---- 指標 ----

def iou_dist(a, b):
    """枠 (N, 4)・(M, 4)（x1, y1, x2, y2）→ 1 − IoU。"""
    a, b = np.asarray(a, float).reshape(-1, 4), np.asarray(b, float).reshape(-1, 4)
    w = np.clip(np.minimum(a[:, None, 2], b[None, :, 2]) - np.maximum(a[:, None, 0], b[None, :, 0]), 0, None)
    h = np.clip(np.minimum(a[:, None, 3], b[None, :, 3]) - np.maximum(a[:, None, 1], b[None, :, 1]), 0, None)
    inter = w * h
    area = lambda x: (x[:, 2] - x[:, 0]) * (x[:, 3] - x[:, 1])  # noqa: E731
    return 1 - inter / np.maximum(area(a)[:, None] + area(b)[None] - inter, 1e-12)


def clear_idf1(gt_frames, pred_frames, frames, tau, dist=cdist):
    """gt_frames {コマ: {人: (x, y)}}、pred_frames {コマ: [(ID, (x, y))]} → CLEAR MOT と IDF1（motmetrics と同じ数え方）。
    dist: 距離の関数（既定は床の上の距離。カメラごとの評価は枠の 1 − IoU）。tau 以内を対応とする。"""
    m = {}                    # 正解 → 直前に対応した予測
    tp = fn = fp = sw = 0
    dists = []
    co = defaultdict(int)     # (正解, 予測) が tau 以内で同じコマにいた回数（IDF1）
    n_gt = n_pr = 0
    for f in frames:
        g = gt_frames.get(f, {})
        gids = list(g)
        pr = pred_frames.get(f, [])
        pids = [i for i, _ in pr]
        n_gt += len(gids)
        n_pr += len(pids)
        if not gids or not pids:
            fn += len(gids)
            fp += len(pids)
            continue
        d = dist(np.array([g[k] for k in gids]), np.array([xy for _, xy in pr]))
        valid = d <= tau
        for i, j in zip(*np.nonzero(valid)):
            co[(gids[i], pids[j])] += 1
        pidx = {p: j for j, p in enumerate(pids)}
        rows, cols = set(), set()
        pairs = []
        for i, gk in enumerate(gids):         # 1. 前のコマの対応を、まだ距離の中なら保つ
            j = pidx.get(m.get(gk))
            if j is not None and valid[i, j] and j not in cols:
                pairs.append((i, j))
                rows.add(i)
                cols.add(j)
        ri = [i for i in range(len(gids)) if i not in rows]
        ci = [j for j in range(len(pids)) if j not in cols]
        if ri and ci:                          # 2. 残りをハンガリアン法で
            sub = d[np.ix_(ri, ci)]
            r, c = linear_sum_assignment(np.where(sub <= tau, sub, 1e6))
            for a, b in zip(r, c):
                if sub[a, b] <= tau:
                    i, j = ri[a], ci[b]
                    if gids[i] in m and m[gids[i]] != pids[j]:
                        sw += 1
                    pairs.append((i, j))
        for i, j in pairs:
            m[gids[i]] = pids[j]
            dists.append(d[i, j])
        tp += len(pairs)
        fn += len(gids) - len(pairs)
        fp += len(pids) - len(pairs)
    # IDF1: ID 同士の一対一の対応で、同じコマに tau 以内にいた回数の和を最大にする
    G = sorted({k for k, _ in co})
    P = sorted({k for _, k in co})
    idtp = 0
    if G and P:
        gi, pi = {k: i for i, k in enumerate(G)}, {k: i for i, k in enumerate(P)}
        W = np.zeros((len(G), len(P)))
        for (a, b), n in co.items():
            W[gi[a], pi[b]] = n
        r, c = linear_sum_assignment(-W)
        idtp = int(W[r, c].sum())
    n = max(n_gt, 1)
    return {"IDF1": 2 * idtp / max(n_gt + n_pr, 1), "MOTA": 1 - (fn + fp + sw) / n, "MODA": 1 - (fn + fp) / n, "IDSW": sw,
            "Prec": tp / max(tp + fp, 1), "Rec": tp / n, "BEV_err_m": float(np.mean(dists)) if dists else None,
            "TP": tp, "FN": fn, "FP": fp, "n_gt": n_gt}


def hota(gt_frames, pred_frames, frames, thr):
    """床の上の HOTA（TrackEval と同じ）。"""
    alphas = np.arange(0.05, 0.96, 0.05)
    gmap, pmap, data = {}, {}, []
    for f in frames:
        g = gt_frames.get(f, {})
        gi = np.array([gmap.setdefault(k, len(gmap)) for k in g], int)
        gxy = np.array(list(g.values()), float).reshape(-1, 2)
        pr = pred_frames.get(f, [])
        pi = np.array([pmap.setdefault(k, len(pmap)) for k, _ in pr], int)
        pxy = np.array([xy for _, xy in pr], float).reshape(-1, 2)
        sim = np.clip(1 - cdist(gxy, pxy) / thr, 0, 1) if len(gi) and len(pi) else np.zeros((len(gi), len(pi)))
        data.append((gi, pi, sim))
    G, Pn = len(gmap), len(pmap)
    pot, gc, pc = np.zeros((G, Pn)), np.zeros(G), np.zeros(Pn)
    for gi, pi, sim in data:
        if len(gi) and len(pi):
            den = sim.sum(0)[None] + sim.sum(1)[:, None] - sim
            pot[np.ix_(gi, pi)] += np.divide(sim, den, out=np.zeros_like(sim), where=den > 1e-12)
        gc[gi] += 1
        pc[pi] += 1
    gas = pot / np.maximum(gc[:, None] + pc[None] - pot, 1e-12)
    TP, mc = np.zeros(len(alphas)), np.zeros((len(alphas), G, Pn))
    ng, npd = sum(len(gi) for gi, _, _ in data), sum(len(pi) for _, pi, _ in data)
    for gi, pi, sim in data:
        if not len(gi) or not len(pi):
            continue
        r, c = linear_sum_assignment(-gas[np.ix_(gi, pi)] * sim)
        for a, al in enumerate(alphas):
            ok = sim[r, c] >= al - 1e-9
            TP[a] += ok.sum()
            mc[a, gi[r[ok]], pi[c[ok]]] += 1
    AssA = np.array([(mc[a] * (mc[a] / np.maximum(gc[:, None] + pc[None] - mc[a], 1e-12))).sum() / max(1, TP[a]) for a in range(len(alphas))])
    DetA = TP / np.maximum(ng + npd - TP, 1)
    return {"HOTA": float(np.sqrt(DetA * AssA).mean()), "DetA": float(DetA.mean()), "AssA": float(AssA.mean())}


def drop_ignored(gt_frames, ign_frames, pred_frames, frames, tau):
    """対応しなかった予測のうち、判定しない人から tau 以内のものを捨てる。"""
    out = {}
    for f in frames:
        pr, ig = pred_frames.get(f, []), ign_frames.get(f, [])
        if not pr or not ig:
            out[f] = pr
            continue
        pxy = np.array([xy for _, xy in pr])
        g = np.array(list(gt_frames.get(f, {}).values()), float).reshape(-1, 2)
        matched = set()
        if len(g):
            d = cdist(g, pxy)
            r, c = linear_sum_assignment(np.where(d <= tau, d, 1e6))
            matched = {j for i, j in zip(r, c) if d[i, j] <= tau}
        near = cdist(pxy, np.array(ig, float)).min(1) <= tau
        out[f] = [o for j, o in enumerate(pr) if j in matched or not near[j]]
    return out


def score(gt_frames, ign_frames, pred_frames, frames, tau):
    pred = drop_ignored(gt_frames, ign_frames, pred_frames, frames, tau)
    r = clear_idf1(gt_frames, pred, frames, tau)
    r.update(hota(gt_frames, pred, frames, tau))
    r["n_gt_ids"] = len({k for f in frames for k in gt_frames.get(f, {})})
    r["n_pred_ids"] = len({i for f in frames for i, _ in pred.get(f, [])})
    return r


# ---- 正解の用意 ----

def visibility(run, gt, f0, n_frames, cameras):
    """{動画のコマ: {人: 群のカメラでいちばん見えている割合}}（mot/ の近似）。mot/ がなければ None。"""
    ids_p = run / "mot/ids.json"
    if not ids_p.exists() or not all((run / "mot" / f"{c}.txt").exists() for c in cameras):
        return None
    name_of = {v: k for k, v in json.loads(ids_p.read_text()).items()}
    vis = defaultdict(dict)
    for c in cameras:
        for line in (run / "mot" / f"{c}.txt").read_text().splitlines():
            q = line.split(",")
            f = int(q[0]) - 1 - f0
            if 0 <= f < n_frames:
                n = name_of[int(q[1])]
                vis[f][n] = max(vis[f].get(n, 0.0), float(q[8]))
    return vis


def ground_truth(run, gt, f0, n_frames, group, vis_min=VIS_MIN, portal_s=PORTAL_S):
    """(評価する正解 {コマ: {人: xy}}, 判定しない人 {コマ: [xy]})。どちらも評価領域の中だけ。"""
    pos = gt_positions(gt, f0, n_frames)
    vis = visibility(run, gt, f0, n_frames, group.cameras) if vis_min is not None else None
    vkeys = np.array(sorted(vis)) if vis else None   # mot_stride > 1 なら枠は何コマかおき: 直前の枠のコマの見え方を使う
    portal = int(round(portal_s * gt["fps"]))
    edge = {}   # 人 → (現れたコマ, 消えたコマ)（評価領域の中で現れた・消えたときだけ）
    if portal:
        for n, p in gt["people"].items():
            tr = gt["tracks"][n]
            a = p["start_frame"] - f0 if group.contains([tr[0][:2]])[0] else None
            b = p["start_frame"] + len(tr) - 1 - f0 if group.contains([tr[-1][:2]])[0] else None
            edge[n] = (a, b)
    keep, ign = {}, {}
    for f in range(n_frames):
        items = pos.get(f, {})
        names = list(items)
        inside = group.contains([items[n] for n in names]) if names else []
        g, i = {}, []
        fv = None
        if vkeys is not None:
            k = np.searchsorted(vkeys, f, side="right") - 1
            fv = vis[int(vkeys[k])] if k >= 0 else {}
        for n, ok in zip(names, inside):
            if not ok:
                continue
            a, b = edge.get(n, (None, None))
            dont = (a is not None and f - a < portal) or (b is not None and b - f < portal)
            if fv is not None and fv.get(n, 0.0) < vis_min:
                dont = True
            if dont:
                i.append(items[n])
            else:
                g[n] = items[n]
        keep[f], ign[f] = g, i
    return keep, ign


def crop(pred_frames, group):
    out = {}
    for f, pr in pred_frames.items():
        if pr:
            ok = group.contains([xy for _, xy in pr])
            out[f] = [o for o, k in zip(pr, ok) if k]
    return out


# ---- カメラごと（副） ----

IOU_MIN = 0.3      # MOPF の比べ方（mopf/io.py の IOU_MIN）と同じ
CAM_KEYS = ["IDF1", "MOTA", "MODA", "IDSW", "Prec", "Rec", "n_gt", "count_mae"]


def mot_stride(run):
    p = run / "scenario_resolved.json"
    return int(json.loads(p.read_text()).get("mot_stride") or 1) if p.exists() else 1


def read_mot(path, f0, n_frames, gt_file=False):
    """MOTChallenge の形 → {動画のコマ: [(ID, (x1, y1, x2, y2), 見えている割合)]}。"""
    out = defaultdict(list)
    if not path.exists():
        return out
    for line in path.read_text().splitlines():
        q = line.split(",")
        f = int(q[0]) - 1 - f0
        if 0 <= f < n_frames:
            x, y, w, h = map(float, q[2:6])
            out[f].append((int(q[1]), (x, y, x + w, y + h), float(q[8]) if gt_file else 1.0))
    return out


def per_camera(run, f0, n_frames, folder, cameras, frames, iou_min=IOU_MIN, vis_min=VIS_MIN, gt=None, group=None):
    """追跡のカメラごとの枠（folder/mot/<カメラ>.txt）を正解の mot/ と比べる（MOPF の mopf-eval と同じ決まり）。
    見えている割合が vis_min 未満の正解は判定しない（その枠とだけ重なる予測も数えない）。正解の枠のあるコマ（mot_stride）だけ。
    group を渡すと、床の位置が評価領域の外の人も判定しない（床の上の追跡器は評価領域の中だけを追うので）。"""
    stride = mot_stride(run)
    outside = None
    if group is not None and gt is not None and (run / "mot/ids.json").exists():
        ids = json.loads((run / "mot/ids.json").read_text())
        pos = gt_positions(gt, f0, n_frames)
        outside = {}
        for f in frames:
            items = pos.get(f, {})
            names = list(items)
            ok = group.contains([items[n] for n in names]) if names else []
            outside[f] = {ids[n] for n, k in zip(names, ok) if not k and n in ids}
    fr = [f for f in frames if (f + f0) % stride == 0]
    out = {}
    for cam in cameras:
        G, T = read_mot(run / "mot" / f"{cam}.txt", f0, n_frames, True), read_mot(folder / "mot" / f"{cam}.txt", f0, n_frames)
        if not (run / "mot" / f"{cam}.txt").exists():
            continue
        gf, pf, cnt = {}, {}, []
        for f in fr:
            g, t = G.get(f, []), T.get(f, [])
            out_f = outside.get(f, set()) if outside is not None else set()
            bad = [x[0] in out_f or (vis_min is not None and x[2] < vis_min) for x in g]
            gv = [x for x, b in zip(g, bad) if not b]
            gi = [x for x, b in zip(g, bad) if b]
            if gi and t:
                mi = 1 - iou_dist([x[1] for x in t], [x[1] for x in gi]).min(1)
                mv = 1 - iou_dist([x[1] for x in t], [x[1] for x in gv]).min(1) if gv else np.zeros(len(t))
                t = [x for x, a, b in zip(t, mi, mv) if not (a >= iou_min and b < iou_min)]
            gf[f] = {i: b for i, b, _ in gv}
            pf[f] = [(i, b) for i, b, _ in t]
            cnt.append(abs(len(t) - len(gv)))
        r = clear_idf1(gf, pf, fr, 1 - iou_min, dist=iou_dist)
        r["count_mae"] = float(np.mean(cnt)) if cnt else None
        out[cam] = {k: r[k] for k in CAM_KEYS}
    return out


# ---- 入口 ----

def evaluate(run_dir, tracker="a", group_names=None, thresholds=THRESHOLDS, vis_min=VIS_MIN, portal_s=PORTAL_S, frames=None, kind="online", label=None):
    """bev_tracks/<追跡器>/<群>/bev.csv を正解と比べ、bev_eval/<label>_<群>.json と bev_eval/summary.csv に書く。
    frames: 評価する動画のコマ（range。既定は全部）。label: 結果の名前（既定は追跡器の名前）。
    bev_tracks/<追跡器>/ がなく <追跡器>_r1, _r2, …（bev-track --repeats）か <追跡器>_p0, …（--perturb）があるなら、
    それぞれを評価し、平均と標準偏差を <label>_<群>.json に書く。"""
    run, gt, f0, n_frames = open_run(run_dir)
    reps = []
    for kind_ in ("r", "p"):  # --repeats（_r1, …）か --perturb（_p0, …）
        reps = reps or sorted((p.name for p in (run / "bev_tracks").glob(f"{tracker}_{kind_}*")
                               if re.fullmatch(rf"{re.escape(tracker)}_{kind_}\d+", p.name)), key=lambda n: int(re.findall(r"\d+$", n)[0]))
    if not (run / "bev_tracks" / tracker).exists() and reps:
        each = [evaluate(run_dir, r, group_names, thresholds, vis_min, portal_s, frames, kind) for r in reps]
        label = label or tracker
        results = {}
        for gname in each[0]:
            res = {**each[0][gname], "tracker": tracker, "label": label, "repeats": reps, "metrics": {}, "std": {}}
            for thr in each[0][gname]["metrics"]:
                ms = [e[gname]["metrics"][thr] for e in each]
                res["metrics"][thr] = {k: float(np.mean([m[k] for m in ms])) for k in ms[0]}
                res["std"][thr] = {k: float(np.std([m[k] for m in ms], ddof=1)) if len(ms) > 1 else 0.0 for k in ms[0]}
            res["per_camera"] = {c: {k: float(np.mean([e[gname]["per_camera"][c][k] for e in each])) for k in CAM_KEYS
                                     if all(e[gname]["per_camera"][c][k] is not None for e in each)}
                                 for c in each[0][gname].get("per_camera", {})}
            (run / "bev_eval" / f"{label}_{gname}.json").write_text(json.dumps(res, ensure_ascii=False, indent=1))
            results[gname] = res
        write_summary(run / "bev_eval")
        return results
    label = label or tracker
    frames = list(frames if frames is not None else range(n_frames))
    out_dir = run / "bev_eval"
    out_dir.mkdir(parents=True, exist_ok=True)
    results = {}
    space = run_space(run, gt)
    for gname, group in load_groups(gt, group_names, space).items():
        tp = run / "bev_tracks" / tracker / gname / "bev.csv"
        if not tp.exists():
            raise FileNotFoundError(f"追跡がない: {tp}（先に twinsim bev-track）")
        pred = crop(load_bev(tp, kind), group)
        keep, ign = ground_truth(run, gt, f0, n_frames, group, vis_min, portal_s)
        res = {"tracker": tracker, "label": label, "group": gname, "kind": kind, "frames": [frames[0], frames[-1]] if frames else None,
               "rules": {"vis_min": vis_min, "portal_s": portal_s, "crop_to_area": True, "vis_source": "mot (approximate)"},
               "version": version(), "metrics": {}}
        for tau in thresholds:
            res["metrics"][f"{tau:g}m"] = score(keep, ign, pred, frames, tau)
        res["graph"] = bev_graph(gt, f0, n_frames, space, group, pred, frames=frames)
        truth = json.loads((run / "tasks.json").read_text()) if (run / "tasks.json").exists() else None
        pt = predict_tasks(gt, f0, n_frames, space, group, pred, truth["5_congestion"]["levels_per_m2"] if truth else None)
        res["tasks"] = compare_tasks(pt, truth, gt, f0, frames)
        res["per_camera"] = per_camera(run, f0, n_frames, tp.parent, group.cameras, frames,
                                       vis_min=VIS_MIN if vis_min is not None else None, gt=gt, group=group)
        (out_dir / f"{label}_{gname}.json").write_text(json.dumps(res, ensure_ascii=False, indent=1))
        results[gname] = res
    write_summary(out_dir)
    return results


STD_KEYS = ("IDF1", "HOTA", "MOTA")  # summary.csv に標準偏差も書く指標（--repeats の結果）


def write_summary(out_dir):
    rows = []
    for p in sorted(Path(out_dir).glob("*.json")):
        r = json.loads(p.read_text())
        if "metrics" not in r:
            continue
        for thr, m in r["metrics"].items():
            sd = r.get("std", {}).get(thr, {})
            rows.append({"label": r["label"], "tracker": r["tracker"], "group": r["group"], "kind": r["kind"], "threshold": thr,
                         **{k: (round(m[k], 4) if isinstance(m.get(k), float) else m.get(k)) for k in KEYS},
                         "runs": len(r.get("repeats", [])) or 1, **{f"{k}_std": round(sd[k], 4) if k in sd else "" for k in STD_KEYS}})
    cam_rows = []
    for p in sorted(Path(out_dir).glob("*.json")):
        r = json.loads(p.read_text())
        for cam, m in (r.get("per_camera") or {}).items():
            cam_rows.append({"label": r["label"], "group": r["group"], "camera": cam,
                             **{k: (round(m[k], 4) if isinstance(m.get(k), float) else m.get(k)) for k in CAM_KEYS}})
    with open(Path(out_dir) / "per_camera.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["label", "group", "camera", *CAM_KEYS])
        w.writeheader()
        w.writerows(cam_rows)
    with open(Path(out_dir) / "summary.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["label", "tracker", "group", "kind", "threshold", *KEYS, "runs", *(f"{k}_std" for k in STD_KEYS)])
        w.writeheader()
        w.writerows(rows)
