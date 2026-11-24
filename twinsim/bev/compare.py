"""ブランチ with-mopf の MOPF の連携（twinsim/mopf）と比べる。with-mopf のコードには依存せず、その出力のファイルだけを読む。

  mopf_tracks/<カメラ>.jsonl・info.json   MOPF の 1 台ずつの追跡（with-mopf の mopf-process。frame_id は frame_offset からのコマ）
  mopf_eval/links.json                    with-mopf の mopf-link の結果（トラックレット → 全体の番号）

  camera_tracks  MOPF がないときの代わり: 検出（bev_dets/<名前>/）から 1 台ずつの追跡を mopf_tracks の形で書く。
                 ByteTrack の考え方の簡単な追跡で、MOPF（YOLO11m・QAQS・BoostTrack）より ID がよく切れる。数字は MOPF の代わりにならない
  import_mopf    links.json の全体の番号で、MOPF の枠を bev_tracks/<label>/<群>/ に入れる
                 （mot/<カメラ>.txt は MOPF の枠そのまま、bev.csv は枠の足元を床に写した点。同じ番号・同じコマの点は平均）
  compare        いくつかの追跡を、同じカメラ（既定は MOPF の 2 台）の画面の上で、with-mopf の mopf-link と同じ決まりで採点する:
                 見えている割合 0.4 以上の正解だけ、IoU 0.3 以上で対応、カメラをまたいで同じ番号なら同じ人（2 台を合わせた IDF1）。
                 床の上の追跡は、そのカメラの枠がなければ床の位置から作る（背 1.7 m・半径 0.3 m の円柱）。
                 評価領域（area_group の area）の外の人も判定しない（関係ない人は数えない）。
                 カメラの間の移り（ある人が a で見えなくなり、90 秒以内に b に現れた）を同じ番号でつないだ割合も出す（with-mopf の cross_recall と同じ考え方）
"""

import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

from ..camera import Camera
from .calib import calib as make_calib
from .evaluate import IOU_MIN, VIS_MIN, camera_frames, clear_idf1, iou_dist, mot_stride, outside_ids, read_mot
from .groups import groups as load_groups
from .io import load_bev, load_dets, open_run, run_space
from .trackers.a import cylinder_box

GAP_CROSS_S = 90.0   # カメラの間の移りとみなす間（with-mopf の linking.py と同じ）


# ---- MOPF の代わりの 1 台ずつの追跡 ----
HI, LO, NEW, IOU_HI, IOU_LO, MAX_AGE_S, MIN_HITS = 0.5, 0.2, 0.6, 0.3, 0.5, 3.0, 3


class _Box:
    """等速のカルマンフィルタ（枠の中心・幅・高さとその速さ）。"""
    F = np.eye(8)
    F[:4, 4:] = np.eye(4)
    H = np.eye(4, 8)

    def __init__(self, b, i):
        self.x = np.r_[self.z(b), np.zeros(4)]
        self.P = np.diag([10.0] * 4 + [100.0] * 4)
        self.id, self.age, self.hits = i, 0, 1

    @staticmethod
    def z(b):
        return np.array([(b[0] + b[2]) / 2, (b[1] + b[3]) / 2, b[2] - b[0], b[3] - b[1]])

    def box(self):
        cx, cy, w, h = self.x[:4]
        return [cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2]

    def predict(self):
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + np.diag([1.0] * 4 + [0.1] * 4)
        self.age += 1

    def update(self, b):
        y = self.z(b) - self.H @ self.x
        S = self.H @ self.P @ self.H.T + np.eye(4) * 4
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x, self.P = self.x + K @ y, (np.eye(8) - K @ self.H) @ self.P
        self.age, self.hits = 0, self.hits + 1


def _assoc(tracks, dets, thr):
    if not tracks or not len(dets):
        return [], list(range(len(tracks))), list(range(len(dets)))
    m = 1 - iou_dist([t.box() for t in tracks], dets[:, :4])
    r, c = linear_sum_assignment(-m)
    pairs = [(i, j) for i, j in zip(r, c) if m[i, j] >= thr]
    ti, dj = {i for i, _ in pairs}, {j for _, j in pairs}
    return pairs, [i for i in range(len(tracks)) if i not in ti], [j for j in range(len(dets)) if j not in dj]


def track_camera(dets, n_frames, fps):
    """{コマ: [(x1, y1, x2, y2, conf)]} → [(コマ, ID, 枠)]。確信度の高い検出から IoU で対応し、残りの軌跡を低い検出で拾う（ByteTrack）。"""
    tracks, nid, out = [], 1, []
    max_age = int(round(MAX_AGE_S * fps))
    for f in range(n_frames):
        D = np.array(dets.get(f, []), float).reshape(-1, 5)
        for t in tracks:
            t.predict()
        hi, lo = D[D[:, 4] >= HI], D[(D[:, 4] >= LO) & (D[:, 4] < HI)]
        pairs, rest_t, rest_d = _assoc(tracks, hi, IOU_HI)
        for i, j in pairs:
            tracks[i].update(hi[j, :4])
        rest = [tracks[i] for i in rest_t]
        for i, j in _assoc(rest, lo, IOU_LO)[0]:
            rest[i].update(lo[j, :4])
        for j in rest_d:
            if hi[j, 4] >= NEW:
                tracks.append(_Box(hi[j, :4], nid))
                nid += 1
        tracks = [t for t in tracks if t.age <= max_age]
        out += [(f, t.id, t.box()) for t in tracks if t.age == 0 and t.hits >= MIN_HITS]
    return out


def camera_tracks(run_dir, cameras, dets="peoplenet", out=None):
    """bev_dets/<dets>/ の検出から、1 台ずつの追跡を MOPF の形（<out>/<カメラ>.jsonl・info.json）で書く。out の既定は <結果>/mopf_tracks_<dets>。"""
    run, gt, f0, n_frames = open_run(run_dir)
    out = Path(out) if out else run / f"mopf_tracks_{dets}"
    out.mkdir(parents=True, exist_ok=True)
    D = load_dets(run / "bev_dets" / dets, cameras)
    for cam in cameras:
        per = defaultdict(list)
        for f, items in D.items():
            per[f] += [(*b, conf) for c, b, conf in items if c == cam]
        with open(out / f"{cam}.jsonl", "w") as fh:
            for f, i, b in track_camera(per, n_frames, gt["fps"]):
                fh.write(json.dumps({"frame_id": f, "tracking_id": i, "position_x_min": round(b[0], 1), "position_y_min": round(b[1], 1),
                                     "position_x_max": round(b[2], 1), "position_y_max": round(b[3], 1)}) + "\n")
    (out / "info.json").write_text(json.dumps({"frame_offset": f0, "source": f"bev_dets/{dets}",
                                               "tracker": "twinsim.bev.compare.track_camera（MOPF の代わり。ByteTrack の考え方）"}, ensure_ascii=False))
    return out


# ---- MOPF の結果を入れる ----
def import_mopf(run_dir, group, label="mopf", tracks_dir=None, links=None):
    """MOPF の枠（tracks_dir、既定 <結果>/mopf_tracks）と with-mopf の links.json（既定 <結果>/mopf_eval/links.json）
    → bev_tracks/<label>/<group>/（mot/<カメラ>.txt・bev.csv・run.json）。"""
    run, gt, f0, n_frames = open_run(run_dir)
    tdir = Path(tracks_dir) if tracks_dir else run / "mopf_tracks"
    lk = json.loads(Path(links or run / "mopf_eval" / "links.json").read_text())
    info = json.loads((tdir / "info.json").read_text()) if (tdir / "info.json").exists() else {}
    off = int(info.get("frame_offset", f0)) - f0   # MOPF の frame_id → 動画のコマ
    fps = gt["fps"]
    space = run_space(run, gt)
    spans = defaultdict(list)
    for t in lk["tracklets"]:
        spans[(t["camera"], t["mopf_id"])].append((t["t0"], t["t1"], t["global_id"]))
    num = {g: i for i, g in enumerate(sorted({t["global_id"] for t in lk["tracklets"]}), start=1)}
    folder = run / "bev_tracks" / label / group
    (folder / "mot").mkdir(parents=True, exist_ok=True)
    floor = defaultdict(list)
    for cam in lk["cameras"]:
        c = Camera(gt["cameras"][cam])
        rows = []
        for line in (tdir / f"{cam}.jsonl").read_text().splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            f = int(r["frame_id"]) + off
            if not 0 <= f < n_frames:
                continue
            t = (f + f0) / fps
            g = next((g for a, b, g in spans.get((cam, int(r["tracking_id"])), []) if a - 1e-6 <= t <= b + 1e-6), None)
            if g is None:   # linking が使わなかった短い切れ端
                continue
            i = num[g]
            x0, y0, x1, y1 = r["position_x_min"], r["position_y_min"], r["position_x_max"], r["position_y_max"]
            rows.append(f"{f + f0 + 1},{i},{x0:.1f},{y0:.1f},{x1 - x0:.1f},{y1 - y0:.1f},1,-1,-1\n")
            q = c.floor_point((x0 + x1) / 2, y1, space.z)
            if q is not None:
                floor[(f, i)].append(q)
        (folder / "mot" / f"{cam}.txt").write_text("".join(rows))
    with open(folder / "bev.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["frame", "t", "id", "x", "y", "cov_xx", "cov_xy", "cov_yy", "n_views", "kind", "delay"])
        for (f, i), qs in sorted(floor.items()):
            x, y = np.mean(qs, 0)
            w.writerow([f, f"{(f + f0) / fps:.3f}", i, f"{x:.3f}", f"{y:.3f}", "nan", "nan", "nan", len(qs), "online", 0])
    (folder / "run.json").write_text(json.dumps({"tracker": "mopf", "group": group, "cameras": lk["cameras"], "method": lk.get("method"),
                                                 "tracks_dir": str(tdir), "tracks_info": info, "global_ids": len(num)}, ensure_ascii=False, indent=1))
    return folder


# ---- 同じカメラの画面の上で比べる ----
def _boxes(run, gt, f0, n_frames, folder, cam, cal, floor_z):
    """追跡の 1 台の枠（read_mot の形）。folder/mot/<カメラ>.txt がなければ bev.csv の床の位置から作る。"""
    p = folder / "mot" / f"{cam}.txt"
    if p.exists():
        return read_mot(p, f0, n_frames)
    out = defaultdict(list)
    P = np.asarray(cal[cam]["P"], float)
    W, H = cal[cam]["width"], cal[cam]["height"]
    for f, items in load_bev(folder / "bev.csv").items():
        xy = np.array([q for _, q in items], float).reshape(-1, 2)
        depth = np.c_[xy, np.full(len(xy), floor_z), np.ones(len(xy))] @ P[2]
        for (i, _), b, z in zip(items, cylinder_box(P, xy, floor_z), depth):
            x0, y0, x1, y1 = max(b[0], 0), max(b[1], 0), min(b[2], W), min(b[3], H)
            if z > 0.1 and x1 - x0 >= 2 and y1 - y0 >= 2:
                out[f].append((i, (x0, y0, x1, y1), 1.0))
    return out


def _match(gf, pf, iou_min):
    """コマごとの正解 → 予測の ID（IoU が iou_min 以上、ハンガリー法）。"""
    out = {}
    for f, g in gf.items():
        p = pf.get(f, [])
        if not g or not p:
            continue
        names = list(g)
        m = 1 - iou_dist([g[n] for n in names], [b for _, b in p])
        r, c = linear_sum_assignment(-m)
        out[f] = {names[i]: p[j][0] for i, j in zip(r, c) if m[i, j] >= iou_min}
    return out


def handovers(segs, fps):
    """{人: [(カメラ, 最初のコマ, 最後のコマ)]}（時刻の順）→ カメラの間の移り [(人, a, a の最後, b, b の最初)]。"""
    out = []
    for n, ss in segs.items():
        for (ca, _, la), (cb, fb, _) in zip(ss, ss[1:]):
            if ca != cb and 0 <= fb - la <= GAP_CROSS_S * fps:
                out.append((n, ca, la, cb, fb))
    return out


def compare(run_dir, trackers, cameras=("CamGate", "CamEV"), area_group=None, frames=None, iou_min=IOU_MIN, vis_min=VIS_MIN):
    """trackers: [(名前, bev_tracks の下の場所 "<label>/<群>")]。cameras の画面の上で、2 台を合わせた IDF1 などを出す
    （with-mopf の mopf-link の採点と同じ決まり）。area_group の評価領域の外の人は判定しない（既定は最初の追跡の群）。
    返り値と bev_eval/compare_<カメラ>.json: {名前: {IDF1, IDP, IDR, MOTA, IDSW, Prec, Rec, n_gt, handovers, handovers_found, handovers_kept}}"""
    run, gt, f0, n_frames = open_run(run_dir)
    space = run_space(run, gt)
    frames = list(frames if frames is not None else range(n_frames))
    stride = mot_stride(run)
    fr = [f for f in frames if (f + f0) % stride == 0]
    gname = area_group or trackers[0][1].split("/")[1]
    group = load_groups(gt, [gname], space)[gname]
    outside = outside_ids(run, gt, f0, n_frames, group, frames)
    cal = make_calib(gt, list(cameras))
    G = {c: read_mot(run / "mot" / f"{c}.txt", f0, n_frames, True) for c in cameras}
    ids = json.loads((run / "mot/ids.json").read_text())
    name_of = {v: k for k, v in ids.items()}
    res = {}
    for name, where in trackers:
        folder = run / "bev_tracks" / where
        gall, pall, segs, matched = {}, {}, defaultdict(list), {}
        for k, cam in enumerate(cameras):
            T = _boxes(run, gt, f0, n_frames, folder, cam, cal, space.z)
            gf, pf, _ = camera_frames(G[cam], T, fr, outside, iou_min, vis_min)
            for f in fr:  # カメラをまたいで 1 つの列にする（with-mopf の score と同じく、コマの番号をカメラごとにずらす）
                gall[k * 10_000_000 + f] = gf.get(f, {})
                pall[k * 10_000_000 + f] = pf.get(f, [])
            matched[cam] = _match(gf, pf, iou_min)
            vis = defaultdict(list)
            for f in fr:
                for i in gf.get(f, {}):
                    vis[i].append(f)
            for i, fs in vis.items():   # 見えている区間（2 秒より長い途切れで分ける）
                a = b = fs[0]
                for f in fs[1:] + [None]:
                    if f is not None and f - b <= 2 * gt["fps"]:
                        b = f
                        continue
                    segs[i].append((cam, a, b))
                    if f is not None:
                        a = b = f
        keys = sorted(gall)
        r = clear_idf1(gall, pall, keys, 1 - iou_min, dist=iou_dist)
        hs = handovers({i: sorted(v, key=lambda s: s[1]) for i, v in segs.items()}, gt["fps"])
        found = kept = 0
        for i, ca, la, cb, fb in hs:
            ma = next((matched[ca][f][i] for f in range(la, la - gt["fps"] - 1, -1) if i in matched[ca].get(f, {})), None)
            mb = next((matched[cb][f][i] for f in range(fb, fb + gt["fps"] + 1) if i in matched[cb].get(f, {})), None)
            found += ma is not None and mb is not None
            kept += ma is not None and ma == mb
        res[name] = {"tracks": where, "IDF1": r["IDF1"], "MOTA": r["MOTA"], "IDSW": r["IDSW"], "Prec": r["Prec"], "Rec": r["Rec"],
                     "n_gt": r["n_gt"], "handovers": len(hs), "handovers_found": found, "handovers_kept": kept,
                     "handover_rate": kept / len(hs) if hs else None}
    out = {"cameras": list(cameras), "area_group": gname, "frames": [frames[0], frames[-1] + 1] if frames else None, "iou_min": iou_min,
           "vis_min": vis_min, "people_seen": len({name_of.get(i) for c in cameras for f in fr for i, _, v in G[c].get(f, []) if v >= vis_min}),
           "results": res}
    d = run / "bev_eval"
    d.mkdir(exist_ok=True)
    (d / f"compare_{'_'.join(cameras)}.json").write_text(json.dumps(out, ensure_ascii=False, indent=1))
    return out
