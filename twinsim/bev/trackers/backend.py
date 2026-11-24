"""A の窓最適化（後ろの段）: 直近の窓 [t − n, t] の観測の節を、前の段のつながり（系統の鍵）で小片に分け、
窓の始まりまでに決めた ID（錨）と小片を、最小費用の経路被覆（ハンガリアン法）でつなぎ直す。外の実験（cltrack.py の BackEnd）と同じ計算。

  費用   位置: 速度で補正した変位の 2 成分の混合ガウス（止まっている人・歩いている人。間が空くほど広がる）の負の対数尤度
         見え方: 胴の HSV の色のヒストグラム（カメラごと）の Bhattacharyya 係数の対数尤度比（同じ人・別の人の分布は正解を使わずに合わせた値）
  結果   窓の中の各コマの訂正した位置と ID（corrected）と、t での軌跡の状態（閉ループでは前の段の状態を置き換える）

時間の設定は秒で持つ（BE_DEFAULTS）。コマに直すのは max(1, round(秒 × fps))（min_len_s だけ 0 を許す）。
"""

from collections import defaultdict

import numpy as np
from scipy.optimize import linear_sum_assignment

APP_SAME, APP_DIFF = (0.769, 0.132), (0.652, 0.145)   # 小片の色の似かた（同じ人・別の人）の平均と標準偏差。正解を使わずに合わせた値

# 外の実験の値（15 fps のコマで書いた値を秒に直したもの）。gap_bonus・lineage_bonus などは費用の単位
BE_DEFAULTS = dict(split="", m_amb=0.1, g_cut_s=5 / 15, cp_len_s=10 / 15, d_cp=0.3, cp_kappa=1.0, fit_len_s=1.0,
                   sigma0=0.12, q_static=0.003, q_move=0.06, pi_static=0.75, gap_max_s=30.0, anchor_age_s=20.0,
                   app="hist", link_gain=5.0, lineage_bonus=3.0, gap_bonus=0.0, w_app=1.0, min_len_s=8 / 15, merge=True, d_dup=0.4,
                   merge_ov_s=5 / 15, shared_max=0.2, keep_label=True, lost_motion=False)


def to_frames(q, fps):
    """秒の設定 → コマ（外の実験の _fr と同じ丸め）。"""
    out = {k: v for k, v in q.items() if not k.endswith("_s")}
    for k, v in q.items():
        if k.endswith("_s"):
            n = int(round(v * fps))
            out[k[:-2]] = n if k == "min_len_s" else max(1, n)
    return out


def fit_end(nodes, L, at_end, fps):
    """小片の終わり（か始まり）の位置と、L 個までの節の最小二乗の速度。"""
    sel = nodes[-L:] if at_end else nodes[:L]
    fr = np.array([n["f"] for n in sel], float)
    Z = np.array([n["xy"] for n in sel])
    if len(sel) < 3 or np.ptp(fr) < max(1, int(round(3 * fps / 15))):
        return Z.mean(0), np.zeros(2)
    A = np.c_[np.ones(len(fr)), fr - fr.mean()]
    coef, *_ = np.linalg.lstsq(A, Z, rcond=None)
    v = coef[1] * fps
    v *= len(sel) / (len(sel) + 10.0 * fps / 15)   # 短い小片（0.7 秒未満）の速度は 0 へ縮める
    f0 = fr[-1] if at_end else fr[0]
    return coef[0] + coef[1] * (f0 - fr.mean()), v


def link_cost(pa, va, fa, pb, vb, fb, q, dt):
    """b が a の続きである負の対数尤度（間 0・ずれ 0 を基準）。大きく外れたら None。"""
    tg = (fb - fa) * dt
    e2 = float(np.sum((pb - pa - 0.5 * (va + vb) * tg) ** 2))
    ss = q["sigma0"] ** 2 + q["q_static"] * tg
    sm = q["sigma0"] ** 2 + q["q_move"] * tg
    if e2 > (3 * np.sqrt(sm) + 0.05) ** 2:
        return None
    lik = q["pi_static"] / ss * np.exp(-e2 / (2 * ss)) + (1 - q["pi_static"]) / sm * np.exp(-e2 / (2 * sm))
    return float(-np.log(lik * q["sigma0"] ** 2))


def piece_app(nodes, L, at_end):
    """終わり（か始まり）の L 個までの節のカメラごとの胴の色の平均（3 回以上見えたカメラだけ）。"""
    acc = defaultdict(list)
    for nd in (nodes[::-1] if at_end else nodes):
        for c, h in nd.get("app", {}).items():
            if len(acc[c]) < L:
                acc[c].append(h)
    return {c: np.mean(v, 0) for c, v in acc.items() if len(v) >= 3}


def app_cost(a, b, kind="hist"):
    """−log p(似かた | 同じ人) / p(似かた | 別の人)。共通のカメラがなければ 0。"""
    if kind != "hist":
        return 0.0
    ll = lambda s, m, sd: -0.5 * ((s - m) / sd) ** 2 - np.log(sd)  # noqa: E731
    cs = (a.keys() & b.keys()) - {-1}
    if not cs:
        return 0.0
    s = np.mean([np.sqrt(a[c] * b[c]).sum() for c in cs])
    return float(np.clip(ll(s, *APP_DIFF) - ll(s, *APP_SAME), -5, 5))


def change_points(F, Z, L, d_cp, kappa):
    """節の位置が跳ぶ所（前後 L 個の平均の差 > d_cp で、段差が直線よりずっと良く説明する所）。"""
    N = len(Z)
    if N < 2 * L:
        return []
    c = lambda x: np.r_[np.zeros((1,) + x.shape[1:]), np.cumsum(x, 0)]  # noqa: E731
    F = F - F[0]
    cf, cff, cz, czf, czz = c(F), c(F**2), c(Z), c(Z * F[:, None]), c((Z**2).sum(1))
    s = np.arange(L, N - L + 1)
    seg = lambda i, j: (j - i, cf[j] - cf[i], cff[j] - cff[i], cz[j] - cz[i], czf[j] - czf[i], czz[j] - czz[i])  # noqa: E731
    n, Sf, Sff, Sz, Szf, Szz = seg(s - L, s + L)
    var_f = np.maximum(Sff - Sf**2 / n, 1e-9)
    sse_line = Szz - (Sz**2).sum(1) / n - ((Szf - Sf[:, None] * Sz / n[:, None]) ** 2).sum(1) / var_f
    n1, _, _, Sz1, _, Szz1 = seg(s - L, s)
    n2, _, _, Sz2, _, Szz2 = seg(s, s + L)
    sse_step = Szz1 - (Sz1**2).sum(1) / n1 + Szz2 - (Sz2**2).sum(1) / n2
    J = np.linalg.norm(Sz2 / n2[:, None] - Sz1 / n1[:, None], axis=1)
    ok = (J > d_cp) & (sse_step < kappa * np.maximum(sse_line, 1e-9))
    out = []
    for k in np.argsort(-J):
        if ok[k] and all(abs(s[k] - o) >= L for o in out):
            out.append(int(s[k]))
    return sorted(out)


def make_tracklets(window_nodes, q):
    """窓の節を前の段の系統の鍵でまとめ、(a) g_cut より長い間、(b) あいまいな対応の区間の境、(c) 位置の跳びで切る。"""
    by = defaultdict(list)
    for f in sorted(window_nodes):
        for n in window_nodes[f]:
            by[n["key"]].append(n)
    pieces = []
    for key, ns in by.items():
        cuts = {}
        for k in range(1, len(ns)):
            if ns[k]["f"] - ns[k - 1]["f"] > q["g_cut"]:
                cuts[k] = "gap"
            elif "amb" in q["split"] and (ns[k]["amb"] < q["m_amb"]) != (ns[k - 1]["amb"] < q["m_amb"]):
                cuts[k] = "amb"
        if "cp" in q["split"]:
            for k in change_points(np.array([n["f"] for n in ns], float), np.array([n["xy"] for n in ns]), q["cp_len"], q["d_cp"], q["cp_kappa"]):
                cuts.setdefault(k, "cp")
        start, cut = 0, "start"
        for k in sorted(cuts) + [len(ns)]:
            pieces.append(dict(nodes=ns[start:k], key=key, cut=cut))
            start, cut = k, cuts.get(k)
    return pieces


def anchors_from_history(H, t0, q, max_age, fps):
    """窓の始まり t0 で決まっている ID（[t0 − max_age, t0] に見えたもの）の終わりの位置・速度・最後のコマ・系統の鍵。"""
    hist = defaultdict(list)
    for f in range(max(0, t0 - max_age), t0 + 1):
        for i, xy, key, nd in H.get(f, ()):
            hist[i].append(dict(f=f, xy=xy, key=key, app=nd["app"]))
    out = {}
    for i, ns in hist.items():
        p, v = fit_end(ns, q["fit_len"], True, fps)
        out[i] = dict(id=i, xy=p, v=v, f=ns[-1]["f"], key=ns[-1]["key"], n=len(ns), app=piece_app(ns, q["fit_len"], True))
    return out


def merge_duplicates(paths, q, fuse_dets):
    """時間の重なる経路で、ずっと近くにいて同じカメラをほとんど使わないものを 1 つにする（視点間の整合）。"""
    changed = True
    while changed:
        changed = False
        idx = [{nd["f"]: nd for nd in p["nodes"]} for p in paths]
        best = None
        for a in range(len(paths)):
            for b in range(a + 1, len(paths)):
                common = idx[a].keys() & idx[b].keys()
                if len(common) < q["merge_ov"]:
                    continue
                if paths[a]["anchor"] is not None and paths[b]["anchor"] is not None:
                    continue
                dist = np.mean([np.linalg.norm(idx[a][f]["xy"] - idx[b][f]["xy"]) for f in common])
                shared = np.mean([bool(idx[a][f]["cams"] & idx[b][f]["cams"]) for f in common])
                if dist < q["d_dup"] and shared <= q["shared_max"] and (best is None or dist < best[0]):
                    best = (dist, a, b)
        if best:
            _, a, b = best
            ia, ib = idx[a], idx[b]
            merged = {}
            for f in ia.keys() | ib.keys():
                if f in ia and f in ib:
                    na_, nb_ = ia[f], ib[f]
                    z, Rz = fuse_dets(np.array([na_["xy"], nb_["xy"]]), np.array([na_["R"], nb_["R"]]))
                    merged[f] = dict(na_, xy=z, R=Rz, cams=na_["cams"] | nb_["cams"], det=na_["det"] + nb_["det"])
                else:
                    merged[f] = ia.get(f) or ib.get(f)
            keep = a if (paths[a]["anchor"] is not None or (paths[b]["anchor"] is None and len(ia) >= len(ib))) else b
            paths[keep] = dict(nodes=[merged[f] for f in sorted(merged)], anchor=paths[keep]["anchor"])
            del paths[b if keep == a else a]
            changed = True
    return paths


class BackEnd:
    """[t − n, t] の小片を、決まった履歴の錨と最小費用の経路被覆でつなぐ（リンクごとに 費用 − link_gain を足す）。
    fe: 前の段の設定（probation・max_age はコマ、sigma_a・v_damp）。track_cls(id, key, x, P, hits, tsu, confirmed) は前の段の軌跡。"""

    def __init__(self, q, fe, fps, kf_mats, kf_update, fuse_dets, track_cls, id_start=1_000_000):
        self.q, self.fe, self.fps, self.dt = q, fe, fps, 1.0 / fps
        self.kf_mats, self.kf_update, self.fuse_dets, self.Track = kf_mats, kf_update, fuse_dets, track_cls
        self.next_id = id_start
        self.next_key = 10**9              # 後ろの段が作る軌跡の系統の鍵
        self.used_ids = set()
        self.owner = {}                    # ID → 最後にその ID を運んだ経路の系統の鍵（keep_label）

    def optimize(self, t, n, nodes_hist, H):
        q, fps, dt = self.q, self.fps, self.dt
        t0 = t - n - 1
        W = {f: nodes_hist[f] for f in range(max(0, t - n), t + 1) if f in nodes_hist}
        pieces = make_tracklets(W, q)
        anchors = anchors_from_history(H, t0, q, q["anchor_age"], fps) if t0 >= 0 else {}
        A = list(anchors.values())
        ends = [fit_end(pc["nodes"], q["fit_len"], True, fps) for pc in pieces]
        starts = [fit_end(pc["nodes"], q["fit_len"], False, fps) for pc in pieces]
        app_end = [piece_app(pc["nodes"], q["fit_len"], True) for pc in pieces]
        app_start = [piece_app(pc["nodes"], q["fit_len"], False) for pc in pieces]
        na, npc = len(A), len(pieces)
        nxt = {}
        for k in range(npc - 1):
            if pieces[k + 1]["key"] == pieces[k]["key"] and pieces[k + 1]["cut"] != "start":
                nxt[k] = k + 1
        BIG = 1e6
        cost = np.full((na + npc, npc), BIG)

        def edge(c, bonus):
            if bonus > 0:
                return min(c if c is not None else q["link_gain"], q["link_gain"]) - bonus - q["link_gain"]
            return BIG if c is None else c - q["link_gain"]
        for b, pb in enumerate(pieces):
            fb = pb["nodes"][0]["f"]
            for a, an in enumerate(A):
                if fb - an["f"] <= q["gap_max"]:
                    c = link_cost(an["xy"], an["v"], an["f"], starts[b][0], starts[b][1], fb, q, dt)
                    c = None if c is None else c + q["w_app"] * app_cost(an["app"], app_start[b], q["app"])
                    cost[a, b] = edge(c, q["lineage_bonus"] * (an["key"] == pb["key"] and pb["cut"] == "start"))
            for a, pa in enumerate(pieces):
                fa = pa["nodes"][-1]["f"]
                if fa < fb and fb - fa <= q["gap_max"]:
                    c = link_cost(ends[a][0], ends[a][1], fa, starts[b][0], starts[b][1], fb, q, dt)
                    c = None if c is None else c + q["w_app"] * app_cost(app_end[a], app_start[b], q["app"])
                    cost[na + a, b] = edge(c, q["gap_bonus"] * (nxt.get(a) == b and pb["cut"] == "gap"))
        succ, pred = {}, {}
        if npc:
            M = np.full((na + npc + npc, npc + na + npc), 0.0)
            M[:na + npc, :npc] = np.minimum(cost, BIG)
            M[:na + npc, npc:] = np.where(np.eye(na + npc), 0.0, BIG)
            M[na + npc:, :npc] = np.where(np.eye(npc), 0.0, BIG)
            M[na + npc:, npc:] = 0.0
            r, c = linear_sum_assignment(M)
            for i, j in zip(r, c):
                if i < na + npc and j < npc and M[i, j] < 0:
                    succ[i] = j
                    pred[j] = i
        paths = []
        for b in range(npc):
            if b in pred and pred[b] >= na:
                continue
            nodes = list(pieces[b]["nodes"])
            k = b
            while na + k in succ:
                k = succ[na + k]
                nodes += pieces[k]["nodes"]
            paths.append(dict(nodes=nodes, anchor=A[pred[b]] if b in pred else None))
        if q["merge"]:
            paths = merge_duplicates(paths, q, self.fuse_dets)
        # ID: (1) 錨につながった、(2) 重なるコマの前の解と節を多く共有する、(3) 前の段の多数の番号がまだ空いている、(4) 新しい番号
        taken = set()
        for pth in paths:
            if pth["anchor"] is not None:
                pth["id"] = pth["anchor"]["id"]
                taken.add(pth["id"])
        for pth in paths:
            if pth["anchor"] is None and len(pth["nodes"]) < q["min_len"] and pth["nodes"][-1]["f"] < t:
                pth["id"] = None          # 過去の短いはぐれ: 誤検出として捨てる
        prev = {(f, key): i for f in range(max(0, t - n), t + 1) for i, _, key, _ in H.get(f, ())}
        free = [pth for pth in paths if pth["anchor"] is None and "id" not in pth]
        cand = sorted({prev[(nd["f"], nd["key"])] for pth in free for nd in pth["nodes"] if (nd["f"], nd["key"]) in prev} - taken)
        if free and cand:
            votes = np.zeros((len(free), len(cand)))
            ci = {c: k for k, c in enumerate(cand)}
            for a, pth in enumerate(free):
                for nd in pth["nodes"]:
                    i = prev.get((nd["f"], nd["key"]))
                    if i in ci:
                        votes[a, ci[i]] += 1
            r, c = linear_sum_assignment(-votes)
            for a, k in zip(r, c):
                if votes[a, k] > 0:
                    free[a]["id"] = cand[k]
                    taken.add(cand[k])
        for pth in sorted(paths, key=lambda p: -len(p["nodes"])):
            if "id" in pth:
                continue
            labs = defaultdict(int)
            for nd in pth["nodes"]:
                labs[nd["lab"]] += 1
            u = max(labs, key=labs.get)
            own = q["keep_label"] and self.owner.get(u) in {nd["key"] for nd in pth["nodes"]}
            if u not in taken and (u not in self.used_ids or own):
                pth["id"] = u
            else:
                pth["id"] = self.next_id
                self.next_id += 1
            taken.add(pth["id"])
        paths = [pth for pth in paths if pth["id"] is not None]
        for pth in paths:
            self.used_ids.add(pth["id"])
            self.owner[pth["id"]] = pth["nodes"][-1]["key"]
        # 窓の各コマの訂正した出力と、t での状態
        frames, tracks, keys_taken = defaultdict(list), [], set()
        paths.sort(key=lambda p: -max(nd["f"] for nd in p["nodes"]))
        F, Qm = self.kf_mats(self.fe["sigma_a"], dt)
        for pth in paths:
            ns = sorted(pth["nodes"], key=lambda nd: nd["f"])
            an = pth["anchor"]
            confirmed = len(ns) + (an["n"] if an else 0) >= self.fe["probation"]
            if an is not None:
                x, P, fcur = np.r_[an["xy"], an["v"]], np.diag([0.05, 0.05, 0.25, 0.25]), an["f"]
            else:
                x = np.r_[ns[0]["xy"], 0.0, 0.0]
                P = np.zeros((4, 4))
                P[:2, :2] = ns[0]["R"]
                P[2:, 2:] = np.eye(2) * 0.25
                fcur = ns[0]["f"] - 1
            for nd in ns:
                for _ in range(nd["f"] - fcur):
                    x = F @ x
                    P = F @ P @ F.T + Qm
                fcur = nd["f"]
                x, P = self.kf_update(x, P, nd["xy"], nd["R"])
                if confirmed:     # 確かにならない経路は出さない（前の段と同じ）
                    frames[nd["f"]].append((pth["id"], x[:2].copy(), nd["key"], nd))
            hits = len(ns) + (an["n"] if an else 0)
            tsu = t - fcur
            if tsu <= self.fe["max_age"] and (hits >= self.fe["probation"] or tsu == 0):
                for _ in range(tsu):
                    x[2:] *= self.fe["v_damp"]
                    x = F @ x
                    P = F @ P @ F.T + Qm
                key = ns[-1]["key"]
                if key in keys_taken:
                    key = self.next_key
                    self.next_key += 1
                keys_taken.add(key)
                tracks.append(self.Track(pth["id"], key, x, P, hits=hits, tsu=tsu, confirmed=hits >= self.fe["probation"]))
        path_ids = {pth["id"] for pth in paths}
        for an in A:                       # 窓で続かなかった錨は、見失った軌跡として残す
            if an["id"] not in path_ids and t - an["f"] <= self.fe["max_age"]:
                key = an["key"]
                if key in keys_taken:
                    key = self.next_key
                    self.next_key += 1
                keys_taken.add(key)
                if q["lost_motion"]:
                    x, P = np.r_[an["xy"], an["v"]], np.diag([0.05, 0.05, 0.25, 0.25])
                    for _ in range(t - an["f"]):
                        x[2:] *= self.fe["v_damp"]
                        x = F @ x
                        P = F @ P @ F.T + Qm
                else:
                    x, P = np.r_[an["xy"], 0.0, 0.0], np.diag([0.1, 0.1, 0.25, 0.25]) + np.eye(4) * 0.01 * (t - an["f"])
                tracks.append(self.Track(an["id"], key, x, P, hits=an["n"], tsu=t - an["f"], confirmed=True))
        return dict(t=t, frames=dict(frames), tracks=tracks)
