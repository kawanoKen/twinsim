"""A: 集中型のオンラインの床の上の追跡（numpy と scipy だけ）。

1. 検出の枠ごとに床の位置を出す: 人を円柱（背 1.7 m・半径 0.3 m）とみなし、写した円柱の左右の中心・上・下が枠に合う床の点を
   ガウス・ニュートン法で求める（画面の端で切れた上下の辺は使わない）。誤差はカメラからの光線の向きに sig_ray、横に sig_perp。
2. 追跡（毎コマ）: 等速のカルマンフィルタ。カメラごとにハンガリアン法で 1 つの軌跡に 1 つの検出を割り当てる（費用は負の対数尤度、
   距離 gate_m と χ² gate_chi2 の門）。直前に更新された確かな軌跡が先、見失い中・仮の軌跡が後（MV3DT・DeepSORT と同じ段の順）。
   割り当てた検出は情報の重みで融合して更新する。
3. 生まれ: 余った検出を、別のカメラ同士で r_cluster 以内なら束ね、生きている軌跡の近く（r_suppress か門の中）でなく、
   2 台以上で見えたか確信度が conf_new 以上なら仮の軌跡にする。probation_s の間に見えたら確か。見失ってから max_age_s で消す。

外の実験（WILDTRACK・SCOUT・SALSA・twinsim の 2 つの場面）で使ったものと同じ計算。時間の設定は秒で持ち、動画の fps でコマに直す。
窓最適化・閉ループの後付けは段階 4 で足す。
"""

from collections import defaultdict

import numpy as np
from scipy.optimize import linear_sum_assignment

from .base import Tracker, TrackPoint, register

PERSON_H, PERSON_R = 1.7, 0.3     # 人の円柱（MV3DT の modelInfo と同じ）
BODY_CENTER_Z = 0.85              # 初めの値: 枠の中心を、円柱の高さの半分の面に戻す

DEFAULTS = dict(det_conf=0.5, sig_ray=0.2, sig_perp=0.1, sigma_a=1.0, gate_m=1.0, gate_chi2=20.0, conf_new=0.6,
                r_cluster=0.8, r_suppress=1.0, probation_s=0.67, max_age_s=3.0, v_damp=0.9, live_s=1 / 3)

_RING = np.c_[PERSON_R * np.cos(np.linspace(0, 2 * np.pi, 16, endpoint=False)), PERSON_R * np.sin(np.linspace(0, 2 * np.pi, 16, endpoint=False))]


def cylinder_box(P, xy, floor_z=0.0):
    """床の点 xy (N, 2) に立つ円柱の画像の枠 (N, 4)。"""
    pts = xy[:, None, :] + _RING[None]
    uv = []
    for z in (floor_z, floor_z + PERSON_H):
        h = np.concatenate([pts, np.full(pts.shape[:2] + (1,), z), np.ones(pts.shape[:2] + (1,))], 2) @ P.T
        uv.append(h[..., :2] / h[..., 2:3])
    uv = np.concatenate(uv, 1)
    return np.c_[uv[..., 0].min(1), uv[..., 1].min(1), uv[..., 0].max(1), uv[..., 1].max(1)]


def fit_cylinder(P, xy0, box, img_h, iters=5, floor_z=0.0):
    """写した円柱が枠（左右の中心・上・下。画面の端で切れた辺は使わない）に合う床の点。"""
    mask = np.c_[np.ones(len(box)), box[:, 1] > 2, box[:, 3] < img_h - 2].astype(float)

    def res(xy):
        cb = cylinder_box(P, xy, floor_z)
        return np.c_[(cb[:, 0] + cb[:, 2] - box[:, 0] - box[:, 2]) / 2, cb[:, 1] - box[:, 1], cb[:, 3] - box[:, 3]] * mask
    xy = xy0.copy()
    for _ in range(iters):
        r0 = res(xy)
        J = np.stack([(res(xy + e) - r0) / 1e-3 for e in (np.array([1e-3, 0]), np.array([0, 1e-3]))], 2)
        xy = xy - np.linalg.solve(np.einsum("nri,nrj->nij", J, J) + np.eye(2) * 1e-6, np.einsum("nri,nr->ni", J, r0)[..., None])[..., 0]
    return xy


def kf_mats(sigma_a, dt):
    F = np.eye(4)
    F[0, 2] = F[1, 3] = dt
    q = np.array([[dt**4 / 4, dt**3 / 2], [dt**3 / 2, dt**2]]) * sigma_a**2
    Q = np.zeros((4, 4))
    for i in range(2):
        Q[np.ix_([i, i + 2], [i, i + 2])] = q
    return F, Q


def kf_update(x, P, z, R):
    S = P[:2, :2] + R
    K = P[:, :2] @ np.linalg.inv(S)
    return x + K @ (z - x[:2]), P - K @ P[:2, :]


def fuse_dets(zs, Rs):
    """カメラごとの床の点を情報の重みで融合する（極限では光線の交点）。"""
    Ii = np.linalg.inv(Rs)
    Rf = np.linalg.inv(Ii.sum(0))
    return Rf @ np.einsum("kij,kj->i", Ii, zs), Rf


class _Track:
    __slots__ = ("id", "x", "P", "hits", "tsu", "confirmed", "n_views")

    def __init__(self, id, x, P):
        self.id, self.x, self.P, self.hits, self.tsu, self.confirmed, self.n_views = id, x, P, 1, 0, False, 0


@register
class ATracker(Tracker):
    name = "a"

    def __init__(self, calib, area, fps, params, floor_z=0.0):
        super().__init__(calib, area, fps, params, floor_z)
        unknown = set(self.params) - set(DEFAULTS)
        if unknown:
            raise ValueError(f"A の設定に知らないキー {', '.join(sorted(unknown))}（使えるのは {', '.join(sorted(DEFAULTS))}）")
        p = dict(DEFAULTS, **self.params)
        self.p = p
        fr = lambda s: max(1, int(round(s * fps)))  # 秒 → コマ
        self.probation, self.max_age, self.live = fr(p["probation_s"]), fr(p["max_age_s"]), fr(p["live_s"])
        self.F, self.Q = kf_mats(p["sigma_a"], 1.0 / fps)
        self.cams = list(calib)
        self.P = {c: np.asarray(calib[c]["P"], float) for c in self.cams}
        self.hinv, self.centre = {}, {}
        for c, P in self.P.items():
            self.hinv[c] = np.linalg.inv(np.c_[P[:, 0], P[:, 1], P[:, 2] * (self.floor_z + BODY_CENTER_Z) + P[:, 3]])
            self.centre[c] = -np.linalg.solve(P[:, :3], P[:, 3])[:2]
        self.tracks, self.next_id = [], 1

    # ---- 1. 検出 → 床の点 ----
    def prepare(self, detections):
        obs = [o for o in detections if o[2] >= self.p["det_conf"] and o[0] in self.P]
        cam = np.array([self.cams.index(o[0]) for o in obs], int)
        box = np.array([o[1] for o in obs], float).reshape(-1, 4)
        conf = np.array([o[2] for o in obs], float)
        xy = np.zeros((len(obs), 2))
        for k, o in enumerate(obs):
            b = o[1]
            X = self.hinv[o[0]] @ np.array([(b[0] + b[2]) / 2, (b[1] + b[3]) / 2, 1.0])
            xy[k] = X[:2] / X[2]
        for ci, c in enumerate(self.cams):
            m = cam == ci
            if m.any():
                xy[m] = fit_cylinder(self.P[c], xy[m], box[m], self.calib[c]["height"], floor_z=self.floor_z)
        u = np.array([(xy[k] - self.centre[o[0]]) / np.linalg.norm(xy[k] - self.centre[o[0]]) for k, o in enumerate(obs)]).reshape(-1, 2)
        keep = self.area.contains(xy) if len(xy) else np.zeros(0, bool)
        d = dict(cam=cam[keep], conf=conf[keep], xy=xy[keep], u=u[keep])
        uu = np.einsum("ki,kj->kij", d["u"], d["u"])
        d["R"] = self.p["sig_ray"] ** 2 * uu + self.p["sig_perp"] ** 2 * (np.eye(2)[None] - uu)
        return d

    # ---- 2・3. 追跡 ----
    def step(self, frame, detections):
        p, F, Q = self.p, self.F, self.Q
        d = self.prepare(detections)
        for t in self.tracks:
            if t.tsu > 0:
                t.x[2:] *= p["v_damp"]
            t.x = F @ t.x
            t.P = F @ t.P @ F.T + Q
        T = self.tracks
        assigned = defaultdict(list)
        used = np.zeros(len(d["conf"]), bool)
        if T and len(d["conf"]):
            X = np.array([t.x[:2] for t in T])
            Pp = np.array([t.P[:2, :2] for t in T])
            level = np.array([0 if (t.confirmed and t.tsu == 0) else 1 for t in T])
            for c in range(len(self.cams)):
                idx = np.flatnonzero(d["cam"] == c)
                if not len(idx):
                    continue
                diff = d["xy"][idx][None] - X[:, None]
                dist = np.linalg.norm(diff, axis=2)
                S = Pp[:, None] + d["R"][idx][None]
                m2 = np.einsum("tdi,tdij,tdj->td", diff, np.linalg.inv(S), diff)
                nll = 0.5 * m2 + 0.5 * np.log(np.linalg.det(S))
                ok = (dist < p["gate_m"]) & (m2 < p["gate_chi2"])
                free = np.ones(len(idx), bool)
                for lv in (0, 1):
                    ti = np.flatnonzero(level == lv)
                    if not len(ti) or not free.any():
                        continue
                    fi = np.flatnonzero(free)
                    cost = np.where(ok[np.ix_(ti, fi)], nll[np.ix_(ti, fi)], 1e6)
                    r, k = linear_sum_assignment(cost)
                    for i, j in zip(r, k):
                        if cost[i, j] < 1e5:
                            assigned[ti[i]].append(idx[fi[j]])
                            used[idx[fi[j]]] = True
                            free[fi[j]] = False
        for i, t in enumerate(T):
            js = assigned.get(i)
            if js:
                z, Rz = fuse_dets(d["xy"][js], d["R"][js])
                t.x, t.P = kf_update(t.x, t.P, z, Rz)
                t.hits += 1
                t.tsu = 0
                t.n_views = len(set(d["cam"][js].tolist()))
                if t.hits >= self.probation:
                    t.confirmed = True
            else:
                t.tsu += 1
                t.n_views = 0
        self._births(d, used, T)
        self.tracks = [t for t in self.tracks if (t.confirmed and t.tsu <= self.max_age) or (not t.confirmed and t.tsu == 0)]
        return [TrackPoint(frame, t.id, float(t.x[0]), float(t.x[1]), tuple(map(tuple, t.P[:2, :2].round(6).tolist())), t.n_views)
                for t in self.tracks if t.confirmed and t.tsu == 0]

    def _births(self, d, used, T):
        p = self.p
        U = np.flatnonzero(~used)
        if not len(U):
            return
        Zu, Ru = d["xy"][U], d["R"][U]
        diff = Zu[:, None] - Zu[None]
        m2 = np.einsum("abi,abij,abj->ab", diff, np.linalg.inv(Ru[:, None] + Ru[None]), diff)
        Du = np.where(m2 < p["gate_chi2"], np.linalg.norm(diff, axis=2), np.inf)
        clusters = [[k] for k in range(len(U))]
        owner = list(range(len(U)))
        pairs = sorted((Du[a, b], a, b) for a in range(len(U)) for b in range(a + 1, len(U))
                       if Du[a, b] < p["r_cluster"] and d["cam"][U[a]] != d["cam"][U[b]])
        for _, a, b in pairs:
            ca, cb = owner[a], owner[b]
            if ca == cb:
                continue
            A, B = clusters[ca], clusters[cb]
            if {d["cam"][U[k]] for k in A} & {d["cam"][U[k]] for k in B}:
                continue
            if Du[np.ix_(A, B)].max() >= p["r_cluster"]:
                continue
            clusters[ca] = A + B
            clusters[cb] = []
            for k in B:
                owner[k] = ca
        live = [t for t in T if t.tsu == 0 or (t.confirmed and t.tsu <= self.live)]
        for cl in clusters:
            if not cl:
                continue
            js = U[cl]
            z, Rz = fuse_dets(d["xy"][js], d["R"][js])
            if any(np.linalg.norm(z - t.x[:2]) < p["r_suppress"] or (z - t.x[:2]) @ np.linalg.solve(t.P[:2, :2] + Rz, z - t.x[:2]) < p["gate_chi2"]
                   for t in live):
                continue      # 生きている軌跡の門の中: 同じ人らしい（MV3DT の minIouDiff4NewTarget）
            if d["conf"][js].max() < p["conf_new"] and len(js) < 2:
                continue
            P0 = np.zeros((4, 4))
            P0[:2, :2] = Rz
            P0[2:, 2:] = np.eye(2) * 0.25
            t = _Track(self.next_id, np.r_[z, 0.0, 0.0], P0)
            t.n_views = len(set(d["cam"][js].tolist()))
            self.next_id += 1
            self.tracks.append(t)
