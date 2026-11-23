"""カメラの群（シナリオの camera_groups。ground_truth.json に残る）と評価領域。

群は視野の重なるカメラの組。床の上の追跡と評価は群ごとに行う。camera_groups がなければ、全部のカメラを 1 つの群 all とする。
area を書かなければ、評価領域は、群のカメラのどれかから床が見える歩ける所（twinsim の近似の見え方: camera.py の sees_floor）。
"""

import numpy as np

from ..camera import Camera
from ..space import load_space


def points_in_poly(poly, xy):
    """床の点 (N, 2) が多角形の中か（space.point_in_poly を並べて計算する形）。"""
    inside = np.zeros(len(xy), bool)
    x, y = xy[:, 0], xy[:, 1]
    for (x1, y1), (x2, y2) in zip(poly, np.roll(poly, -1, axis=0)):
        if y1 == y2:
            continue
        cross = ((y1 > y) != (y2 > y)) & (x < (x2 - x1) * (y - y1) / (y2 - y1) + x1)
        inside ^= cross
    return inside


class Group:
    def __init__(self, name, cameras, area=None, mask=None, note=None):
        self.name, self.cameras, self.area, self.note = name, list(cameras), area, note
        self._poly = np.asarray(area, float) if area is not None else None
        self._mask = mask  # (格子, x0, y0, res)

    def contains(self, xy):
        """床の点 (N, 2) が評価領域の中か（bool の配列）。"""
        xy = np.asarray(xy, float).reshape(-1, 2)
        if self._poly is not None:
            return points_in_poly(self._poly, xy)
        grid, x0, y0, res = self._mask
        i = np.floor((xy[:, 1] - y0) / res + 0.5).astype(int)
        j = np.floor((xy[:, 0] - x0) / res + 0.5).astype(int)
        ok = (i >= 0) & (i < grid.shape[0]) & (j >= 0) & (j < grid.shape[1])
        out = np.zeros(len(xy), bool)
        out[ok] = grid[i[ok], j[ok]]
        return out

    def bounds(self):
        """評価領域を囲む四角（x0, x1, y0, y1）。"""
        if self._poly is not None:
            a = self._poly
            return float(a[:, 0].min()), float(a[:, 0].max()), float(a[:, 1].min()), float(a[:, 1].max())
        grid, x0, y0, res = self._mask
        ii, jj = np.nonzero(grid)
        return x0 + jj.min() * res, x0 + jj.max() * res, y0 + ii.min() * res, y0 + ii.max() * res

    def to_json(self):
        return {"cameras": self.cameras, "area": self.area if self.area is not None else "seen_floor", "note": self.note}


def seen_floor_mask(gt, cameras, space=None):
    """群のカメラのどれかから床が見える歩ける格子。"""
    sp = space or load_space(gt["space"])
    cams = [Camera(gt["cameras"][c], sp.z) for c in cameras]
    mask = np.zeros_like(sp.grid, bool)
    for i, j in zip(*np.nonzero(sp.grid)):
        x, y = sp.center(i, j)
        mask[i, j] = any(c.sees_floor(sp, x, y) for c in cams)
    return mask, sp.x0, sp.y0, sp.res


def groups(gt, names=None, space=None):
    """{群の名前: Group}（names で絞る）。space は結果の空間（io.run_space。area を書かない群の評価領域に使う）。"""
    spec = gt.get("camera_groups") or {"all": {"cameras": list(gt["cameras"])}}
    if names:
        missing = [n for n in names if n not in spec]
        if missing:
            raise ValueError(f"camera_groups に {', '.join(missing)} がない（あるのは {', '.join(spec)}）")
        spec = {n: spec[n] for n in names}
    out = {}
    for n, g in spec.items():
        mask = None if g.get("area") is not None else seen_floor_mask(gt, g["cameras"], space)
        out[n] = Group(n, g["cameras"], g.get("area"), mask, g.get("note"))
    return out
