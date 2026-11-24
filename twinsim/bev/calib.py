"""正解のカメラ（ground_truth.json の cameras: 4×4 の行列と画角）→ 3×4 の投影行列（世界の m → 画素）。

twinsim のカメラは USD と同じ表し方（-Z を見て +Y が上、行ベクトル p_cam = p_world · M）。画素は
u = cx + fx·x_cam / (−z_cam)、v = cy − fy·y_cam / (−z_cam) なので、P = [[fx, 0, −cx], [0, −fy, −cy], [0, 0, −1]] · Mᵀ[:3]。
"""

import json
from pathlib import Path

import numpy as np


def projection(cam):
    M = np.asarray(cam["world_to_camera_rowmajor"], float)
    K = np.array([[cam["fx_px"], 0.0, -cam["cx_px"]], [0.0, -cam["fy_px"], -cam["cy_px"]], [0.0, 0.0, -1.0]])
    return K @ M.T[:3]


def project(P, pts):
    """世界の点 (N, 3) → 画素 (N, 2)。"""
    h = np.c_[np.asarray(pts, float), np.ones(len(pts))] @ P.T
    return h[:, :2] / h[:, 2:3]


def calib(gt, cameras=None):
    """{カメラ: {P, K, width, height, fps, eye}}（cameras を省くと全部）。"""
    out = {}
    for name in cameras or list(gt["cameras"]):
        c = gt["cameras"][name]
        out[name] = {"P": projection(c), "K": [[c["fx_px"], 0.0, c["cx_px"]], [0.0, c["fy_px"], c["cy_px"]], [0.0, 0.0, 1.0]],
                     "width": int(c["width"]), "height": int(c["height"]), "fps": c.get("fps", gt["fps"]), "eye": list(c["eye"])}
    return out


def write_calib(run, gt, cameras=None):
    """<結果>/calib/<カメラ>.json を書く（外の追跡器に渡す形）。"""
    folder = Path(run) / "calib"
    folder.mkdir(parents=True, exist_ok=True)
    cal = calib(gt, cameras)
    for name, c in cal.items():
        (folder / f"{name}.json").write_text(json.dumps({**c, "P": c["P"].tolist()}, indent=1))
    return folder
