"""追跡器の約束と登録。

  Tracker(calib, area, fps, params, floor_z=0.0)
    calib   {カメラ: {P (3×4), width, height, eye, ...}}（calib.calib）。群のカメラだけ
    area    評価領域（groups.Group。contains(xy) で床の点が中か）
    fps     動画のコマの速さ
    params  追跡器の設定（presets/<名前>.json と、コマンドの --param）
    floor_z 床の高さ（空間の levels[0].z）
  step(frame, detections) -> [TrackPoint]   毎コマ。その時点で出せる結果（kind="online"）
  finish() -> [TrackPoint]                  終わりに、遅れて訂正した結果（kind="corrected"）。なければ空
  detections: [(カメラ, (x1, y1, x2, y2), 確信度), ...]

外の手法（MV3DT など。offline = True）は step を使わず、run_offline(run, gt, n_frames, work, gpu=...) でまとめて動かし、
(TrackPoint の並び, {コマ: [(カメラ, ID, 枠, 床の位置, 確信度)]}, 記録) を返す。検出は中で行う（bev_dets/ を使わない）。
"""

from dataclasses import dataclass


@dataclass
class TrackPoint:
    frame: int
    id: int
    x: float
    y: float
    cov: tuple | None = None   # 床の位置の誤差 ((xx, xy), (xy, yy))（m²）
    n_views: int = 0           # このコマで見えたカメラの数（0 は見えずに予測だけで出した点）
    kind: str = "online"       # online（そのコマで出せた）か corrected（後から訂正した）
    delay: int = 0             # 出せたのが何コマ後か


class Tracker:
    name = "base"
    offline = False
    default_preset = None  # presets/<名前>.json（--preset を書かないとき）

    def __init__(self, calib, area, fps, params, floor_z=0.0):
        self.calib, self.area, self.fps, self.params, self.floor_z = calib, area, fps, dict(params or {}), float(floor_z)

    def step(self, frame, detections):
        raise NotImplementedError

    def finish(self):
        return []


TRACKERS = {}


def register(cls):
    TRACKERS[cls.name] = cls
    return cls


def get(name):
    from . import a, mv3dt  # noqa: F401  登録のため
    if name not in TRACKERS:
        raise ValueError(f"追跡器 {name} はない（あるのは {', '.join(sorted(TRACKERS))}）")
    return TRACKERS[name]
