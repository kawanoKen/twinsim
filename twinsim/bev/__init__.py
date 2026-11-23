"""床の上（BEV）の複数カメラの追跡（視野の重なるカメラの群ごと）。手順と形は docs/bev-tracking.md。

  calib      正解のカメラ → 3×4 の投影行列（calib/<カメラ>.json）
  groups     カメラの群（シナリオの camera_groups）と評価領域
  detect     検出: 正解の枠から（gt_boxes）・外の検出の読み込み（external）→ bev_dets/<検出器>/
  trackers   追跡器（a: 集中型のオンライン追跡）と入口 track → bev_tracks/<追跡器>/<群>/
  evaluate   床の上の指標（IDF1・HOTA・MOTA・MODA など）→ bev_eval/
  synthetic  確かめ用の追跡（正解から作る）
  io         共通の読み書き
"""

from .calib import calib, projection, write_calib
from .detect import detect
from .evaluate import evaluate
from .groups import Group, groups
from .io import load_bev, load_dets, open_run
from .synthetic import synthetic_tracks
from .trackers import presets, track

__all__ = ["Group", "calib", "detect", "evaluate", "groups", "load_bev", "load_dets", "open_run", "presets", "projection", "synthetic_tracks", "track",
           "write_calib"]
