"""床の上（BEV）の複数カメラの追跡（視野の重なるカメラの群ごと）。手順と形は docs/bev-tracking.md。

  calib       正解のカメラ → 3×4 の投影行列（calib/<カメラ>.json）
  groups      カメラの群（シナリオの camera_groups）と評価領域
  detect      検出: 正解の枠から（gt_boxes）・PeopleNet（peoplenet、DeepStream）・外の検出（external）→ bev_dets/<検出器>/
  deepstream  DeepStream のコンテナを動かす共通の部分
  trackers    追跡器（a: 集中型のオンライン追跡、mv3dt: 公式 MV3DT）と入口 track → bev_tracks/<追跡器>/<群>/
  evaluate    床の上の指標（IDF1・HOTA・MOTA・MODA など）・カメラごとの指標 → bev_eval/
  graph       ゾーンの人数・移動（graph/ と同じ形）→ bev_tracks/<追跡器>/<群>/graph/ と正解との比べ
  tasks       ゲートの通過・止まった区間・混み具合・EV の待ち時間（tasks.json と同じ形）→ bev_tracks/<追跡器>/<群>/tasks.json と正解との比べ
  overlay     上面図とカメラを並べた確かめの動画 → bev_overlay/
  synthetic   確かめ用の追跡（正解から作る）
  io          共通の読み書き
"""

from .calib import calib, projection, write_calib
from .detect import detect
from .evaluate import evaluate
from .groups import Group, groups
from .io import load_bev, load_dets, open_run
from .overlay import overlay
from .synthetic import synthetic_tracks
from .trackers import presets, track

__all__ = ["Group", "calib", "detect", "evaluate", "groups", "load_bev", "load_dets", "open_run", "overlay", "presets", "projection", "synthetic_tracks",
           "track", "write_calib"]
