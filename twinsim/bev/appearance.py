"""検出の見え方: 胴の HSV の色のヒストグラム（8 × 4 × 4）を動画から作り、bev_dets/<検出器>/app_<カメラ>.npy に置く（A の窓最適化が使う）。

胴 = 枠の横の中央 50%・縦の上から 15〜55%。外の実験（cltrack.py の _camera_histograms）と同じ計算で、OpenCV が要る
（任意の依存: `uv sync --extra bev`。動画の読み込みと HSV への変換も OpenCV で行い、外の実験と同じ値にする）。
行は bev_dets/<検出器>/<カメラ>.txt の行の順。float16 で置く（外の実験の置き方と同じ）。
"""

import json

import numpy as np

HIST_BINS = (8, 4, 4)


def _cv2():
    try:
        import cv2
    except ImportError as e:
        raise ImportError("見え方（胴の色）には OpenCV が要る: uv sync --extra bev（pip なら opencv-python-headless）。"
                          "使わないなら A の設定を be_w_app=0 にする") from e
    return cv2


def histograms(video, rows, width, height):
    """rows: [(コマ, x1, y1, x2, y2), ...]（コマの順）→ (N, 128) の float16。"""
    cv2 = _cv2()
    out = np.zeros((len(rows), int(np.prod(HIST_BINS))), np.float32)
    by = {}
    for k, r in enumerate(rows):
        by.setdefault(int(r[0]), []).append(k)
    if not by:
        return out.astype(np.float16)
    cap = cv2.VideoCapture(str(video))
    last = max(by)
    for f in range(last + 1):
        ok, img = cap.read()
        if not ok:
            break
        if f not in by:
            continue
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        for k in by[f]:
            _, x1, y1, x2, y2 = rows[k]
            w, h = x2 - x1, y2 - y1
            xa, xb = int(max(0, x1 + 0.25 * w)), int(min(width, x2 - 0.25 * w))
            ya, yb = int(max(0, y1 + 0.15 * h)), int(min(height, y1 + 0.55 * h))
            if xb - xa < 2 or yb - ya < 2:
                continue
            hist = cv2.calcHist([hsv[ya:yb, xa:xb]], [0, 1, 2], None, list(HIST_BINS), [0, 180, 0, 256, 0, 256])
            out[k] = hist.ravel() / max(hist.sum(), 1)
    cap.release()
    return out.astype(np.float16)


def ensure(run, gt, det_dir, cameras):
    """足りないカメラの app_<カメラ>.npy を作る（検出の CSV が新しければ作り直す）。作ったカメラの一覧を返す。"""
    import csv
    from concurrent.futures import ProcessPoolExecutor
    todo = []
    for c in cameras:
        npy, txt = det_dir / f"app_{c}.npy", det_dir / f"{c}.txt"
        if not npy.exists() or npy.stat().st_mtime < txt.stat().st_mtime:
            todo.append(c)
    if not todo:
        return []
    _cv2()
    jobs = []
    for c in todo:
        with open(det_dir / f"{c}.txt", newline="") as fh:
            rows = [(int(r["frame"]), float(r["x1"]), float(r["y1"]), float(r["x2"]), float(r["y2"])) for r in csv.DictReader(fh)]
        video = run / "video" / f"{c}.mp4"
        if not video.exists():
            raise FileNotFoundError(f"見え方には動画が要る: {video}（描画していなければ A の設定を be_w_app=0 にする）")
        jobs.append((video, rows, gt["cameras"][c]["width"], gt["cameras"][c]["height"]))
    with ProcessPoolExecutor(min(len(jobs), 8)) as ex:
        for c, h in zip(todo, ex.map(histograms, *zip(*jobs))):
            np.save(det_dir / f"app_{c}.npy", h)
    (det_dir / "app.json").write_text(json.dumps({"bins_hsv": HIST_BINS, "torso": "x 25-75 %, y 15-55 %", "source": "video/<camera>.mp4"}))
    return todo
