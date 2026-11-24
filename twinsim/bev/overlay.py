"""見て確かめる: 上面図と群のカメラを並べ、正解・予測・誤りの印を重ねた動画（bev_overlay/<追跡器>_<群>.mp4）。

上面図（左）: 評価領域（枠）、正解の人（白い丸。判定しない人は灰色）、予測（ID の色の点と番号）、対応した組（細い線）、
  見逃し（赤い丸）、誤検出（赤い ×）、ID の切り替わり（黄色の輪と文字、1 秒）。対応は評価と同じ（tau 以内、前のコマの対応を保つ）。
カメラ（右）: 正解の枠（緑。見えている割合 0.4 未満は灰色）と、追跡のカメラごとの枠（ID の色。A は床の位置から作った円柱の枠）。
動画の読み書きは ffmpeg、描くのは Pillow（twinsim の依存を増やさない）。
"""

import colorsys
import subprocess

import numpy as np
from PIL import Image, ImageDraw
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist

from .. import config
from .evaluate import VIS_MIN, crop, ground_truth, mot_stride, read_mot
from .groups import groups as load_groups
from .io import load_bev, open_run, run_space

TILE_W, TILE_H, MAP = 640, 360, 1080


def color(i):
    r, g, b = colorsys.hsv_to_rgb((int(i) * 0.61803) % 1.0, 0.75, 1.0)
    return int(r * 255), int(g * 255), int(b * 255)


def clip_box(bb, sx, sy, ox, oy, tw, th, inset=0):
    """画素の枠 → タイルの中に切った枠（外なら None）。"""
    a, b, e, h = bb
    a, b, e, h = max(a * sx, 0) + inset, max(b * sy, 0) + inset, min(e * sx, tw - 1) - inset, min(h * sy, th - 1) - inset
    return [ox + a, oy + b, ox + e, oy + h] if e - a > 1 and h - b > 1 else None


def frames_of(path, start, n, fps, w, h):
    """動画の start コマ目から n コマを (h, w, 3) の配列で順に返す（ffmpeg）。"""
    p = subprocess.Popen(["ffmpeg", "-loglevel", "error", "-ss", f"{start / fps:.4f}", "-i", str(path), "-frames:v", str(n),
                          "-vf", f"scale={w}:{h}", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"], stdout=subprocess.PIPE)
    size = w * h * 3
    for _ in range(n):
        buf = p.stdout.read(size)
        if len(buf) < size:
            yield None
            continue
        yield np.frombuffer(buf, np.uint8).reshape(h, w, 3)
    p.stdout.close()
    p.wait()


def overlay(run_dir, tracker="a", group_name=None, frames=None, tau=1.0, out=None):
    """bev_overlay/<tracker>_<群>.mp4 を書く。frames: 動画のコマ（range。既定は全部）。"""
    run, gt, f0, n_frames = open_run(run_dir)
    space = run_space(run, gt)
    gs = load_groups(gt, [group_name] if group_name else None, space)
    gname, group = next(iter(gs.items()))
    fps = gt["fps"]
    frames = list(frames if frames is not None else range(n_frames))
    folder = run / "bev_tracks" / tracker / gname
    pred = crop(load_bev(folder / "bev.csv"), group)
    keep, ign = ground_truth(run, gt, f0, n_frames, group)
    cams = group.cameras
    G = {c: read_mot(run / "mot" / f"{c}.txt", f0, n_frames, True) for c in cams}
    T = {c: read_mot(folder / "mot" / f"{c}.txt", f0, n_frames) for c in cams}
    stride = mot_stride(run)
    want = set(frames)
    # 上面図の座標
    x0, x1, y0, y1 = group.bounds()
    x0, x1, y0, y1 = x0 - 1, x1 + 1, y0 - 1, y1 + 1
    s = min((MAP - 60) / (x1 - x0), (MAP - 60) / (y1 - y0))
    mw = int((x1 - x0) * s) + 40

    def to_px(x, y):  # 北（+y）を上に
        return 20 + (x - x0) * s, 40 + (y1 - y) * s

    ncol = 2 if len(cams) > 1 else 1
    nrow = int(np.ceil(len(cams) / ncol))
    th = min(TILE_H, MAP // nrow)
    tw = th * 16 // 9
    W, H = (mw + ncol * tw) // 2 * 2, max(MAP, nrow * th) // 2 * 2
    out = out or run / "bev_overlay" / f"{tracker}_{gname}.mp4"
    out.parent.mkdir(parents=True, exist_ok=True)
    ff = subprocess.Popen(["ffmpeg", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(fps), "-i", "-",
                           "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "23", str(out)], stdin=subprocess.PIPE)
    readers = {c: frames_of(run / "video" / f"{c}.mp4", frames[0], frames[-1] - frames[0] + 1, fps, tw, th) for c in cams}
    sc = {c: (tw / gt["cameras"][c]["width"], th / gt["cameras"][c]["height"]) for c in cams}
    font, small = config.font(22), config.font(14)
    poly = [to_px(x, y) for x, y in group.area] if group.area is not None else None
    m, switched, tot = {}, {}, {"TP": 0, "FN": 0, "FP": 0, "IDSW": 0}
    for f in range(frames[0], frames[-1] + 1):
        imgs = {c: next(readers[c]) for c in cams}
        if f not in want:
            continue
        canvas = Image.new("RGB", (W, H), (24, 24, 24))
        d = ImageDraw.Draw(canvas)
        # ---- 上面図 ----
        if poly:
            d.polygon(poly, outline=(90, 90, 90))
        else:
            d.rectangle([*to_px(x0 + 1, y1 - 1), *to_px(x1 - 1, y0 + 1)], outline=(90, 90, 90))
        g, pr = keep.get(f, {}), pred.get(f, [])
        for x, y in ign.get(f, []):
            px, py = to_px(x, y)
            d.ellipse([px - 6, py - 6, px + 6, py + 6], outline=(110, 110, 110))
        gids, pids = list(g), [i for i, _ in pr]
        pairs = []
        if gids and pids:
            dm = cdist(np.array([g[k] for k in gids]), np.array([xy for _, xy in pr]))
            used_r, used_c = set(), set()
            pidx = {p: j for j, p in enumerate(pids)}
            for i, k in enumerate(gids):
                j = pidx.get(m.get(k))
                if j is not None and dm[i, j] <= tau and j not in used_c:
                    pairs.append((i, j))
                    used_r.add(i)
                    used_c.add(j)
            ri = [i for i in range(len(gids)) if i not in used_r]
            ci = [j for j in range(len(pids)) if j not in used_c]
            if ri and ci:
                sub = dm[np.ix_(ri, ci)]
                r, c = linear_sum_assignment(np.where(sub <= tau, sub, 1e6))
                for a, b in zip(r, c):
                    if sub[a, b] <= tau:
                        i, j = ri[a], ci[b]
                        if gids[i] in m and m[gids[i]] != pids[j]:
                            switched[gids[i]] = f
                            tot["IDSW"] += 1
                        pairs.append((i, j))
        mi, mj = {i for i, _ in pairs}, {j for _, j in pairs}
        for i, j in pairs:
            m[gids[i]] = pids[j]
            d.line([to_px(*g[gids[i]]), to_px(*pr[j][1])], fill=(150, 150, 150), width=1)
        for i, k in enumerate(gids):
            px, py = to_px(*g[k])
            if i in mi:
                d.ellipse([px - 7, py - 7, px + 7, py + 7], outline=(235, 235, 235), width=2)
            else:
                d.ellipse([px - 9, py - 9, px + 9, py + 9], outline=(255, 50, 50), width=3)
            if k in switched and f - switched[k] < fps:
                d.ellipse([px - 14, py - 14, px + 14, py + 14], outline=(255, 220, 0), width=3)
                d.text((px + 14, py - 22), "ID 切替", font=small, fill=(255, 220, 0))
        for j, (i, (x, y)) in enumerate(pr):
            px, py = to_px(x, y)
            cc = color(i)
            if j in mj:
                d.ellipse([px - 4, py - 4, px + 4, py + 4], fill=cc)
            else:
                d.line([px - 7, py - 7, px + 7, py + 7], fill=(255, 50, 50), width=3)
                d.line([px - 7, py + 7, px + 7, py - 7], fill=(255, 50, 50), width=3)
            d.text((px + 6, py - 6), str(i), font=small, fill=cc)
        tot["TP"] += len(pairs)
        tot["FN"] += len(gids) - len(pairs)
        tot["FP"] += len(pids) - len(pairs)
        d.text((10, 6), f"{(f + f0) / fps:7.1f} 秒  {tracker} / {gname}   見逃し {len(gids) - len(pairs)}  誤検出 {len(pids) - len(pairs)}", font=font,
               fill=(255, 255, 255))
        d.text((10, H - 46), f"累計  対応 {tot['TP']}  見逃し {tot['FN']}  誤検出 {tot['FP']}  ID 切替 {tot['IDSW']}", font=small, fill=(200, 200, 200))
        d.text((10, H - 24), "白 = 正解・赤丸 = 見逃し・赤 × = 誤検出・灰色 = 判定しない人", font=small, fill=(200, 200, 200))
        # ---- カメラ ----
        gf = f - ((f + f0) % stride)  # 正解の枠のある直前のコマ
        for k, c in enumerate(cams):
            ox, oy = mw + (k % ncol) * tw, (k // ncol) * th
            if imgs[c] is not None:
                canvas.paste(Image.fromarray(imgs[c]), (ox, oy))
            sx, sy = sc[c]

            for _, bb, vis in G[c].get(gf, []):
                if (r := clip_box(bb, sx, sy, ox, oy, tw, th)):
                    d.rectangle(r, outline=(60, 220, 60) if vis >= VIS_MIN else (140, 140, 140), width=1)
            for i, bb, _ in T[c].get(f, []):
                if (r := clip_box(bb, sx, sy, ox, oy, tw, th, 2)):
                    cc = color(i)
                    d.rectangle(r, outline=cc, width=2)
                    d.text((r[0] + 1, r[1] + 1), str(i), font=small, fill=cc)
            d.text((ox + 6, oy + 4), c, font=small, fill=(255, 255, 255))
        ff.stdin.write(np.asarray(canvas, np.uint8).tobytes())
    for r in readers.values():
        r.close()
    ff.stdin.close()
    ff.wait()
    return out
