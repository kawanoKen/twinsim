"""版の見本の動画を作り、GitHub の Releases に付ける（版を出すときの最後の手順。docs/architecture.md の「版の出し方」）。

    ./isaac_up.sh                                        # Isaac Sim を起動する（GPU。空きを確かめてから）
    uv run python tools/release_sample.py 1.2.0          # 計算 → 描画 → 正解の枠と版の番号を重ねる → Releases の v1.2.0 に付ける
    uv run python tools/release_sample.py 1.2.0 --no-upload   # 作るだけ
    isim down                                            # 終わったら GPU を返す

付けるのは twinsim-sample.mp4（同じ名前があれば差し替える）と、作り方の記録 twinsim-sample.json。
README のリンク https://github.com/fujisawa-yoshihiko/twinsim/releases/latest/download/twinsim-sample.mp4 は、
いつもいちばん新しい版に付けた見本を指す。

作るのは、今の作業ツリーのコード（twinsim/・pyproject.toml・uv.lock）がタグ vX.Y.Z と同じで、版の番号が X.Y.Z のときだけ
（タグのあとに足した文書や道具は構わない）。計算は CPU、描画は Isaac Sim（./isaac_up.sh が今の作業ツリーのコードを写す）。
結果の置き場所は --out（既定 <TWINSIM_DATA>/samples/vX.Y.Z。git の外）。
"""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import twinsim  # noqa: E402
from twinsim import config  # noqa: E402

TEMPLATE, SEED, CAMERA, SECONDS = "office_lunch", 1, "CamCafe", 20  # カフェの列・店員・ソファ（1.2 の部品）が映る
MIN_VISIBILITY = 0.4  # 重ねる枠（examples/05 と同じ）
NAME = "twinsim-sample"
CREDIT = "Space assets: NVIDIA Isaac Sim (CC BY 4.0)"  # NOTICE.md の表示（素材から作った映像を配るとき）


def git(*args):
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True)


def check_code(version):
    """今のコードがタグ vX.Y.Z と同じか。違えば理由を返す。"""
    tag = f"v{version}"
    if twinsim.__version__ != version:
        return f"版の番号が {twinsim.__version__}（{version} ではない）"
    if git("rev-parse", "--verify", "--quiet", f"{tag}^{{commit}}").returncode:
        return f"タグ {tag} がない"
    paths = ["twinsim", "pyproject.toml", "uv.lock"]
    if git("diff", "--quiet", tag, "--", *paths).returncode:
        return f"コードがタグ {tag} と違う（git diff {tag} -- {' '.join(paths)}）"
    if git("ls-files", "--others", "--exclude-standard", "--", *paths).stdout.strip():
        return "twinsim/ に git にないファイルがある"
    return None


def busiest(result, camera, seconds):
    """枠（見えている割合 MIN_VISIBILITY 以上）がいちばん多い区間 (始まり, 終わり)。シミュレータのコマで、端を含む（examples/04 と同じ）。"""
    gt = result.ground_truth()
    n = int(seconds * gt["fps"])
    labels = result.mot(camera)
    labels = labels[labels[:, 8] >= MIN_VISIBILITY]
    count = np.zeros(gt["total_frames"])
    np.add.at(count, labels[:, 0].astype(int) - 1, 1)  # MOT の frame は 1 始まり
    start = int(np.convolve(count, np.ones(n), "valid").argmax())
    return start, start + n - 1


def font(size):
    try:
        return ImageFont.load_default(size=size)
    except TypeError:  # Pillow 10.1 より前
        return ImageFont.load_default()


def overlay(result, camera, frames, version, out):
    """描いた動画に正解の枠（mot/）と版の番号・素材の表示を重ねて mp4 にする。"""
    gt = result.ground_truth()
    fps, w, h = gt["fps"], gt["cameras"][camera]["width"], gt["cameras"][camera]["height"]
    labels = result.mot(camera)
    labels = labels[labels[:, 8] >= MIN_VISIBILITY]
    head, small = font(20), font(16)
    top = f"twinsim {version}  |  {TEMPLATE}, seed {SEED}, {camera}  |  yellow boxes: ground truth (mot/)"
    cmd = ["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}", "-r", str(fps), "-i", "-",
           "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "23", "-movflags", "+faststart", str(out)]
    n = 0
    with subprocess.Popen(cmd, stdin=subprocess.PIPE) as proc:
        for frame, image in result.video_frames(camera, frames=frames):
            canvas = Image.fromarray(image)
            draw = ImageDraw.Draw(canvas, "RGBA")
            for _, person, x, y, bw, bh, *_ in labels[labels[:, 0] == frame + 1]:  # MOT の frame = シミュレータのコマ ＋ 1
                draw.rectangle([x, y, x + bw, y + bh], outline=(255, 220, 0), width=2)
                draw.text((x + 3, y + 2), str(int(person)), fill=(255, 220, 0), font=small)
            draw.rectangle([0, 0, w, 34], fill=(0, 0, 0, 150))
            draw.text((10, 6), f"{top}  |  t = {frame / fps:.1f} s", fill=(255, 255, 255), font=head)
            tw = draw.textlength(CREDIT, font=small)
            draw.rectangle([w - tw - 20, h - 28, w, h], fill=(0, 0, 0, 150))
            draw.text((w - tw - 10, h - 24), CREDIT, fill=(255, 255, 255), font=small)
            proc.stdin.write(canvas.tobytes())
            n += 1
        proc.stdin.close()
    if proc.returncode:
        raise RuntimeError(f"ffmpeg が失敗した（{proc.returncode}）")
    return n


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("version", help="版の番号 X.Y.Z（タグ vX.Y.Z）")
    ap.add_argument("--out", type=Path, help="置き場所（既定 <TWINSIM_DATA>/samples/vX.Y.Z）")
    ap.add_argument("--no-upload", action="store_true", help="Releases に付けない（作るだけ）")
    a = ap.parse_args()
    why = check_code(a.version)
    if why:
        sys.exit(f"作らない: {why}。見本はタグの版のコードで作る")
    tag = f"v{a.version}"
    if not a.no_upload and subprocess.run(["gh", "release", "view", tag], cwd=ROOT, capture_output=True).returncode:
        sys.exit(f"Releases に {tag} がない（先に gh release create。docs/architecture.md の「版の出し方」）")
    out = (a.out or config.DATA / "samples" / tag).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.monotonic()
    result = twinsim.simulate(TEMPLATE, out / "run", overrides={"seed": SEED})
    frames = busiest(result, CAMERA, SECONDS)
    start_s = frames[0] / result.ground_truth()["fps"]
    print(f"計算 {time.monotonic() - t0:.0f} 秒。描くコマ {frames[0]}〜{frames[1]}（{start_s:.1f} 秒から {SECONDS} 秒）", flush=True)
    done = twinsim.render(result.out_dir, frames=frames, cameras=[CAMERA], preroll=3)  # Isaac Sim が動いていなければ RuntimeError
    print(f"描画 {done['seconds']} 秒", flush=True)
    mp4 = out / f"{NAME}.mp4"
    n = overlay(result, CAMERA, frames, a.version, mp4)
    commit = git("rev-parse", f"{tag}^{{commit}}").stdout.strip()
    record = {"version": a.version, "tag": tag, "commit": commit, "template": TEMPLATE, "seed": SEED, "camera": CAMERA,
              "frames": list(frames), "fps": result.ground_truth()["fps"], "video_frames": n, "people": len(result.ground_truth()["people"]),
              "boxes": f"mot/{CAMERA}.txt, visibility >= {MIN_VISIBILITY}", "credit": CREDIT, "made": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
              "how": f"uv run python tools/release_sample.py {a.version}"}
    (out / f"{NAME}.json").write_text(json.dumps(record, ensure_ascii=False, indent=1) + "\n")
    print(f"{mp4}（{mp4.stat().st_size / 1e6:.1f} MB、{n} コマ）")
    if not a.no_upload:
        subprocess.run(["gh", "release", "upload", tag, str(mp4), str(out / f"{NAME}.json"), "--clobber"], cwd=ROOT, check=True)
        print(f"Releases の {tag} に付けた")


if __name__ == "__main__":
    main()
