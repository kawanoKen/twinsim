"""PeopleNet の検出を DeepStream（コンテナ）で出す（detect.py の peoplenet）。

群ごとに 1 回、群のカメラの動画（video/<カメラ>.mp4）をまとめて DeepStream の deepstream-app に通し、検出器の出力
（KITTI の形）だけを書き出して、bev_dets/<名前>/<カメラ>.txt に直す。追跡器は動かさない。
検出の設定は外の実験（MV3DT と A を比べた実験）と同じ: PeopleNet v2.6.3（ResNet-34、FP16）、人だけ、cluster-mode 2（NMS）・上位 100 件・
NMS の IoU 0.5、入力は動画の画素の大きさのまま（streammux の幅・高さ = 動画）。

  TWINSIM_DEEPSTREAM_IMAGE   コンテナ（既定 ghcr.io/nvidia/deepstream:9.2-triton-multiarch）
  TWINSIM_DEEPSTREAM_MODELS  モデルの置き場所。PeopleNet2.6.3/{resnet34_peoplenet.onnx, detector_labels.txt} がある所
                             （TensorRT のエンジンは初めの 1 回に作られ、同じ所に置かれる。書ける場所にする）
"""

import os
import shutil
import subprocess
import time
from collections import defaultdict
from pathlib import Path

from .. import config

PGIE = {"cluster-mode": 2, "topk": 100, "nms-iou-threshold": 0.5}  # 外の実験の pn_nms100_dump と同じ
MODEL = "PeopleNet2.6.3"
ONNX = "resnet34_peoplenet.onnx"


def pgie_config(settings):
    """PeopleNet の nvinfer の設定。settings の cluster-mode は [property]、ほかは [class-attrs-all] に書く（書かないものは DeepStream の既定）。"""
    s = dict(settings or {})
    prop = f"cluster-mode={s.pop('cluster-mode')}\n" if "cluster-mode" in s else ""
    attrs = "".join(f"{k}={v}\n" for k, v in s.items())
    return (f"[property]\n{prop}net-scale-factor=0.0039215697906911373\noffsets=0.0;0.0;0.0\n"
            f"labelfile-path=/workspace/models/{MODEL}/detector_labels.txt\nonnx-file=/workspace/models/{MODEL}/{ONNX}\n"
            "gie-unique-id=1\nnetwork-type=0\nnetwork-mode=2\nnum-detected-classes=3\nfilter-out-class-ids=1;2\n"
            "infer-dims=3;544;960\nmodel-color-format=0\nmaintain-aspect-ratio=0\noutput-tensor-meta=0\n"
            + (f"\n[class-attrs-all]\n{attrs}" if attrs else ""))


def app_config(cameras, width, height, tracker=None):
    """deepstream-app（-test5-app）の設定。tracker = 追跡器の設定ファイルの名前なら追跡器を動かし、その出力を tracker-kitti/ に書く。"""
    n = len(cameras)
    src = "".join(f"[source{k}]\nenable=1\ntype=3\nnum-sources=1\ngpu-id=0\ncudadec-memtype=0\nuri=file:///workspace/videos/{c}.mp4\n\n"
                  for k, c in enumerate(cameras))
    trk = ("[tracker]\nenable=0\n\n" if tracker is None else
           f"[tracker]\nenable=1\ngpu-id=0\ntracker-width={width}\ntracker-height={height}\ndisplay-tracking-id=1\n"
           f"ll-lib-file=/opt/nvidia/deepstream/deepstream/lib/libnvds_nvmultiobjecttracker.so\nll-config-file={tracker}\n\n")
    return ("[application]\nenable-perf-measurement=1\nperf-measurement-interval-sec=5\ngie-kitti-output-dir=kitti\n"
            + ("kitti-track-output-dir=tracker-kitti\n" if tracker else "") + "\n" + src
            + f"[streammux]\ngpu-id=0\nlive-source=0\nbatch-size={n}\nbatched-push-timeout=-1\nwidth={width}\nheight={height}\n"
              "enable-padding=0\nnvbuf-memory-type=0\n\n"
            + f"[primary-gie]\nenable=1\ngpu-id=0\nbatch-size={n}\ngie-unique-id=1\ninterval=0\nnvbuf-memory-type=0\n"
              f"model-engine-file=/workspace/models/{MODEL}/{ONNX}_b{n}_gpu0_fp16.engine\nconfig-file=config_pgie.txt\n\n"
            + trk + "[osd]\nenable=0\n\n[tiled-display]\nenable=0\n\n"
              "[sink0]\nenable=1\ntype=1\nsync=0\nsource-id=0\ngpu-id=0\nnvbuf-memory-type=0\n\n[tests]\nfile-loop=0\n")


def read_kitti(folder, cameras):
    """deepstream-app の KITTI の出力 <番号>_<源>_<コマ>.txt → {カメラ: [(コマ, x1, y1, x2, y2, 確信度)]}。"""
    out = defaultdict(list)
    for path in Path(folder).glob("*.txt"):
        *_, src, frame = path.stem.split("_")
        cam = cameras[int(src)]
        for line in path.read_text().splitlines():
            v = line.split()
            if len(v) >= 16:
                out[cam].append((int(frame), *map(float, v[4:8]), float(v[15])))
    return {c: sorted(out.get(c, [])) for c in cameras}


def resolve(image=None, models=None, need=(f"{MODEL}/{ONNX}",)):
    """(コンテナ, モデルの置き場所)。モデルがなければ誤り。"""
    image = image or os.environ.get("TWINSIM_DEEPSTREAM_IMAGE", "ghcr.io/nvidia/deepstream:9.2-triton-multiarch")
    models = Path(models or os.environ.get("TWINSIM_DEEPSTREAM_MODELS", config.DATA / "deepstream_models")).expanduser().resolve()
    for f in need:
        if not (models / f).exists():
            raise FileNotFoundError(f"モデルがない: {models / f}（TWINSIM_DEEPSTREAM_MODELS か --models で置き場所を指定）")
    return image, models


def check_inputs(run, gt, cams):
    """(動画のフォルダ, 幅, 高さ)。動画がない・画素の大きさがそろわないなら誤り。"""
    videos = (run / "video").resolve()
    missing = [c for c in cams if not (videos / f"{c}.mp4").exists()]
    if missing:
        raise FileNotFoundError(f"動画がない: {', '.join(missing)}（{videos}。先に描画する）")
    sizes = {(gt["cameras"][c]["width"], gt["cameras"][c]["height"]) for c in cams}
    if len(sizes) != 1:
        raise ValueError(f"1 回で通すカメラの画素の大きさがそろっていない: {sizes}")
    (w, h), = sizes
    return videos, w, h


def run_container(work, videos, models, image, gpu, name, app="deepstream-app", host_network=False):
    """work（設定を書いた所）で DeepStream を動かす。終わりの番号が 0 でなければ誤り。"""
    cmd = ["docker", "run", "--rm", "--name", name, "--gpus", f"device={gpu}", "--shm-size=4g", "--user", f"{os.getuid()}:{os.getgid()}",
           *(["--network", "host"] if host_network else []),
           "-v", f"{models}:/workspace/models", "-v", f"{videos}:/workspace/videos:ro", "-v", f"{Path(work).resolve()}:/workspace/work",
           "-w", "/workspace/work", image, app, "-c", "config_deepstream.txt"]
    with (Path(work) / "run.log").open("w") as fh:
        rc = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT).returncode
    if rc != 0:
        raise RuntimeError(f"DeepStream が止まった（終わりの番号 {rc}）。記録: {Path(work) / 'run.log'}")


def peoplenet(run, gt, chunks, out, gpu=None, settings=None, image=None, models=None, keep_raw=False):
    """chunks: [[カメラ, ...], ...]（1 回の DeepStream で通すカメラ。群ごと）→ ({カメラ: 行}, 実行の記録)。"""
    if gpu is None:
        raise ValueError("peoplenet は GPU を使う。--gpu で番号を決める（nvidia-smi で空きを見てから）")
    image, models = resolve(image, models)
    rows, log = {}, []
    for k, cams in enumerate(chunks):
        videos, w, h = check_inputs(run, gt, cams)
        work = out / "_deepstream" / f"{k:02d}"
        if work.exists():
            shutil.rmtree(work)
        (work / "kitti").mkdir(parents=True)
        (work / "config_deepstream.txt").write_text(app_config(cams, w, h))
        (work / "config_pgie.txt").write_text(pgie_config({**PGIE, **(settings or {})}))
        t0 = time.time()
        run_container(work, videos, models, image, gpu, f"twinsim-bev-detect-{os.getpid()}-{k}")
        got = read_kitti(work / "kitti", cams)
        rows.update(got)
        log.append({"cameras": cams, "width": w, "height": h, "elapsed_s": round(time.time() - t0, 1)})
        if not keep_raw:
            shutil.rmtree(work / "kitti")
    return rows, {"image": image, "models": str(models), "model": f"{MODEL}/{ONNX}", "pgie": {**PGIE, **(settings or {})}, "gpu": gpu, "runs": log}
