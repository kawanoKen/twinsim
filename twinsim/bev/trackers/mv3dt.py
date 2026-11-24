"""公式 MV3DT（NVIDIA DeepStream 9.2 の多視点 3D 追跡）の包み。外のアプリとしてまとめて動かす（offline、GPU）。

群ごとに、群のカメラの動画を deepstream-test5-app（PeopleNet → MV3DT）に通す。MV3DT はカメラごとに追跡して、
MQTT で ID と床の位置をカメラ間でそろえる分散型なので、実行のたびに、この計算だけの MQTT の受け口（mosquitto。127.0.0.1 の空いた番号）を
立てて終わりに止める。出力（カメラごとの枠・ID・床の足の位置）を、コマと ID ごとにカメラ間で平均して床の上の軌跡にする（外の実験と同じ）。

  TWINSIM_MOSQUITTO   mosquitto の実行ファイル（既定は PATH の mosquitto）。…/usr/sbin/mosquitto なら …/usr/lib/x86_64-linux-gnu を共有ライブラリの場所に足す
  TWINSIM_MQTT        すでに動いている受け口 host:port を使う（立てない）
  コンテナとモデルは deepstream.py と同じ（TWINSIM_DEEPSTREAM_IMAGE・TWINSIM_DEEPSTREAM_MODELS。BodyPose3DNet も要る）

設定（presets/<名前>.json の params）:
  pgie     PeopleNet の設定の上書き（{} なら DeepStream の既定。外の実験の MV3DT の値）
  tracker  presets/mv3dt_tracker.yml（公式の見本）の上書き {"節.キー": 値} か {"節": {キー: 値}}（節を丸ごと置き換える）
MV3DT は同じ設定でも実行ごとに結果がぶれる（カメラ間で ID がそろう割合が変わる）。track の repeats で何回か動かして比べる。
"""

import os
import re
import shutil
import socket
import subprocess
import time
import uuid
from collections import defaultdict
from pathlib import Path

import numpy as np

from .. import deepstream as ds
from ..calib import projection
from .base import Tracker, TrackPoint, register

BASE_YML = Path(__file__).resolve().parent.parent / "presets" / "mv3dt_tracker.yml"
NEED = (f"{ds.MODEL}/{ds.ONNX}", "BodyPose3DNet/bodypose3dnet_accuracy.onnx")


def set_yaml(text, section, key, value):
    """設定の節 section のキー key の値を value に（なければ節の終わりに足す）。"""
    out, inside, done = [], False, False
    for line in text.splitlines():
        top = line and not line.startswith((" ", "#", "-", "%")) and line.rstrip().endswith(":")
        if top:
            if inside and not done:
                out.append(f"  {key}: {value}")
                done = True
            inside = line.rstrip()[:-1] == section
        elif inside and re.match(rf"^  {re.escape(key)}:", line):
            out.append(f"  {key}: {value}")
            done = True
            continue
        out.append(line)
    if inside and not done:
        out.append(f"  {key}: {value}")
        done = True
    if not done:
        raise ValueError(f"MV3DT の設定に節 {section} がない")
    return "\n".join(out) + "\n"


def replace_section(text, section, keys):
    """節 section を {キー: 値} で丸ごと置き換える（書かないキーはライブラリの既定。公式の --config-overrides と同じ）。"""
    out, skip, found = [], False, False
    for line in text.splitlines():
        top = line and not line.startswith((" ", "#", "-", "%")) and line.rstrip().endswith(":")
        if top:
            skip = line.rstrip()[:-1] == section
            if skip:
                found = True
                out += [f"{section}:"] + [f"  {k}: {v}" for k, v in keys.items()]
                continue
        if not skip:
            out.append(line)
    if not found:
        raise ValueError(f"MV3DT の設定に節 {section} がない")
    return "\n".join(out) + "\n"


def tracker_config(cameras, overrides):
    text = BASE_YML.read_text()
    paths = "".join(f"    - /workspace/work/camInfo/{c}.yml\n" for c in cameras)
    text = re.sub(r"  cameraModelFilepath:\n(    - .*\n)+", "  cameraModelFilepath:\n" + paths, text)
    text = re.sub(r"pubSubInfoConfigPath: .*", "pubSubInfoConfigPath: /workspace/work/pub_sub_info_config.yml", text)
    text = re.sub(r"mqttProtoAdaptorConfigPath: .*", "mqttProtoAdaptorConfigPath: /workspace/work/config_mqtt.txt", text)
    for k, v in (overrides or {}).items():
        if isinstance(v, dict):
            text = replace_section(text, k, v)
        else:
            sec, key = k.split(".", 1)
            text = set_yaml(text, sec, key, v)
    return text


def cam_info(P):
    """MV3DT のカメラの形（人の模型 + 世界の m → 画素の 3×4 行列）。"""
    return ("modelInfo:\n- classID: 0\n  height: 1.7\n  radius: 0.3\nprojectionMatrix_3x4_w2p:\n"
            + "".join(f"- {float(x)!r}\n" for x in np.asarray(P, float).ravel()))


def pub_sub(n, broker, prefix):
    """カメラ k は /<prefix>/cam<k> に出し、ほかの全部のカメラのものを受ける。"""
    topic = [f"{broker};/{prefix}/cam{k}" for k in range(1, n + 1)]
    return ("pubBrokerTopicStr:\n" + "".join(f"- {t}\n" for t in topic) + "subPeerBrokerTopicStrs:\n"
            + "".join("- " + "\n  ".join(f"- {t}" for j, t in enumerate(topic) if j != i) + "\n" for i in range(n)))


class Broker:
    """この計算だけの MQTT の受け口（with で立てて止める）。TWINSIM_MQTT があればそれを使う。"""

    def __init__(self, work):
        self.work, self.proc = Path(work), None

    def __enter__(self):
        if os.environ.get("TWINSIM_MQTT"):
            self.address = os.environ["TWINSIM_MQTT"]
            return self
        exe = os.environ.get("TWINSIM_MOSQUITTO") or shutil.which("mosquitto")
        if not exe or not Path(exe).exists():
            raise FileNotFoundError("mosquitto がない（TWINSIM_MOSQUITTO で実行ファイルを指定するか、TWINSIM_MQTT で動いている受け口を指定）")
        with socket.socket() as s:
            s.bind(("127.0.0.1", 0))
            port = s.getsockname()[1]
        conf = self.work / "mosquitto.conf"
        conf.write_text(f"listener {port} 127.0.0.1\nallow_anonymous true\nset_tcp_nodelay true\npersistence false\n")
        env = dict(os.environ)
        lib = Path(exe).resolve().parents[1] / "lib" / "x86_64-linux-gnu"
        if lib.is_dir():
            env["LD_LIBRARY_PATH"] = f"{lib}:{env.get('LD_LIBRARY_PATH', '')}"
        self.log = (self.work / "mosquitto.log").open("w")
        self.proc = subprocess.Popen([exe, "-c", str(conf)], env=env, stdout=self.log, stderr=subprocess.STDOUT, start_new_session=True)
        for _ in range(50):
            with socket.socket() as s:
                if s.connect_ex(("127.0.0.1", port)) == 0:
                    break
            if self.proc.poll() is not None:
                raise RuntimeError(f"mosquitto が立たなかった（{self.work / 'mosquitto.log'}）")
            time.sleep(0.1)
        self.address = f"127.0.0.1:{port}"
        return self

    def __exit__(self, *exc):
        if self.proc is not None:
            self.proc.terminate()
            try:
                self.proc.wait(10)
            except subprocess.TimeoutExpired:
                self.proc.kill()
            self.log.close()


def read_tracks(folder, cameras):
    """MV3DT の KITTI の出力 → {コマ: [(カメラ, ID, (x1, y1, x2, y2), (x, y) 床の足の位置, 確信度)]}。"""
    out = defaultdict(list)
    for path in Path(folder).glob("*.txt"):
        *_, src, frame = path.stem.split("_")
        for line in path.read_text().splitlines():
            v = line.split()
            if len(v) >= 17:
                out[int(frame)].append((cameras[int(src)], int(v[1]), tuple(map(float, v[5:9])), (float(v[12]), float(v[13])), float(v[16])))
    return out


def to_bev(obs, n_frames):
    """コマと ID ごとに、カメラの床の位置を平均する（外の実験と同じ）。"""
    pts = []
    for f in sorted(obs):
        if f >= n_frames:
            continue
        by = defaultdict(list)
        for _, i, _, xy, _ in obs[f]:
            if np.isfinite(xy).all():
                by[i].append(xy)
        for i, v in sorted(by.items()):
            x, y = np.mean(v, 0)
            pts.append(TrackPoint(f, i, float(x), float(y), n_views=len(v)))
    return pts


@register
class MV3DTTracker(Tracker):
    name = "mv3dt"
    offline = True
    default_preset = "mv3dt"
    DEFAULTS = {"pgie": {}, "tracker": {}}

    def __init__(self, calib, area, fps, params, floor_z=0.0):
        # --param の pgie.topk=100・tracker.MultiViewAssociator.minPeerTrackletMatchScore=0.3 の形は、pgie・tracker の中へ入れる
        p = {k: dict(v) for k, v in self.DEFAULTS.items()}
        for k, v in (params or {}).items():
            head, _, rest = k.partition(".")
            if head not in self.DEFAULTS:
                raise ValueError(f"MV3DT の知らない設定: {k}（使えるのは pgie・tracker と、pgie.<キー>・tracker.<節>.<キー>）")
            if rest:
                p[head][rest] = v
            else:
                p[head].update(v or {})
        super().__init__(calib, area, fps, p, floor_z)

    def run_offline(self, run, gt, n_frames, work, gpu=None, image=None, models=None, keep_raw=False):
        """work に設定を書いて DeepStream を動かす → (TrackPoint の並び, {コマ: 観測}, 記録)。"""
        if gpu is None:
            raise ValueError("mv3dt は GPU を使う。--gpu で番号を決める（nvidia-smi で空きを見てから）")
        cams = list(self.calib)
        image, models = ds.resolve(image, models, NEED)
        videos, w, h = ds.check_inputs(run, gt, cams)
        work = Path(work)
        if work.exists():
            shutil.rmtree(work)
        (work / "camInfo").mkdir(parents=True)
        (work / "kitti").mkdir()
        (work / "tracker-kitti").mkdir()
        for c in cams:
            (work / "camInfo" / f"{c}.yml").write_text(cam_info(projection(gt["cameras"][c])))
        prefix = f"twinsim_{uuid.uuid4().hex[:12]}"
        t0 = time.time()
        with Broker(work) as b:
            (work / "config_deepstream.txt").write_text(ds.app_config(cams, w, h, tracker="config_tracker.yml"))
            (work / "config_pgie.txt").write_text(ds.pgie_config(self.params["pgie"]))
            (work / "config_tracker.yml").write_text(tracker_config(cams, self.params["tracker"]))
            (work / "pub_sub_info_config.yml").write_text(pub_sub(len(cams), b.address, prefix))
            (work / "config_mqtt.txt").write_text("[message-broker]\nusername = user\npassword = password\nset-threaded = 0\n")
            ds.run_container(work, videos, models, image, gpu, f"twinsim-bev-mv3dt-{prefix}", app="deepstream-test5-app", host_network=True)
        obs = read_tracks(work / "tracker-kitti", cams)
        if not keep_raw:
            shutil.rmtree(work / "kitti")
            shutil.rmtree(work / "tracker-kitti")
        info = {"image": image, "models": str(models), "gpu": gpu, "elapsed_s": round(time.time() - t0, 1), "broker": b.address}
        return to_bev(obs, n_frames), obs, info
