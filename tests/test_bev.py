"""床の上の追跡（twinsim.bev）の確かめ（CPU だけ、数秒）: 投影・camera_groups の検査・評価・A。"""

import json
from pathlib import Path

import numpy as np
import pytest

from twinsim import api
from twinsim.__main__ import main
from twinsim.bev import detect, evaluate, groups, synthetic_tracks, track
from twinsim.bev.calib import project, projection
from twinsim.bev.io import gt_positions, load_dets, open_run
from twinsim.camera import Camera
from twinsim.scenario import load_scenario

GATE_AREA = [[-4.5, 13.0], [-1.5, 13.0], [-1.5, 21.0], [-4.5, 21.0]]
CONFIG = {
    "name": "bev_test", "space": "office", "seed": 1, "duration_s": 30, "fps": 10,
    "streams": [{"name": "arrivals", "origin": "entrance", "rate": {"type": "constant", "per_min": 20},
                 "destinations": [{"to": "south_exit", "via": ["gate"]}]}],
    # ゲートの廊下を前後から見る 4 台と、ロビーから見る 1 台（視野が重なる）
    "cameras": [{"name": "G1", "eye": [-4.3, 22.5, 2.8], "target": [-3.0, 15.5, 0.8], "focal_mm": 8.0},
                {"name": "G2", "eye": [-1.7, 22.5, 2.8], "target": [-3.0, 15.5, 0.8], "focal_mm": 8.0},
                {"name": "G3", "eye": [-4.3, 13.0, 2.8], "target": [-3.0, 17.5, 0.8], "focal_mm": 8.0},
                {"name": "G4", "eye": [-1.7, 13.0, 2.8], "target": [-3.0, 17.5, 0.8], "focal_mm": 8.0},
                {"name": "CamGate"}],
    "camera_groups": {"gate": {"cameras": ["G1", "G2", "G3", "G4", "CamGate"], "area": GATE_AREA}},
    "outputs": ["ground_truth", "frames", "tasks", "mot", "aicity"],
}


@pytest.fixture(scope="module")
def run_dir(tmp_path_factory):
    out = tmp_path_factory.mktemp("bev") / "run"
    api.run(CONFIG, out)
    return out


def test_camera_groups_in_ground_truth(run_dir):
    gt = json.loads((run_dir / "ground_truth.json").read_text())
    assert gt["camera_groups"] == CONFIG["camera_groups"]


@pytest.mark.parametrize("bad, msg", [
    ({"gate": {"cameras": ["G1", "X9"]}}, "X9 が cameras[] にない"),
    ({"gate": {"cameras": ["G1"], "zone": []}}, "知らないキー zone"),
    ({"gate": {"cameras": ["G1"], "area": [[0, 0], [1, 1]]}}, "3 点以上の多角形"),
    ({"../x": {"cameras": ["G1"]}}, "名前に使えるのは"),
])
def test_camera_groups_validation(bad, msg):
    with pytest.raises(ValueError, match=msg.replace("[", r"\[").replace("]", r"\]")):
        load_scenario({**CONFIG, "camera_groups": bad})


def test_projection_matches_twinsim_camera(run_dir):
    _, gt, _, _ = open_run(run_dir)
    rng = np.random.default_rng(0)
    for name, c in gt["cameras"].items():
        P = projection(c)
        pts = np.c_[rng.uniform(-5, 0, 50), rng.uniform(13, 21, 50), rng.uniform(0, 2, 50)]
        uv, depth = Camera(c).project(pts)
        ok = depth > 0.1
        assert np.abs(project(P, pts)[ok] - uv[ok]).max() < 1e-6, name


def test_evaluate_on_synthetic_tracks(run_dir):
    rules = {"vis_min": None, "portal_s": 0}
    synthetic_tracks(run_dir, label="perfect")
    m = evaluate(run_dir, "perfect", **rules)["gate"]["metrics"]["1m"]
    assert m["IDF1"] == 1 and m["MOTA"] == 1 and m["HOTA"] > 0.999 and m["IDSW"] == 0 and m["n_gt"] > 100

    synthetic_tracks(run_dir, label="miss", miss=0.2, seed=1)
    m = evaluate(run_dir, "miss", **rules)["gate"]["metrics"]["1m"]
    assert abs(m["Rec"] - 0.8) < 0.04 and m["Prec"] == 1 and m["IDSW"] == 0

    # every 5 s everybody gets a new ID: one switch for each person seen again after a boundary
    synthetic_tracks(run_dir, label="switch", switch_every_s=5.0)
    m = evaluate(run_dir, "switch", **rules)["gate"]["metrics"]["1m"]
    _, gt, f0, n = open_run(run_dir)
    group = groups(gt)["gate"]
    epochs = {}
    for f, items in gt_positions(gt, f0, n).items():
        for name, xy in items.items():
            if group.contains([xy])[0]:
                epochs.setdefault(name, set()).add(f // 50)
    assert m["IDSW"] == sum(len(e) - 1 for e in epochs.values()) > 0


def test_tracker_a_on_gt_boxes(run_dir):
    detect(run_dir, "gt_boxes")
    info = track(run_dir, "a", dets="gt_boxes")["gate"]
    assert info["ids"] > 0 and info["version"]["twinsim"]
    m = evaluate(run_dir, "a")["gate"]["metrics"]["1m"]
    assert m["IDF1"] >= 0.95 and m["MOTA"] >= 0.9, m
    for f in ("bev_tracks/a/gate/bev.csv", "bev_tracks/a/gate/mot/G1.txt", "bev_tracks/a/gate/aicity.txt", "bev_eval/a_gate.json", "bev_eval/summary.csv"):
        assert (run_dir / f).exists(), f


def test_cli(run_dir, capsys):
    assert main(["bev-detect", str(run_dir), "--name", "noisy", "--miss", "0.1", "--jitter-px", "2"]) == 0
    assert main(["bev-track", str(run_dir), "--dets", "noisy", "--label", "a_noisy", "--param", "max_age_s=1.5"]) == 0
    assert main(["bev-eval", str(run_dir), "--tracker", "a_noisy"]) == 0
    assert "IDF1" in capsys.readouterr().out
    assert json.loads((run_dir / "bev_tracks/a_noisy/gate/run.json").read_text())["params"]["max_age_s"] == 1.5
    assert main(["bev-track", str(run_dir), "--param", "no_such=1"]) == 1   # 知らない設定は誤り


def test_peoplenet_without_gpu(run_dir, tmp_path, monkeypatch):
    """peoplenet の流れ（設定を書く → docker → KITTI を読む → bev_dets/）を、docker の代わりに KITTI を書く関数で確かめる。"""
    import subprocess

    from twinsim.bev import deepstream

    models = tmp_path / "models"
    (models / deepstream.MODEL).mkdir(parents=True)
    (models / deepstream.MODEL / deepstream.ONNX).write_bytes(b"")
    (run_dir / "video").mkdir(exist_ok=True)
    cams = CONFIG["camera_groups"]["gate"]["cameras"]
    for c in cams:
        (run_dir / "video" / f"{c}.mp4").write_bytes(b"")
    calls, real_run = [], subprocess.run

    def fake_run(cmd, **kw):
        if cmd[0] != "docker":  # git など（version()）はそのまま
            return real_run(cmd, **kw)
        work = Path(cmd[cmd.index("-w") - 1].split(":")[0])
        cfg = (work / "config_deepstream.txt").read_text()
        calls.append((cmd, cfg))
        assert "batch-size=5" in cfg and "[tracker]\nenable=0" in cfg and "topk=100" in (work / "config_pgie.txt").read_text()
        # 源 1（G2）のコマ 3 に 1 つ、源 4（CamGate）のコマ 7 に 2 つ
        (work / "kitti" / "00_001_000003.txt").write_text("Person 0.0 0 0.0 10.5 20.25 30.125 80.0 0 0 0 0 0 0 0 0.75\n")
        (work / "kitti" / "00_004_000007.txt").write_text("Person 0.0 0 0.0 1 2 3 4 0 0 0 0 0 0 0 0.5\nPerson 0.0 0 0.0 5 6 7 8 0 0 0 0 0 0 0 0.25\n")
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr(deepstream.subprocess, "run", fake_run)
    with pytest.raises(ValueError, match="--gpu"):
        detect(run_dir, "peoplenet", models=str(models))
    out, info = detect(run_dir, "peoplenet", gpu=3, models=str(models))
    assert len(calls) == 1 and "device=3" in calls[0][0] and info["boxes"] == 3 and info["deepstream"]["gpu"] == 3
    D = load_dets(out, cams)
    assert D[3] == [("G2", (10.5, 20.25, 30.125, 80.0), 0.75)] and len(D[7]) == 2
    assert not (out / "_deepstream" / "00" / "kitti").exists() and (out / "_deepstream" / "00" / "config_pgie.txt").exists()
