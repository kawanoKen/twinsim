"""床の上の追跡（twinsim.bev）の確かめ（CPU だけ、数秒）: 投影・camera_groups の検査・評価・A。"""

import json

import numpy as np
import pytest

from twinsim import api
from twinsim.__main__ import main
from twinsim.bev import detect, evaluate, groups, synthetic_tracks, track
from twinsim.bev.calib import project, projection
from twinsim.bev.io import gt_positions, open_run
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
