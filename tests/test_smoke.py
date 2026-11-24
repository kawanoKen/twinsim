"""動作確認（CPU だけ、数秒）: uv run --no-project --with numpy --with scipy --with pyyaml --with pillow --with pytest pytest -q tests"""

import json

import pytest

from twinsim import api
from twinsim.quality import check
from twinsim.scenario import TEMPLATES, load_scenario


def test_templates_validate():
    for p in TEMPLATES.glob("*.yaml"):
        load_scenario(p.stem)


def test_smoke_run(tmp_path):
    s = api.run("office_smoke", tmp_path / "run")
    assert s["people"] > 0
    gt = json.loads((tmp_path / "run/ground_truth.json").read_text())
    assert set(gt["tracks"]) == set(gt["people"])
    for f in ("frames.csv.gz", "tasks.json", "graph/graph.json", "mot/CamEV.txt"):
        assert (tmp_path / "run" / f).exists(), f
    q = check(tmp_path / "run")
    assert q["not_walkable_frac"] < 0.005
    assert q["closest_non_group_pair_m"]["p1"] > 0.05


def test_same_seed_same_result(tmp_path):
    a = api.run("office_smoke", tmp_path / "a")
    b = api.run("office_smoke", tmp_path / "b")
    ga = json.loads((tmp_path / "a/ground_truth.json").read_text())
    gb = json.loads((tmp_path / "b/ground_truth.json").read_text())
    assert ga["tracks"] == gb["tracks"] and a["people"] == b["people"]


GROUPS = {"ev": {"cameras": ["CamEV"], "area": [[-14.5, -7.0], [-9.5, -7.0], [-9.5, 2.0], [-14.5, 2.0]], "note": "EV 前"}}


def test_camera_groups_kept_in_ground_truth(tmp_path):
    """camera_groups はそのまま正解に残り、人の動きは変わらない。書かなければ {}。"""
    api.run({**load_scenario("office_smoke"), "camera_groups": GROUPS}, tmp_path / "g")
    api.run("office_smoke", tmp_path / "n")
    gg = json.loads((tmp_path / "g/ground_truth.json").read_text())
    gn = json.loads((tmp_path / "n/ground_truth.json").read_text())
    assert gg["camera_groups"] == GROUPS and gn["camera_groups"] == {}
    assert gg["tracks"] == gn["tracks"]


@pytest.mark.parametrize("bad, msg", [
    ({"gate": {"cameras": ["CamEV", "X9"]}}, "X9 が cameras[] にない"),
    ({"gate": {"cameras": ["CamEV", "CamEV"]}}, "2 回ある"),
    ({"gate": {"cameras": []}}, "1 台以上"),
    ({"gate": {"cameras": ["CamEV"], "zone": []}}, "知らないキー zone"),
    ({"gate": {"cameras": ["CamEV"], "area": [[0, 0], [1, 1]]}}, "3 点以上の多角形"),
    ({"../x": {"cameras": ["CamEV"]}}, "名前に使えるのは"),
    (["CamEV"], "camera_groups は"),
])
def test_camera_groups_validation(bad, msg):
    with pytest.raises(ValueError, match=msg.replace("[", r"\[").replace("]", r"\]")):
        load_scenario({**load_scenario("office_smoke"), "camera_groups": bad})
