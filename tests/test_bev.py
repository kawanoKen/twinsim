"""床の上の追跡（twinsim.bev）の確かめ（CPU だけ、数秒）: 投影・群・評価・A。camera_groups の検査は test_smoke.py。"""

import json
from pathlib import Path

import numpy as np
import pytest

from twinsim import api
from twinsim.__main__ import main
from twinsim.bev import detect, evaluate, groups, synthetic_tracks, track
from twinsim.bev.calib import project, projection
from twinsim.bev.io import gt_positions, load_dets, open_run
from twinsim.bev.tasks import MATCH_S
from twinsim.camera import Camera

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
    for f in ("bev_tracks/a/gate/bev.csv", "bev_tracks/a/gate/mot/G1.txt", "bev_tracks/a/gate/aicity/tracks.txt", "bev_tracks/a/gate/tasks.json",
              "bev_tracks/a/gate/graph/zone_counts.csv", "bev_eval/a_gate.json", "bev_eval/summary.csv"):
        assert (run_dir / f).exists(), f
    # カメラの番号は正解の aicity/cameras.json と同じ
    assert json.loads((run_dir / "bev_tracks/a/gate/aicity/cameras.json").read_text()) == json.loads((run_dir / "aicity/cameras.json").read_text())


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


def test_mv3dt_without_gpu(run_dir, tmp_path, monkeypatch):
    """mv3dt の流れ（設定を書く → docker → KITTI を読む → bev.csv）を、docker の代わりに MV3DT の KITTI を書く関数で確かめる。
    --repeats の評価（平均と標準偏差）も。"""
    import subprocess

    from twinsim.bev.trackers import mv3dt

    models = tmp_path / "models"
    for f in mv3dt.NEED:
        (models / f).parent.mkdir(parents=True, exist_ok=True)
        (models / f).write_bytes(b"")
    (run_dir / "video").mkdir(exist_ok=True)
    cams = CONFIG["camera_groups"]["gate"]["cameras"]
    for c in cams:
        (run_dir / "video" / f"{c}.mp4").write_bytes(b"")
    monkeypatch.setenv("TWINSIM_MQTT", "127.0.0.1:1")  # 受け口は立てない
    _, gt, _, _ = open_run(run_dir)
    people = gt_positions(gt, *open_run(run_dir)[2:])
    real_run, seen = subprocess.run, []

    def fake_run(cmd, **kw):
        if cmd[0] != "docker":
            return real_run(cmd, **kw)
        work = Path(cmd[cmd.index("-w") - 1].split(":")[0])
        seen.append(cmd)
        assert cmd[-3] == "deepstream-test5-app" and "--network" in cmd
        trk = (work / "config_tracker.yml").read_text()
        assert "/workspace/work/camInfo/CamGate.yml" in trk and "maxPeerToPredDistance4Fusion: 4.0" in trk
        assert "127.0.0.1:1;/" in (work / "pub_sub_info_config.yml").read_text()
        assert "kitti-track-output-dir=tracker-kitti" in (work / "config_deepstream.txt").read_text()
        # 正解の位置をそのまま、2 台（源 0・1）から同じ ID で出す
        names = sorted({n for f in people for n in people[f]})
        for f in range(0, 50):
            for src in (0, 1):
                lines = [f"Person {names.index(n) + 1} 0 0 0 1 2 3 4 0 0 0 {x:.3f} {y:.3f} 0 0 0.9" for n, (x, y) in people.get(f, {}).items()]
                (work / "tracker-kitti" / f"00_{src:03d}_{f:06d}.txt").write_text("\n".join(lines))
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr(subprocess, "run", fake_run)
    with pytest.raises(ValueError, match="--gpu"):
        track(run_dir, "mv3dt", models=str(models))
    with pytest.raises(ValueError, match="知らない設定"):
        track(run_dir, "mv3dt", params={"nope": 1}, gpu=0, models=str(models))
    info = track(run_dir, "mv3dt", params={"tracker.MultiViewAssociator.maxPeerToPredDistance4Fusion": 4.0}, label="mvt",
                 repeats=2, gpu=0, models=str(models))
    assert len(seen) == 2 and set(info) == {"gate/mvt_r1", "gate/mvt_r2"} and info["gate/mvt_r1"]["dets"] == "internal"
    pts = (run_dir / "bev_tracks/mvt_r1/gate/bev.csv").read_text().splitlines()
    assert len(pts) > 1 and pts[1].split(",")[8] == "2"  # 2 台で見えた
    assert (run_dir / "bev_tracks/mvt_r1/gate/mot/G1.txt").read_text()
    res = evaluate(run_dir, "mvt", frames=range(0, 50), vis_min=None, portal_s=0)["gate"]
    assert res["repeats"] == ["mvt_r1", "mvt_r2"] and res["metrics"]["1m"]["IDF1"] == 1 and res["std"]["1m"]["IDF1"] == 0
    assert "IDF1_std" in (run_dir / "bev_eval/summary.csv").read_text().splitlines()[0]


def test_per_camera_and_graph(run_dir):
    """カメラごとの評価（正解の枠そのままなら 1）と graph/・tasks.json（正解から作った追跡なら人数の誤差 0、ゲートの通過は全部対応）。"""
    import shutil

    from twinsim.bev.evaluate import per_camera
    synthetic_tracks(run_dir, label="perfect_g")
    res = evaluate(run_dir, "perfect_g", vis_min=None, portal_s=0)["gate"]
    g = res["graph"]
    assert g["area_count"]["mae"] == 0 and all(z["mae"] == 0 for z in g["zones_compared"].values())
    gt_in = res["tasks"]["gates"]["gate:in"]  # 正解から作った追跡なら、ゲートの通過は全部対応する（推定は線を越えた時刻なので少し遅れる）
    assert gt_in["truth"] > 0 and gt_in["precision"] == gt_in["recall"] == 1 and 0 <= gt_in["lag_s"] < MATCH_S
    t = json.loads((run_dir / "bev_tracks/perfect_g/gate/tasks.json").read_text())
    assert t["observed"]["gates"] == ["gate"] and t["not_estimated"] and len(t["1_count"]["present"]) == t["seconds"]
    assert (run_dir / "bev_tracks/perfect_g/gate/graph/zone_counts.csv").exists() and (run_dir / "bev_eval/per_camera.csv").exists()
    # 正解の mot/ を追跡の枠として置けば、カメラごとの指標は 1
    folder = run_dir / "bev_tracks/perfect_g/gate"
    (folder / "mot").mkdir(exist_ok=True)
    for c in CONFIG["camera_groups"]["gate"]["cameras"]:
        shutil.copyfile(run_dir / "mot" / f"{c}.txt", folder / "mot" / f"{c}.txt")
    _, _, f0, n = open_run(run_dir)
    pc = per_camera(run_dir, f0, n, folder, ["G1", "CamGate"], range(n), vis_min=None)
    assert pc["G1"]["IDF1"] == 1 and pc["CamGate"]["MOTA"] == 1 and pc["G1"]["count_mae"] == 0


@pytest.mark.skipif(not __import__("shutil").which("ffmpeg"), reason="ffmpeg がない")
def test_overlay(run_dir):
    import subprocess

    from twinsim.bev import overlay
    (run_dir / "video").mkdir(exist_ok=True)
    for c in CONFIG["camera_groups"]["gate"]["cameras"]:
        subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-f", "lavfi", "-i", "color=c=gray:s=320x180:r=10:d=2", "-pix_fmt", "yuv420p",
                        str(run_dir / "video" / f"{c}.mp4")], check=True)
    synthetic_tracks(run_dir, label="perfect_o")
    out = overlay(run_dir, "perfect_o", "gate", range(0, 10))
    assert out.exists() and out.stat().st_size > 1000


def test_perturb(run_dir):
    """A の 15 通りの揺らしと、その平均・標準偏差の評価。"""
    from twinsim.bev.trackers import perturbations
    from twinsim.bev.trackers.a import DEFAULTS
    ps = perturbations(DEFAULTS, 10)
    assert len(ps) == 15 and ps[0] == {} and ps[-1]["max_age_s"] == pytest.approx(DEFAULTS["max_age_s"] + 0.1)
    detect(run_dir, "gt_boxes")
    info = track(run_dir, "a", dets="gt_boxes", label="pt", perturb=True)
    assert len(info) == 15 and info["gate/pt_p3"]["params"]["r_suppress"] != info["gate/pt_p0"]["params"].get("r_suppress", -1)
    res = evaluate(run_dir, "pt")["gate"]
    assert len(res["repeats"]) == 15 and res["repeats"][0] == "pt_p0" and res["metrics"]["1m"]["IDF1"] >= 0.9
    with pytest.raises(ValueError, match="perturb"):
        track(run_dir, "mv3dt", perturb=True, gpu=0)


@pytest.mark.parametrize("mode", ["post", "closed"])
def test_a_window_optimisation(run_dir, mode):
    """A の窓最適化（位置だけ）: 訂正した結果（corrected）が出て、正解の枠からなら高い値。post は毎コマの出力を変えない。"""
    detect(run_dir, "gt_boxes")
    params = {"be_mode": mode, "be_window_s": 4.5, "be_every_s": 1.5, "be_w_app": 0, "be_link_gain": 0.01, "be_merge": False, "be_min_len_s": 0}
    track(run_dir, "a", dets="gt_boxes", params=params, label=f"be_{mode}")
    on = evaluate(run_dir, f"be_{mode}", kind="online", label=f"be_{mode}_on")["gate"]["metrics"]["1m"]
    co = evaluate(run_dir, f"be_{mode}", kind="corrected", label=f"be_{mode}_co")["gate"]["metrics"]["1m"]
    assert co["IDF1"] >= 0.9 and on["IDF1"] >= 0.9
    if mode == "post":
        base = evaluate(run_dir, "a")["gate"]["metrics"]["1m"]
        assert on["IDF1"] == base["IDF1"]
    with pytest.raises(ValueError, match="be_mode"):
        track(run_dir, "a", dets="gt_boxes", params={"be_mode": "nope"}, label="bad")


def test_appearance(run_dir):
    """胴の色のヒストグラム（OpenCV があるときだけ）: 行の数・合計 1・作り直さない。"""
    cv2 = pytest.importorskip("cv2")
    import numpy as np

    from twinsim.bev.appearance import ensure
    _, gt, _, _ = open_run(run_dir)
    (run_dir / "video").mkdir(exist_ok=True)
    w, h = gt["cameras"]["G1"]["width"], gt["cameras"]["G1"]["height"]
    vw = cv2.VideoWriter(str(run_dir / "video" / "G1.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 10, (w, h))
    for _ in range(40):
        vw.write(np.full((h, w, 3), (30, 90, 200), np.uint8))
    vw.release()
    detect(run_dir, "gt_boxes", name="app_test", cameras=["G1"])
    folder = run_dir / "bev_dets" / "app_test"
    assert ensure(run_dir, gt, folder, ["G1"]) == ["G1"] and ensure(run_dir, gt, folder, ["G1"]) == []
    H = np.load(folder / "app_G1.npy").astype(np.float32)
    n = len((folder / "G1.txt").read_text().splitlines()) - 1
    assert len(H) == n and n > 0
    early = [k for k, line in enumerate((folder / "G1.txt").read_text().splitlines()[1:]) if int(line.split(",")[0]) < 40]
    assert all(abs(H[k].sum() - 1) < 1e-2 or H[k].sum() == 0 for k in early)
    D = load_dets(folder, ["G1"], app=True)
    assert len(next(iter(D.values()))[0]) == 4


def test_mopf_compare(run_dir):
    """with-mopf との比べ: 1 台ずつの追跡（MOPF の代わり）→ links.json（ここではカメラの ID をそのまま全体の番号にする）→ 取り込み → 同じ画面の上で採点。"""
    from twinsim.bev import compare as mopf
    detect(run_dir, "gt_boxes")
    cams = ["G1", "CamGate"]
    tdir = mopf.camera_tracks(run_dir, cams, "gt_boxes")
    rows = [json.loads(x) for c in cams for x in (tdir / f"{c}.jsonl").read_text().splitlines()]
    assert rows and {"frame_id", "tracking_id", "position_x_min", "position_y_max"} <= set(rows[0])
    tl = []
    for c in cams:
        per = {}
        for x in (tdir / f"{c}.jsonl").read_text().splitlines():
            r = json.loads(x)
            per.setdefault(r["tracking_id"], []).append(r["frame_id"] / 10)
        tl += [{"camera": c, "mopf_id": i, "t0": min(ts), "t1": max(ts), "global_id": f"{c}:{i}"} for i, ts in per.items()]
    links = run_dir / "links_test.json"
    links.write_text(json.dumps({"cameras": cams, "method": "none", "tracklets": tl}))
    folder = mopf.import_mopf(run_dir, "gate", "mopf_test", tdir, links)
    assert (folder / "bev.csv").exists() and all((folder / "mot" / f"{c}.txt").exists() for c in cams)
    synthetic_tracks(run_dir, ["gate"], label="perfect_m")
    res = mopf.compare(run_dir, [("mopf", "mopf_test/gate"), ("perfect", "perfect_m/gate")], cams)
    m, p = res["results"]["mopf"], res["results"]["perfect"]
    assert m["Rec"] > 0.6 and m["Prec"] > 0.8          # 正解の枠から作った検出なので、枠はよく合う
    assert p["IDF1"] > m["IDF1"] - 0.05                  # 床の上の正解の追跡は、床の位置から作った枠でも合う
    assert p["handovers"] == m["handovers"]
    if p["handovers"]:
        assert p["handover_rate"] > 0.8 and m["handover_rate"] < 0.2   # つながない 1 台ずつの追跡は、カメラの間で番号が変わる
    assert (run_dir / "bev_eval" / "compare_G1_CamGate.json").exists()
