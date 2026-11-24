# Plan: multi-camera tracking on the floor plane (BEV tracking)

[日本語](bev-tracking-plan.md)

2026-11-16. Fork `kawanoKen/twinsim`, branch `feature/bev` (on `feature/camera-groups`; the earlier branch `feature/bev-tracking` started from `upstream/main` `7dfe9a6`, 1.1.0).

Make tracking that follows each person with one ID on the floor plane (BEV) across several cameras with overlapping views run end to end inside twinsim: "render → detect → track → evaluate".
This follows "1.3 two or more cameras" in the author's roadmap (groups of cameras with and without overlapping views; ground truth and evaluation for matching the same person across cameras).
The integration follows the same pattern as the MOPF integration on branch `with-mopf` (`twinsim/mopf/`, `mopf-process`, `mopf-eval`, `docs/integration.md`).

Two trackers are added.

- **A**: a centralised online tracker that assigns the detections of all cameras directly to one set of tracks on the floor (no training, CPU). Optional add-ons: window optimisation and closed loop.
  Originally `experiments/closed_loop/code/cltrack.py` (outside this repository).
- **Official MV3DT**: multi-view 3D tracking of NVIDIA DeepStream 9.2. Distributed: each camera tracks on its own and IDs are aligned over MQTT. Called as an external application, like MOPF (GPU).

## 0. Progress

| Date | Stage | Contents |
|---|---|---|
| 2026-11-16 | 1 done | `camera_groups` in the scenario; `twinsim/bev/` (`calib`, `groups`, `io`, `detect` (`gt_boxes`, `external`), `trackers/a`, `evaluate`, `synthetic`); `bev-detect`, `bev-track`, `bev-eval`; `tests/test_bev.py`; documents ([bev-tracking.en.md](bev-tracking.en.md), `formats` and `config-reference` in both languages). All tests and ruff pass |
| 2026-11-16 | part of 2 | The PeopleNet detections of scene 2 were loaded with `external`; A gives the same IDs in every frame as the earlier implementation, with positions within 0.0005 m (the evaluation also gives the same values as motmetrics). `detect peoplenet` remains |
| 2026-11-24 | 2 done | `detect peoplenet` (`twinsim/bev/deepstream.py`: DeepStream's `deepstream-app` per group, only the detections written). On scene 2 every box equals the earlier experiment's, and A reproduces gate 0.761, elevator hall 0.637 (0.768, 0.699 with the exclusion rules) (one GPU, 40 s). Boxes are written with 6 decimals (rounding to 2 changed A's values slightly) |
| 2026-11-24 | 3 done | `trackers/mv3dt` (configuration, a private MQTT broker per run, `deepstream-test5-app`, output conversion), `bev-track --repeats`, mean and standard deviation in `bev-eval`. On scene 2 the mean of 3 runs is 0.640 ± 0.021 for the gate and 0.390 ± 0.011 for the elevator hall (earlier experiment: 0.639, 0.395). 3 runs × 2 groups take 10 min on one GPU |
| 2026-11-24 | part of 4 | Per-camera evaluation (`per_camera.csv`; people outside the evaluation area are not judged), `bev_graph/` (zone counts and moves compared with the ground truth), `bev-overlay`, A's 15 perturbed settings (`--perturb`: gate 0.764 ± 0.006, elevator hall 0.635 ± 0.011; earlier experiment 0.763, 0.635), `examples/09_bev_tracking.py`. Left: A's window optimisation and closed loop (`finish`), comparison of gate passages and elevator waiting times |
| 2026-11-24 | more of 4 | A's window optimisation (`trackers/backend.py`, `be_mode` = `post`, `closed`) and appearance (`appearance.py`; OpenCV is the optional dependency `bev`). Scene 2 values equal the earlier experiment's current code (gate: post 0.783, closed 0.774 / corrected 0.789; elevator hall: 0.586, 0.579 / 0.596). The C values in the earlier record (`robust.json`: 0.789, 0.809) are not produced by the current code (settings changed after the record) |
| 2026-11-24 | Upstream 1.2.0 merged | Rebased onto `upstream/main` (v1.2.0). Spaces written in the config are read from `scenario_resolved.json`; people inside rooms (`in_room`) are left out of the ground truth. All tests (241) and ruff pass |
| 2026-11-25 | Match the simulator's output formats | Trackers write `aicity/` (camera numbers as in the ground truth), `graph/` and `tasks.json` in the ground-truth formats under `bev_tracks/<tracker>/<group>/` (`bev_graph/` is gone). `tasks.json` estimates gate passages, stops, congestion and elevator waits, and `bev-eval` compares them with the ground truth (`twinsim/bev/tasks.py`). Scene 2, last 100 s (A with PeopleNet / tracks made from the ground truth): gate entries 62 against 61 in the ground truth (60 matched within 2 s) / 61 (61); mean elevator wait 19.7 s (73 people) / 29.7 s (47) against 23.8 s. People who walk through the lanes without using the gate are not in the ground-truth `gate_events`, so the outward direction is overcounted (2 in the ground truth, 7 even from the ground-truth tracks) |

## 1. What is known so far (results of the earlier experiments)

Compared on twinsim scenes (rendered with Isaac Sim; the same PeopleNet v2.6.3 detections for both). Values are IDF1 over the evaluation period (a match is within 1 m on the floor).

| Scene | Cameras | People (mean per frame) | Official MV3DT | A |
|---|---|---|---|---|
| Office lobby (19 × 19 m) | 7, overlapping | 28 | 0.36–0.41 | 0.670 |
| Gate corridor of `office_gate_ev` (population 1,200) | 5 (G1–G4 + CamGate) | 5.7 | 0.639 | 0.763 (mean of 15 perturbed settings; 0.760 unperturbed) |
| Elevator hall of the same scene | 5 (E1–E4 + CamEV) | 27.9 | 0.395 | 0.635 (same; 0.637) |

- MV3DT does not align IDs across cameras: the same person gets a different ID in each view (1,323 ID switches in the elevator hall, 352 for A).
- A's add-on (window optimisation) helps only when detections are consistent over time and across views (+0.017 in the gate group, −0.029 in the elevator hall).

## 2. Overall flow

```
scenario (add camera_groups) ─→ twinsim run (CPU) ─→ ground truth (ground_truth, mot, aicity, frames), mid-level graph (graph/)
                                         │
                                         └→ twinsim render (GPU) ─→ video/<camera>.mp4
                                                                      │
              twinsim bev-detect (GPU, or from the ground-truth boxes) ─→ bev_dets/<detector>/
                                                                      │
              twinsim bev-track (A on CPU, MV3DT on GPU) ──────────→ bev_tracks/<tracker>/<group>/
                                                                      │
              twinsim bev-eval ─→ bev_eval/ (floor metrics, per-camera metrics, counts)
                               └→ bev_tracks/<tracker>/<group>/graph/, tasks.json (ground-truth formats) ─→ GNN, optimisation (used like MOPF's mopf_graph/)
```

- Detection and tracking are separate; detections are saved to files once. Trackers can then be compared on the same detections (A and MV3DT are compared on the same detections).
- MV3DT has its detector inside. Detections are not passed in from outside; instead the detector settings inside DeepStream are made the same (NMS keeping the top 100, etc.).
  The detections are also written out of DeepStream into `bev_dets/` and can be A's input (as in the earlier experiments).

## 3. Commands

Following MOPF's `mopf-process`, `mopf-eval`, `mopf-overlay`, `mopf-link`, the commands start with `bev-`. Inside the repository, `./run.sh <command>`.

| Command | What it does | GPU |
|---|---|---|
| `twinsim bev-detect <run> --detector {gt_boxes,peoplenet,external} [--group group]` | Detect people in each camera's video and write `bev_dets/<detector>/` | `peoplenet` uses it |
| `twinsim bev-track <run> --tracker {a,mv3dt} [--group group] [--dets detector] [--preset name]` | Write floor trajectories to `bev_tracks/<tracker>/<group>/` | `mv3dt` uses it |
| `twinsim bev-eval <run> [--tracker ...]` | Compare with the ground truth (`bev_eval/`: floor plane, per camera, counts, aggregates) | no |
| `twinsim bev-overlay <run> --tracker ... --group ...` | Video with the top view and each camera side by side, ground truth, predictions and error marks overlaid | no |
| `twinsim bev-link <run>` (stage 5) | Inference and scoring of linking the same person between groups without overlapping views (gate → elevator hall) | no |

- Commands that use a GPU show the expected time and GPU and ask before starting, as MOPF does (without `--confirm` they only print the estimate).
- From Python: `twinsim.bev.detect`, `track`, `evaluate`, `overlay`.

## 4. Layout

In `twinsim/bev/`, arranged like `twinsim/mopf/`.

```
twinsim/bev/
  __init__.py   list of entry points (written like mopf/__init__.py)
  calib.py      ground-truth camera (4×4 matrix and field of view) → 3×4 projection matrix (world metres → pixels); checked against twinsim's projection
  groups.py     reading camera_groups, evaluation areas, how each group sees the floor (by how many cameras)
  detect.py     detector wrappers: gt_boxes (ground-truth boxes with jitter, misses and false positives), peoplenet (DeepStream), external (load)
  trackers/
    base.py     tracker interface (section 6) and registry
    a.py        A (floor position from a cylinder, online tracking, window optimisation, closed loop)
    mv3dt.py    official MV3DT wrapper (write the configs, start the container and the MQTT broker, convert the output to floor trajectories)
  evaluate.py   floor metrics, per-camera metrics, counts, rules for people who are not judged
  graph.py      mid-level graph from the tracks (same format as graph/)
  overlay.py    video with the top view and each camera side by side
  synthetic.py  tracks for checking (made from the ground truth; errors can be added at fixed rates)
  io.py         common reading (ground truth, detections, tracks)
```

- A runs with numpy and scipy only. twinsim's dependencies do not grow.
- For MV3DT, the DeepStream container (`ghcr.io/nvidia/deepstream:9.2-triton-multiarch`) and the location of the models used in it are given by the user's environment.
  No NVIDIA files or weights go into the repository. The MQTT broker (mosquitto) is also the one available in that environment.

## 5. Configuration additions (`camera_groups` in the scenario)

`camera_groups` is a top-level key of the scenario, like `cameras[]` (no separate configuration file for `bev-*`).

```json
"camera_groups": {
  "gate": {"cameras": ["G1", "G2", "G3", "G4", "CamGate"], "area": [[-4.5, 13.0], [-1.5, 13.0], [-1.5, 21.0], [-4.5, 21.0]]},
  "ev":   {"cameras": ["E1", "E2", "E3", "E4", "CamEV"],   "area": [[-14.5, -7.0], [-9.5, -7.0], [-9.5, 2.0], [-14.5, 2.0]]}
}
```

- `area` is the evaluation-area polygon (world metres). Without it, the floor seen by at least one camera of the group.
- The best tracker settings (A's parameters) depend on the scene (how long lost tracks are kept: short where people stand still close together, long where they walk through).
  They are kept as named presets (`twinsim/bev/presets/<name>.json`) and chosen with `--preset`. The values tuned in the earlier experiments are the first presets.
- Checked before simulating (`scenario.validate`): unknown keys (top level and inside `camera_groups.<group>`), camera names not in `cameras[]`, group names (letters, digits, `_`, `-` only, as for `cameras[]`), and whether `area` is a polygon with 3 or more points. Errors, as in the current twinsim rules.
- Without it, "no groups": `bev-*` treats all of `cameras[]` as one group (named `all`). The simulation result (`twinsim run`) does not change.
- The resolved values are kept in `scenario_resolved.{yaml,json}` and `ground_truth.json` (`camera_groups` next to `cameras`). `bev-*` reads the groups from `ground_truth.json` in the run folder, so the scenario file need not be passed separately.
- Added to the key list and the check table of `docs/config-reference.md` (both languages) and to the description of `ground_truth.json` in `docs/formats.md` (both languages).

## 6. Tracker interface

```python
class Tracker:                                   # twinsim/bev/trackers/base.py
    name: str
    def __init__(self, calib, area, fps, params, floor_z=0.0): ...
    def step(self, frame, detections) -> list[TrackPoint]    # every frame: the result available at that time (online)
    def finish(self) -> list[TrackPoint]                      # at the end: results corrected later (window optimisation etc.)
```

- `detections`: camera, box (x1, y1, x2, y2), confidence. A floor position, for detectors that output one directly, is optional.
- `TrackPoint`: frame, ID, floor position (x, y), uncertainty (2×2), number of cameras that saw it, kind (`online` or `corrected`), delay in frames.
- External methods (MV3DT, later UMPN) do not use `step`: the videos or files are passed at once to an external command, and its output is converted to the same form (`run_offline`).

## 7. Output formats (added to `docs/formats.md` in both languages)

| File | Format | Contents |
|---|---|---|
| `calib/<camera>.json` | JSON | `P` (3×4), `K`, `width`, `height`, `fps`, `eye` |
| `bev_dets/<detector>/<camera>.txt` | CSV | `frame, x1, y1, x2, y2, conf` (optionally `xw, yw`). Frames are 0-based video frames (counted from `frames[0]` of `render_log.json`) |
| `bev_dets/<detector>/run.json` | JSON | Detector, settings, elapsed time, twinsim version (git hash) |
| `bev_tracks/<tracker>/<group>/bev.csv` | CSV | `frame, t, id, x, y, cov_xx, cov_xy, cov_yy, n_views, kind, delay` |
| `bev_tracks/<tracker>/<group>/mot/<camera>.txt` | MOTChallenge | Per-camera boxes made from the floor positions (a cylinder 1.7 m tall with radius 0.3 m, projected) |
| `bev_tracks/<tracker>/<group>/aicity.txt` | AI City MTMC | Same format as the ground-truth `aicity/gt.txt` (`camera_id obj_id frame_id xmin ymin width height xworld yworld`) |
| `bev_tracks/<tracker>/<group>/run.json` | JSON | Tracker, settings, detections used, time per frame, twinsim version |
| `bev_eval/<tracker>_<group>.json`, `bev_eval/summary.csv` | JSON, CSV | The metrics of section 8 |
| `bev_tracks/<tracker>/<group>/graph/zone_counts.csv`, `zone_flows.csv`, `tasks.json` | CSV, JSON | Same formats as `graph/` and `tasks.json` (moved from `bev_graph/` on 2026-11-25) |

## 8. Evaluation rules

- **On the floor (main)**: IDF1, HOTA, MOTA, MODA, ID switches, precision, recall and position error per group, at 1 m (and 0.5 m) on the floor.
  - People not judged (counted neither as misses nor as false positives; predictions near them are dropped):
    - people whose visible fraction is below 0.4 in every camera of the group (the same threshold as the MOPF comparison)
    - people at the moment they appear or disappear at doors, elevators or spawn points (the same idea as `portal` in the benchmark scoring tool)
  - Predictions outside the evaluation area are dropped.
- **Per camera (secondary)**: compare `bev_tracks/.../mot/` with the ground-truth `mot/` (IoU 0.3; visible fraction below 0.4 not judged). Can be put next to the values of `mopf_eval/`.
- **Counts and operational metrics**: error of people per zone, gate passages and elevator waiting times, compared with `tasks.json` and `graph/`.
- **Repeated runs**: MV3DT varies between runs even with the same settings, so by default it runs 3 times and reports mean and standard deviation.
  A reports means and differences over small perturbations (15 settings, main parameters ±3 % etc.), as in the earlier experiments.

## 9. Checks (tests)

| Check | How |
|---|---|
| Projection | The matrix of `calib.py` agrees with twinsim's `Camera.project` within 1e−6 pixels |
| Evaluation | Tracks made from the ground truth (`synthetic.py`) give IDF1 and MOTA of 1; 20 % misses give recall 0.8; with ID swaps the number of ID switches matches (the same checks as MOPF's `synthetic_tracks`) |
| A | On a small scene (`office_smoke` with 3 overlapping cameras), IDF1 ≥ 0.95 from `gt_boxes` detections (no errors). Runs in CI without a GPU |
| Agreement with the earlier experiments | In stages 2 and 3, the values of scene 2 (`office_gate_ev`, population 1,200) are reproduced (A the same values; MV3DT within the run-to-run variation) |
| Style | `uv run --group dev ruff check`, `uv run --group dev pytest -q` |

## 10. Stages

| Stage | Contents | Check at the end |
|---|---|---|
| 1 | `camera_groups` in the scenario (checks; kept in `ground_truth.json`), `calib`, `groups`, file formats, `synthetic`, `evaluate` (floor), `detect gt_boxes`, `trackers/a` (online), `bev-detect`, `bev-track`, `bev-eval`, tests | A small scene runs end to end without a GPU and the tests pass |
| 2 | `detect peoplenet` (only the detections of PeopleNet inside DeepStream are written out), A per group, A's named presets | A's values for scene 2 (unperturbed: gate 0.760, elevator hall 0.637) are reproduced |
| 3 | `trackers/mv3dt` (config generation, container and MQTT, output conversion, repeated runs) | MV3DT's values for scene 2 (0.639, 0.395) are reproduced within the run-to-run variation |
| 4 | A's window optimisation and closed loop (results corrected later, from `finish`), per-camera evaluation, counts and operational metrics, `bev-overlay`, `bev_graph/`, documents (both languages), `examples/` | Documents and examples complete; all tests pass |
| 5 (optional) | `bev-link` (linking between groups), UMPN wrapper, procedure for rendering on several GPUs with the author's render farm (`tools/render_farm.sh`), GUI and REST API | "Evaluation of matching between cameras at distant places" in roadmap 1.3 |

## 11. Rules to keep

- Documents come in Japanese and English (`docs/*.md`, `docs/*.en.md`). When one is changed, change the other the same way. This plan also gets an English version when implementation starts.
- When an output format changes, update `docs/formats.md` (and the English version) and "Output formats" in the README.
- Data (results, videos, detections, tracks) is not put in git.
- GPUs are shared. Check free GPUs with `nvidia-smi` before use; when running on the same machine as detection etc., the default is one GPU shared with it, and at most 2 GPUs when splitting (`CLAUDE.md`). Long computations such as rendering start only after showing the expected time and GPUs and getting confirmation.
- Names of specific companies or buildings are not written in documents or code (`CLAUDE.md`).
- Pushes to the fork are confirmed at each stage. Proposals to the author's repository (pull requests) are decided by the owner of the fork.
- The author keeps updating `main`, so `upstream/main` is merged at each stage.
- Results always record the twinsim version used (git hash) (in the earlier experiments the simulator version differed between scenes).

## 12. Open questions

- For the "visible fraction" of people not judged: use the approximate boxes (`mot/`) or the exact boxes from rendering (`labels/` of `render(..., labels=True)`)? The latter makes rendering 2–3 times slower.
- Should MV3DT's detector setting follow the default (top 20) or A's top 100? (In the earlier experiments the better one depended on the scene.)
