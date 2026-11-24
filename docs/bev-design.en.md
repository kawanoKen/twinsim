# Design of multi-camera tracking on the floor plane (differences from main and from with-mopf)

[日本語](bev-design.md)

2026-11-25. Fork `kawanoKen/twinsim`, branch `feature/bev`. It is built on branch `feature/camera-groups` (v1.2.0 of `upstream/main` plus `camera_groups` only; the candidate for main).

A summary of the feature that tracks people on the floor plane (the top view) with one ID per person for each group of cameras with overlapping views, and compares the result with the simulator's ground truth.
It describes what was added to the simulator, what is read and written, how to run it from a rendering, and how it differs from the MOPF integration on the `with-mopf` branch.
How the trackers work (the algorithms) is not covered here (usage and settings: [bev-tracking.en.md](bev-tracking.en.md); plan and progress: [bev-tracking-plan.en.md](bev-tracking-plan.en.md); file formats: [formats.en.md section 12](formats.en.md#bev)).

## 1. Overall flow

```
Scenario (add camera_groups) ─→ twinsim run (CPU) ─→ ground truth (ground_truth, mot, aicity, tasks, graph)
                                      │
                                      └→ twinsim render (GPU, Isaac Sim) ─→ video/<camera>.mp4
                                                                             │
  twinsim bev-detect ─→ bev_dets/<detector>/        from ground-truth boxes (CPU) or PeopleNet (GPU, DeepStream)
         │
  twinsim bev-track ──→ bev_tracks/<name>/<group>/   A (CPU) or MV3DT (GPU, DeepStream)
         │                 bev.csv (floor-plane tracks) plus mot/, aicity/, graph/, tasks.json in the ground-truth formats
         │
  twinsim bev-eval ───→ bev_eval/                   floor-plane metrics, per-camera metrics, counts and aggregates against the ground truth
  twinsim bev-overlay → bev_overlay/                check video with the top view and the cameras side by side
```

- The simulator's own flow (`run` → `render`) is unchanged. Floor-plane tracking reads the run folder and writes into `bev_*` in the same folder.
- Detection and tracking are separate. Detections are saved once, so trackers can be compared on the same detections.
- Tracker output is also written in the ground-truth formats (`mot/`, `aicity/`, `graph/`, `tasks.json`), so tools that read the ground truth can read the estimates as they are.

## 2. What was added to the simulator (main)

### 2.1 Changes to the simulator

Only one configuration item was changed, and it is split out on branch `feature/camera-groups` (the base of this branch). The simulation and the other outputs are unchanged.

| Place | Change |
|---|---|
| `twinsim/scenario.py` | Adds `camera_groups` to the configuration (`{group: {cameras: [...], area: [[x, y], ...], note}}`). Checks group names, unknown keys, cameras not in `cameras[]`, and the evaluation-area polygon (`camera_group_errors`) |
| `twinsim/outputs.py` | Writes `camera_groups` into `ground_truth.json` (`{}` when absent) |
| `docs/config-reference.md`, `formats.md` | Description of `camera_groups` (Japanese and English) |
| `tests/test_smoke.py` | Kept in the ground truth, motion unchanged, checks |

- Earlier configurations load as before, and the same configuration and seed give the same results. `ground_truth.json` only gains the `camera_groups` key.
- Without `camera_groups`, floor-plane tracking treats all cameras as one group `all`, with the floor seen by the cameras as the evaluation area.

### 2.2 Added code (`twinsim/bev/`)

It only reads the simulator's modules and is self-contained in `twinsim/bev/`.

| Place | Role | Main entry | Needs |
|---|---|---|---|
| `calib.py` | Ground-truth cameras → 3×4 projection matrices | `calib`, `projection` | |
| `groups.py` | Camera groups and evaluation areas | `groups`, `Group` | |
| `io.py` | Opening a run, ground-truth positions, reading and writing detections and tracks | `open_run`, `gt_positions` | |
| `detect.py` | Detection: from ground-truth boxes (`gt_boxes`, with optional misses, jitter and false positives), PeopleNet (`peoplenet`), external detections (`external`) | `detect` | GPU for `peoplenet` |
| `deepstream.py` | Shared code for running the DeepStream container | `peoplenet`, `run_container` | docker, the DeepStream container, models |
| `trackers/` | Trackers (`a`, `mv3dt`) and the entry point | `track` | GPU and mosquitto for `mv3dt` |
| `appearance.py` | Torso colour for A's window optimisation | `ensure` | OpenCV (optional dependency) |
| `presets/` | Tracker settings (per scene) | `--preset` | |
| `evaluate.py` | Floor-plane metrics, per-camera metrics, comparison of counts and aggregates | `evaluate` | |
| `graph.py` | Zone counts and moves (same format as `graph/`) | `graph` | |
| `tasks.py` | Gate passages, stops, congestion, elevator waits (same format as `tasks.json`) | `predict`, `compare` | |
| `overlay.py` | Check video | `overlay` | ffmpeg, rendered videos |
| `synthetic.py` | Tracks made from the ground truth (for checking the evaluation) | `synthetic_tracks` | |

Also added: four commands (`twinsim/__main__.py`), tests (`tests/test_bev.py`), an example (`examples/09_bev_tracking.py`), documents (`docs/bev-*.md`, section 12 of `formats.md`, Japanese and English),
and the optional dependency `bev` (`pyproject.toml`; OpenCV; `uv sync --extra bev`).

### 2.3 Commands

| Command | What it does | GPU |
|---|---|---|
| `twinsim bev-detect <run>` | Detection → `bev_dets/<name>/` | Only `--detector peoplenet` (`--gpu`) |
| `twinsim bev-track <run>` | Tracking per group → `bev_tracks/<name>/<group>/`. `--repeats` (repeat runs), `--perturb` (15 slightly perturbed settings) | Only `--tracker mv3dt` (`--gpu`) |
| `twinsim bev-eval <run> --tracker <name>` | Compare with the ground truth → `bev_eval/` | no |
| `twinsim bev-overlay <run> --tracker <name>` | Check video → `bev_overlay/` | no |
| `twinsim bev-mopf tracks / import / compare <run>` | Comparison with with-mopf (5.4) | no |

From Python: `detect`, `track`, `evaluate`, `overlay` and `synthetic_tracks` in `twinsim.bev`.

### 2.4 External components (only for PeopleNet and MV3DT)

| Environment variable | Contents | Default |
|---|---|---|
| `TWINSIM_DEEPSTREAM_IMAGE` | DeepStream container | `ghcr.io/nvidia/deepstream:9.2-triton-multiarch` |
| `TWINSIM_DEEPSTREAM_MODELS` | Model folder (`PeopleNet2.6.3/`, plus `BodyPose3DNet/` for MV3DT; must be writable because TensorRT engines are stored there) | none (`--models` also works) |
| `TWINSIM_MOSQUITTO` | MQTT broker for MV3DT (the mosquitto executable). Started on a free 127.0.0.1 port for each run and stopped afterwards | `mosquitto` on PATH |
| `TWINSIM_MQTT` | To use a broker that is already running (`host:port`) | none |

## 3. Inputs and outputs

### 3.1 What is read from a simulator run

| File | Use |
|---|---|
| `ground_truth.json` | `cameras` (to make projection matrices), `camera_groups` (groups and evaluation areas), `people` and `tracks` (ground-truth positions; people inside rooms are excluded), `gates`, `cars`, `gate_events`, `people[].events` (ground truth for gates and elevators) |
| `scenario_resolved.json` | The space (zones, walkable floor). Read before the `space` of `ground_truth.json`, so that spaces written in the configuration also load |
| `render_log.json` | Which simulator frame is frame 0 of the video (without it, the run is taken as not rendered and simulator frames are used) |
| `video/<camera>.mp4` | Input to PeopleNet and MV3DT, torso colour, check videos |
| `mot/<camera>.txt` | Visible fraction (to decide who is not judged), detections from ground-truth boxes, ground truth for per-camera scoring |
| `aicity/cameras.json` | Camera numbers (so that the tracker's `aicity/` uses the same numbers) |
| `tasks.json` | Ground truth for aggregates (stops, congestion) and the congestion level thresholds |

### 3.2 What is written

```
runs/<name>/
  (simulator output: ground_truth.json, mot/, aicity/, tasks.json, graph/, video/ …)
  bev_dets/<detector>/        <camera>.txt (detection boxes), run.json, app_<camera>.npy (torso colour; only when used)
  bev_tracks/<name>/<group>/
    bev.csv                   floor-plane tracks (twinsim's own format: frame, ID, position, uncertainty, number of cameras that saw the person)
    mot/<camera>.txt          ┐
    aicity/tracks.txt         │ ground-truth formats (table below)
    aicity/cameras.json       │
    graph/zone_counts.csv     │
    graph/zone_flows.csv      │
    tasks.json                ┘
    run.json                  tracker, settings, detections, elapsed time, twinsim version and git hash
  bev_eval/                   <name>_<group>.json, summary.csv, per_camera.csv
  bev_overlay/                <name>_<group>.mp4
```

What is written in the ground-truth formats:

| Ground truth (simulator) | Estimate (tracker) | Differences |
|---|---|---|
| `mot/<camera>.txt` | `bev_tracks/…/mot/<camera>.txt` | IDs are the tracker's. A's boxes are cylinders made from floor positions |
| `aicity/gt.txt`, `cameras.json` | `bev_tracks/…/aicity/tracks.txt`, `cameras.json` | Same camera numbers. IDs are the tracker's (the same across cameras) |
| `graph/zone_counts.csv`, `zone_flows.csv` | the same names under `bev_tracks/…/graph/` | Points inside the evaluation area only |
| `tasks.json` | `bev_tracks/…/tasks.json` | Only what lies in the evaluation area (`observed`). States, groups and destinations are not estimated (`not_estimated`) |

Besides floor-plane metrics (IDF1, HOTA, MOTA and others), `bev-eval` writes the comparison of these estimates with the ground truth (per-camera metrics, zone counts, gate passages, elevator waits, stops, congestion) into `bev_eval/<name>_<group>.json`.

## 4. How to run

### 4.1 Without rendering (CPU only)

Checks the tracking and scoring pipeline with detections made from ground-truth boxes with added misses and so on.

```bash
twinsim run config.json --out runs/test                    # include mot and aicity in outputs; write camera_groups
twinsim bev-detect runs/test --name noisy --miss 0.1 --jitter-px 3 --fp-per-frame 0.3
twinsim bev-track runs/test --dets noisy --label a_noisy
twinsim bev-eval runs/test --tracker a_noisy
```

Example: `uv run python examples/09_bev_tracking.py` (60 s, 5 cameras, about 30 s).

### 4.2 From rendered videos

```bash
twinsim run config.json --out runs/test
./isaac_up.sh && ./run.sh render runs/test                  # rendering (GPU; check the time and GPU first)
export TWINSIM_DEEPSTREAM_MODELS=<model folder>

twinsim bev-detect runs/test --detector peoplenet --gpu 3                        # PeopleNet detections (once per group)
twinsim bev-track runs/test --dets peoplenet --preset office_gate_ev-gate --group gate --label a_gate   # A
twinsim bev-track runs/test --tracker mv3dt --gpu 3 --repeats 3                  # MV3DT (detects internally; 3 runs)

twinsim bev-eval runs/test --tracker a_gate --frames 2000 3000                   # score video frames 2000–2999
twinsim bev-eval runs/test --tracker mv3dt --frames 2000 3000                    # mean and standard deviation of 3 runs
twinsim bev-overlay runs/test --tracker a_gate --group gate --frames 2000 2300   # check video
```

- `--gpu` selects one GPU (no default). Check `nvidia-smi` for a free one first.
- A's window optimisation is `--param be_mode=post` (post-processing) or `closed` (closed loop). Using torso colour needs OpenCV (`uv sync --extra bev`) and the videos.

### 4.3 Time taken (`office_gate_ev`, population 1,200, 10 fps, 300 s, 5 cameras × 2 groups, one GPU)

| Step | Time |
|---|---|
| PeopleNet detection (10 cameras) | about 40 s |
| A (one group) | 4–8 ms per frame (10–25 s for 3,000 frames, CPU) |
| MV3DT (3 runs × 2 groups) | about 10 min |
| Scoring (one group) | a few to a few tens of seconds |

## 5. Differences from with-mopf

`twinsim/mopf/` on the `with-mopf` branch tracks each camera separately with the external application MOPF (`~/MOPF_ALL_video`) and compares the result with the ground truth.
Floor-plane tracking (`twinsim/bev/`) differs both in how it tracks and in where it compares.

### 5.1 What differs

| | `twinsim/mopf/` (with-mopf) | `twinsim/bev/` (this branch) |
|---|---|---|
| Base | Split from main on 2026-11-02 (59 commits behind current main) | Current main (v1.2.0) |
| Tracking | One camera at a time (MOPF; IDs per camera) | Jointly on the floor for each group of overlapping cameras (A, MV3DT; one ID per person in a group) |
| Overlapping cameras | A person seen in several cameras at once stays several people (not linked) | Merged into one person |
| Between non-overlapping cameras | `linking.py` links them (time, direction, appearance) | Not yet (planned for stage 5) |
| Where scoring happens | On the image (IoU 0.3) | On the floor (1 m, 0.5 m); on the image (IoU 0.3) as a secondary score |
| Running detection and tracking | Start the MOPF server and send videos through its API | Run the DeepStream container on the spot (A needs only the CPU) |
| People-flow monitor lines and areas | Yes (`monitor.py`, `crossings.py`) | No (gate passages are estimated in the `tasks.json` format) |
| Output | `mopf_tracks/`, `mopf_eval/`, `mopf_graph/` | `bev_dets/`, `bev_tracks/`, `bev_eval/`, `bev_overlay/` |
| GUI | The "MOPF" tab | None (commands and Python only) |

### 5.2 Overlapping features

| Feature | `twinsim/mopf/` | `twinsim/bev/` | Rules |
|---|---|---|---|
| Per-camera scoring | `evaluate.py` | `evaluate.per_camera` | Same (IoU 0.3; people less than 0.4 visible are not judged) |
| Zone counts and moves | `graph.py` (`mopf_graph/`) | `graph.py` (`bev_tracks/…/graph/`) | Both in the simulator's `graph/` format |
| Overlay video | `overlay.py` (one camera) | `overlay.py` (top view and the group's cameras) | Different presentation |
| Tracks made from the ground truth | `synthetic.py` | `synthetic.py` | Same idea |

If the two are combined, these four can be shared. Tracking (`mopf/process.py`, `bev/trackers/`) and linking across cameras (`mopf/linking.py`) play different roles and stay separate.

### 5.3 How they fit together

The intended form (stage 5) is: merge people within each group of overlapping cameras with floor-plane tracking, and link between groups (non-overlapping views) in the way `linking.py` does.
`linking.py` links per-camera tracks (runs of one camera's ID), so group tracks (runs of an ID in `bev.csv`) should be passed to it in the same way.

### 5.4 How to compare (`twinsim bev-mopf`, `twinsim/bev/compare.py`)

It does not use the with-mopf code; it reads only its output files (`mopf_tracks/<camera>.jsonl`, `mopf_eval/links.json`).
Both trackings are placed in the same rendered result and compared.

```
with-mopf: mopf-process (MOPF) or bev-mopf tracks (stand-in) → mopf_tracks/ → mopf-link → mopf_eval/links.json
this branch: bev-mopf import → bev_tracks/mopf/<group>/ (MOPF boxes, floor points)
             bev-track (A, the group's cameras) → bev_tracks/a/<group>/
             bev-mopf compare → bev_eval/compare_<cameras>.json
```

| Rule | Meaning |
|---|---|
| Views scored | MOPF's cameras (default CamGate, CamEV). For floor-plane tracking without boxes for a camera, boxes are made from the floor positions (a cylinder 1.7 m tall, radius 0.3 m) |
| People judged | Visible fraction 0.4 or more and floor position inside the evaluation area (`area` of `--area-group`). Boxes that overlap only other people are not counted |
| Metrics | IDF1 over the two cameras (as in with-mopf `mopf-link`: IoU 0.3, frames of each camera offset into one sequence), MOTA, precision, recall, and the share of handovers between cameras (a person disappears from one and appears in the other within 90 s) kept under the same ID |
| Stand-in for MOPF | When MOPF cannot be run, `bev-mopf tracks` builds per-camera tracking (ByteTrack-style) from the same detections. Its IDs break more often than MOPF's, so its numbers do not stand in for MOPF |

```bash
twinsim bev-mopf tracks runs/test --cameras CamGate CamEV --dets peoplenet       # without MOPF: mopf_tracks_peoplenet/
cd <with-mopf checkout> && uv run python -c "from twinsim.mopf import link; link('runs/test', cameras=['CamGate', 'CamEV'], tracks_dir='runs/test/mopf_tracks_peoplenet', apps=('color',))"
twinsim bev-mopf import runs/test --group mopf2 --tracks runs/test/mopf_tracks_peoplenet   # → bev_tracks/mopf/mopf2/
twinsim bev-mopf compare runs/test --tracker A=a/s7 --tracker mopf=mopf/mopf2 --area-group s7
```

## 6. Known limitations

- People leaving through the gate are overcounted: in the simulator some people walk through the lanes without using the gate and are not in the ground-truth `gate_events` (this cannot be told apart from video).
- Elevator waits come out long even when estimated from ground-truth positions (slow walking through the crowd before reaching the waiting spot also counts as stopped).
- Seated people (the sitting motion of 1.2) have lower boxes, so A's position, which fits a standing cylinder, is off. Not handled yet.
- MV3DT gives different results from run to run with the same settings (read the mean over `--repeats`).
- PeopleNet and MV3DT need the DeepStream container and models (not run in CI; the tests replace the container call).
