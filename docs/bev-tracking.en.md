# Multi-camera tracking on the floor plane (`twinsim bev-*`)

[日本語](bev-tracking.md)

People are tracked on the floor plane (the top view) for each group of cameras with overlapping views, and compared with the simulator's ground truth.
Detection and tracking are separate steps, so trackers can be compared on the same detections. The plan and its stages are in [bev-tracking-plan.en.md](bev-tracking-plan.en.md); differences from main and from with-mopf are in [bev-design.en.md](bev-design.en.md).

## Procedure

1. Add `camera_groups` to the scenario ([configuration reference](config-reference.en.md#camera-groups)). Without it, all cameras form one group, `all`. Without `area`, the evaluation area is the walkable floor seen by at least one camera of the group.
   ```json
   "camera_groups": {"gate": {"cameras": ["G1", "G2", "G3", "G4", "CamGate"], "area": [[-4.5, 13.0], [-1.5, 13.0], [-1.5, 21.0], [-4.5, 21.0]]}}
   ```
2. Simulate (include `mot` in `outputs`: it gives the ground-truth boxes and the visible fractions). Rendering is not needed (when rendered, frames follow the video).
   ```bash
   twinsim run config.json --out runs/test
   ```
3. Detection (`bev_dets/<name>/`).
   ```bash
   twinsim bev-detect runs/test                                   # from the ground-truth boxes (no GPU)
   twinsim bev-detect runs/test --name noisy --miss 0.1 --jitter-px 3 --fp-per-frame 0.5   # add misses, jitter and false positives
   twinsim bev-detect runs/test --detector peoplenet --gpu 3   # PeopleNet on the rendered video (DeepStream container, GPU; once per group)
   twinsim bev-detect runs/test --detector external --name mydet --source <folder>          # detections made elsewhere (CSV in the same format)
   ```
4. Tracking (`bev_tracks/<name>/<group>/`; also writes `mot/`, `aicity/`, `graph/` and `tasks.json` in the same formats as the simulator's ground truth).
   ```bash
   twinsim bev-track runs/test --dets noisy --preset default --param max_age_s=1.5 --label a_noisy
   ```
5. Evaluation (`bev_eval/`).
   ```bash
   twinsim bev-eval runs/test --tracker a_noisy                    # all frames
   twinsim bev-eval runs/test --tracker a_noisy --frames 2000 3000 # the last 100 s (10 fps)
   ```

From Python, call `detect`, `track`, `evaluate`, `overlay`, `synthetic_tracks`, `calib` and `groups` of `twinsim.bev` (example: [examples/09_bev_tracking.py](../examples/09_bev_tracking.py)). File formats are in [section 12 of formats.en.md](formats.en.md#bev).

## PeopleNet detections (`--detector peoplenet`)

The rendered videos (`video/<camera>.mp4`) go through DeepStream's `deepstream-app` once per group, and only the detector's output is written (no tracker runs).
The settings are those of the earlier experiment: PeopleNet v2.6.3 (ResNet-34, FP16), people only, NMS (top 100, IoU 0.5), input at the video's pixel size.

- Needs: docker with the NVIDIA container runtime, the DeepStream container (`TWINSIM_DEEPSTREAM_IMAGE`, default `ghcr.io/nvidia/deepstream:9.2-triton-multiarch`),
  and the model folder (`TWINSIM_DEEPSTREAM_MODELS` or `--models`, holding `PeopleNet2.6.3/resnet34_peoplenet.onnx` and `detector_labels.txt`; the TensorRT engine is built on first use and saved there, so it must be writable).
- Choose one GPU with `--gpu` (no default; check `nvidia-smi` first). Population 1,200, 10 fps, 300 s, 10 cameras (2 groups of 5): about 40 s.
- Boxes of every confidence are written; the threshold is a tracker setting (A's `det_conf`).

## Trackers

| Name | What it does | Needs |
|---|---|---|
| `mv3dt` | The official MV3DT (multi-view 3D tracking of NVIDIA DeepStream 9.2). Tracks in each camera and unifies IDs and floor positions across cameras over MQTT (distributed). Detection (PeopleNet) runs inside, so `bev_dets/` is not used. The floor position is the mean of the cameras' foot positions per frame and ID | GPU, the DeepStream container, mosquitto (1–2 min for 5 cameras and 3,000 frames) |
| `a` | Centralised online tracker. Fits a person cylinder (1.7 m tall, radius 0.3 m) to each box to get a floor position, assigns detections to tracks with a Hungarian assignment per camera, and follows each person with a Kalman filter fused across cameras. New people are started only when seen by 2 or more cameras | numpy, scipy (CPU, a few ms per frame) |

Settings are in `twinsim/bev/presets/<name>.json` (`--preset`). Durations are written in seconds and converted to frames with the video's fps.

| Preset | Tuned on |
|---|---|
| `default` | A lobby (7 cameras, many people standing and talking) |
| `office_gate_ev-gate` | The gate group of `office_gate_ev` (population 1,200; 5 cameras; people walking through) |
| `office_gate_ev-ev` | The elevator-hall group of the same scene (5 cameras; people waiting and boarding) |
| `mv3dt` (MV3DT's default) | The official sample settings as they are (`presets/mv3dt_tracker.yml`); DeepStream's default detector settings |
| `mv3dt_nms100` | MV3DT with the detector set as for A (NMS, top 100, IoU 0.5) |

### A's window optimisation (`be_mode`)

A can add a back end after the per-frame tracking (the front end) that re-links the observations of the latest window (the same computation as the earlier experiment).
The window's pieces are linked to the IDs fixed before the window by a minimum-cost path cover; the costs are position (velocity-compensated displacement) and appearance (torso colour).

```bash
twinsim bev-track runs/test --dets peoplenet --preset office_gate_ev-gate --label a_closed \
  --param be_mode=closed --param be_window_s=4.5 --param be_every_s=1.5 --param be_link_gain=0.01 --param be_merge=false --param be_min_len_s=0
twinsim bev-eval runs/test --tracker a_closed --kind online       # per-frame output
twinsim bev-eval runs/test --tracker a_closed --kind corrected    # delayed, corrected output
```

- `be_mode`: `none` (default), `post` (post-processing; only the corrected output changes), `closed` (closed loop; the front end continues from the corrected tracks, so the per-frame output changes too).
- `be_window_s` (window length), `be_every_s` (how often it solves). The other `be_*` are `BE_DEFAULTS` in `twinsim/bev/trackers/backend.py`; `link_gain` etc. in the example are the earlier experiment's "conservative" settings.
- Appearance (torso HSV colour histograms) is made from the rendered videos on first use and stored as `bev_dets/<detector>/app_<camera>.npy`. It needs OpenCV (optional: `uv sync --extra bev`).
  To skip it, `--param be_w_app=0` (position only; no OpenCV or video needed).
- Scene 2 (last 100 s, IDF1): gate group, post-processing corrected 0.783 (A 0.761), closed loop 0.774, corrected 0.789; elevator hall, post-processing 0.586 (A 0.637), closed loop 0.579, corrected 0.596.
  It gets worse in the crowded elevator hall (as in the earlier experiment). The values equal those of the earlier experiment's current code (and the appearance histograms are identical for every box).

### MV3DT (`--tracker mv3dt`)

```bash
twinsim bev-track runs/test --tracker mv3dt --gpu 3 --repeats 3          # bev_tracks/mv3dt_r1, _r2, _r3
twinsim bev-eval runs/test --tracker mv3dt --frames 2000 3000             # mean and standard deviation of the 3 runs
twinsim bev-track runs/test --tracker mv3dt --gpu 3 --param tracker.MultiViewAssociator.minPeerTrackletMatchScore=0.3 --param pgie.topk=100
```

- For each group, the videos go through `deepstream-test5-app` (PeopleNet → MV3DT). Each run starts its own MQTT broker (mosquitto on a free port of 127.0.0.1) and stops it at the end.
  mosquitto is `TWINSIM_MOSQUITTO` (the executable; default `mosquitto` on PATH); to use a running broker, set `TWINSIM_MQTT` (`host:port`).
  The model folder also needs `BodyPose3DNet/bodypose3dnet_accuracy.onnx` besides PeopleNet.
- Settings are `pgie` (detector overrides) and `tracker` (overrides of `presets/mv3dt_tracker.yml`: `"Section.key": value`, or a whole section `{"Section": {key: value}}`). `--param pgie.<key>=value` and `--param tracker.<Section>.<key>=value` also work.
- MV3DT gives different results from run to run with the same settings (the share of IDs unified across cameras varies). Run it several times with `--repeats` and read the mean and standard deviation from `bev-eval`.
- The per-camera boxes (`mot/<camera>.txt`) are MV3DT's own boxes (for A they are cylinders made from the floor positions).

## Evaluation rules

- A prediction within 1 m (and 0.5 m) on the floor is the same person. IDF1, HOTA, MOTA, MODA, ID switches, precision and recall are reported per group (CLEAR MOT and IDF1 counted as in motmetrics, HOTA as in TrackEval).
- Ground truth and predictions outside the evaluation area are not counted. People inside rooms (state `in_room`; not visible) are not in the ground truth.
- People not judged (counted neither as misses nor as false positives; unmatched predictions near them are dropped):
  - people whose visible fraction (approximate, from `mot/`) is below 0.4 in every camera of the group (`--vis-min`; the same threshold as the MOPF comparison)
  - people who appear or disappear inside the evaluation area, for 1 s around that moment (`--portal-s`; elevator doors, entrances)
- Per camera (secondary; `bev_eval/per_camera.csv`): the tracker's per-camera boxes (`mot/<camera>.txt`) are compared with the ground-truth `mot/` (a match is IoU ≥ 0.3, the same rules as MOPF's `mopf-eval`).
  People less than 0.4 visible and people whose floor position is outside the evaluation area are not judged. A's boxes are cylinders made from floor positions, so they are coarser than detector boxes.
- Counts (`bev_tracks/<tracker>/<group>/graph/`): per-second zone counts and per-minute moves between zones, in the same format as the simulator's `graph/`, compared with the ground truth
  (only zones with 60 % or more of their floor inside the evaluation area; `graph` in `bev_eval/<name>_<group>.json`).
- Aggregates (`bev_tracks/<tracker>/<group>/tasks.json`): gate passages, stops, congestion and elevator waits estimated from the tracks in the same format as the ground-truth `tasks.json`,
  compared with the ground truth (`tasks` in `bev_eval/<name>_<group>.json`; how each is estimated is in [formats.en.md](formats.en.md#bev)).
  The estimation itself has errors, so read the values next to those of tracks made from the ground truth (`twinsim.bev.synthetic_tracks`).
- Sensitivity to small setting changes (A): `bev-track --perturb` makes 15 variants with the main settings changed by ±3 % (and the deletion time by ±1 frame) as `<name>_p0` to `_p14`; `bev-eval --tracker <name>` gives the mean and standard deviation (the same perturbation as the earlier experiment).

## Check video (`bev-overlay`)

```bash
twinsim bev-overlay runs/test --tracker a_noisy --group gate --frames 2000 2300   # bev_overlay/a_noisy_gate.mp4
```

The top view on the left (white circles = ground truth, coloured dots and numbers = predictions, red circles = misses, red crosses = false positives, yellow rings = ID switches, grey = not judged)
and the group's cameras on the right (green boxes = ground truth, grey = mostly hidden ground truth, coloured boxes = the tracker's boxes). Needs the rendered videos (`video/`) and ffmpeg.

## Checks

- `tests/test_bev.py`: the projection agrees with twinsim's camera within 1e−6 pixels; tracks made from the ground truth give IDF1 and MOTA of 1; 20 % misses give recall 0.8; the number of ID switches matches; on a small scene A reaches IDF1 ≥ 0.95 from the ground-truth boxes.
- Agreement with the earlier experiment (2026-11-16, 11-24): on the rendering of `office_gate_ev` (population 1,200, 10 fps, 300 s, 5 cameras per group), `--detector peoplenet` gives exactly the boxes of the earlier experiment. With them, A gives the same IDs in every frame as the earlier implementation, with positions within 0.0005 m (output rounding).
  IDF1 over the last 100 s (1 m, nobody excluded) is 0.761 for the gate group and 0.637 for the elevator hall; with the rules above, 0.768 and 0.699.
- A's 15 perturbed settings (2026-11-24): 0.764 ± 0.006 for the gate group and 0.635 ± 0.011 for the elevator hall (the earlier experiment's means: 0.763, 0.635; every setting within 0.0002, from rounding positions to mm in `bev.csv`).
- MV3DT (2026-11-24): on the same scene, the mean and standard deviation of 3 runs are 0.640 ± 0.021 for the gate group and 0.390 ± 0.011 for the elevator hall (1 m, nobody excluded; the earlier experiment: 0.639, 0.395). With the rules, 0.643 ± 0.020 and 0.406 ± 0.011.
