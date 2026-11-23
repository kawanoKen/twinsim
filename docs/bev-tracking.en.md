# Multi-camera tracking on the floor plane (`twinsim bev-*`)

[日本語](bev-tracking.md)

People are tracked on the floor plane (the top view) for each group of cameras with overlapping views, and compared with the simulator's ground truth.
Detection and tracking are separate steps, so trackers can be compared on the same detections. The plan and its stages are in [bev-tracking-plan.md](bev-tracking-plan.md) (Japanese).

## Procedure

1. Add `camera_groups` to the scenario ([configuration reference](config-reference.en.md#camera-groups)). Without it, all cameras form one group, `all`.
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
   twinsim bev-detect runs/test --detector external --name mydet --source <folder>          # detections made elsewhere (CSV in the same format)
   ```
4. Tracking (`bev_tracks/<name>/<group>/`).
   ```bash
   twinsim bev-track runs/test --dets noisy --preset default --param max_age_s=1.5 --label a_noisy
   ```
5. Evaluation (`bev_eval/`).
   ```bash
   twinsim bev-eval runs/test --tracker a_noisy                    # all frames
   twinsim bev-eval runs/test --tracker a_noisy --frames 2000 3000 # the last 100 s (10 fps)
   ```

From Python, call `detect`, `track`, `evaluate`, `synthetic_tracks`, `calib` and `groups` of `twinsim.bev`. File formats are in [section 12 of formats.en.md](formats.en.md#bev).

## Trackers

| Name | What it does | Needs |
|---|---|---|
| `a` | Centralised online tracker. Fits a person cylinder (1.7 m tall, radius 0.3 m) to each box to get a floor position, assigns detections to tracks with a Hungarian assignment per camera, and follows each person with a Kalman filter fused across cameras. New people are started only when seen by 2 or more cameras | numpy, scipy (CPU, a few ms per frame) |

Settings are in `twinsim/bev/presets/<name>.json` (`--preset`). Durations are written in seconds and converted to frames with the video's fps.

| Preset | Tuned on |
|---|---|
| `default` | A lobby (7 cameras, many people standing and talking) |
| `office_gate_ev-gate` | The gate group of `office_gate_ev` (population 1,200; 5 cameras; people walking through) |
| `office_gate_ev-ev` | The elevator-hall group of the same scene (5 cameras; people waiting and boarding) |

## Evaluation rules

- A prediction within 1 m (and 0.5 m) on the floor is the same person. IDF1, HOTA, MOTA, MODA, ID switches, precision and recall are reported per group (CLEAR MOT and IDF1 counted as in motmetrics, HOTA as in TrackEval).
- Ground truth and predictions outside the evaluation area are not counted.
- People not judged (counted neither as misses nor as false positives; unmatched predictions near them are dropped):
  - people whose visible fraction (approximate, from `mot/`) is below 0.4 in every camera of the group (`--vis-min`; the same threshold as the MOPF comparison)
  - people who appear or disappear inside the evaluation area, for 1 s around that moment (`--portal-s`; elevator doors, entrances)

## Checks

- `tests/test_bev.py`: the projection agrees with twinsim's camera within 1e−6 pixels; tracks made from the ground truth give IDF1 and MOTA of 1; 20 % misses give recall 0.8; the number of ID switches matches; on a small scene A reaches IDF1 ≥ 0.95 from the ground-truth boxes.
- Agreement with the earlier experiment (2026-11-16): on `office_gate_ev` (population 1,200, 10 fps, 300 s, 5 cameras per group), with PeopleNet detections on the rendered video, A gives the same IDs in every frame as the earlier implementation, with positions within 0.0005 m (output rounding).
  IDF1 over the last 100 s (1 m, nobody excluded) is 0.761 for the gate group and 0.637 for the elevator hall; with the rules above, 0.768 and 0.699.
