# Offline registration lab

Run from the repository root:

```sh
.venv/bin/python -m pip install -e '.[registration]'
.venv/bin/python -m vision_pipeline.apps.registration_lab --demo
```

The demo downloads Open3D's example room scans. These are **not OpenArm captures**.
It computes actual FPFH features, RANSAC alignment, and three scales of robust
point-to-plane ICP. Every recorded ICP step performs one optimizer iteration.

The orange source moves; the cyan target stays fixed. Use the mouse to orbit and
zoom. Keys: `N` next step, `B` previous, `1` unaligned, `2` global result, `3` final,
`C` source RGB, `D` source-to-model distances (green at zero, red at or above the
evaluation threshold). Stage names, transforms, and metrics appear in the terminal.
The coordinate axes describe the target frame. In demo mode this is a scan frame;
with a robot model it is the model world/base frame.

## Your saved OpenArm sessions

The capture code in commit fd41211 saves under:

```text
calibrations/openarm_v1/registration_sessions/<session>/
    session.json          camera intrinsics, depth scale, model path, capture settings
    view-00.depth.npy     filtered depth in metres, not the original frame burst
    view-00.color.png     colour image
    view-00.json          joint angles, hand positions, board observation, capture time
    view-01.*
```

These directories are ignored by Git. A capture timestamp is present, but the format
does not preserve separate sensor/joint acquisition timestamps or the entire raw
frame burst. The intended acquisition is stationary after settling.

The lab PC captures have now been downloaded to
`recordings/openarm_pc_snapshot/calibrations/openarm_v1/registration_sessions/`.
There are two sessions, each with **one** saved view: `20260924_210557_static`
and `20260924_211043_right`. Both include depth, RGB, and board detections; the
second includes the seven right-arm joint angles. Additional hand-eye captures and
visual-check images are preserved in the same snapshot. `inventory.json` lists
the decoded point counts. Transfer checksums matched the remote calibration files
and robot model directory.

The original scripts are preserved in `recordings/openarm_pc_snapshot/source/`.
The scene and its mesh dependencies are under `recordings/openarm_pc_snapshot/cpf/`.
Original session metadata retains the Linux paths; use `--model-xml` to override
them locally. Camera and posed-model PLY exports are in
`recordings/openarm_pc_snapshot/replay_clouds/<session>/`.

The confirmed Linux repository location is `/home/nschahal/work/vision_pipeline/`.
To inspect future captures:

```sh
ssh figueroa_openarmpc_airpenn \
  'ls /home/nschahal/work/vision_pipeline/calibrations/openarm_v1/registration_sessions'
```

Copy a session and supply the corresponding MuJoCo scene with all referenced assets:

```sh
.venv/bin/python -m vision_pipeline.apps.registration_lab \
  --session calibrations/openarm_v1/registration_sessions/SESSION \
  --view view-00 --model-xml /absolute/path/to/scene_sim_openarm_bimanual.xml
```

For the downloaded right-arm capture:

```sh
.venv/bin/python -m vision_pipeline.apps.registration_lab \
  --session recordings/openarm_pc_snapshot/calibrations/openarm_v1/registration_sessions/20260924_211043_right \
  --model-xml recordings/openarm_pc_snapshot/cpf/src/sim/assets/scene_sim_openarm_bimanual.xml \
  --output recordings/registration_lab_right
```

This starts a new global-registration experiment on the saved data. It does not
replay an original ICP iteration history: the original sessions contain captures
and residual images, not a saved sequence of intermediate transforms.

### Recorded local experiment

**This is a later Mac experiment, not the preserved lab result.** The lab residual
image is reproducible from the saved plate calibration and aligns visibly with the
robot. See [the provenance audit](lab-residual-provenance.md) for the distinction,
image legend, and board-height configuration issue.

The right-arm experiment in `recordings/registration_lab_right/` used camera-space
bounds `--crop-camera -0.25 -0.6 0.45 0.4 0.1 1.6` and ten ICP iterations at each
scale. Its final inlier RMSE was approximately 4.97 mm, but source overlap was only
12%. Its camera pose differed from the saved plate estimate by 2.03 m and 118 degrees:
the global match was incorrect. The plate estimate is a comparison, not independent
ground truth. Neither result's small inlier error establishes calibration accuracy.
`overview.png` shows the original RGB/depth and the fixed-threshold fit metrics.

Replay the actual computed steps without rerunning RANSAC:

```sh
.venv/bin/python -m vision_pipeline.apps.registration_lab \
  --replay recordings/registration_lab_right \
  --comparison recordings/openarm_pc_snapshot/calibrations/openarm_v1/current.json
```

Key `4` selects the saved plate calibration. Keys `1`, `2`, and `3` select unaligned,
global, and final ICP respectively. The viewer starts with the plate comparison.
Each stage fits the viewport to the clouds; mouse orbit/zoom can then inspect it.

Alternatively supply `--target posed-model.ply`, already expressed in robot base
coordinates for that capture's joint angles. For generic clouds:

```sh
.venv/bin/python -m vision_pipeline.apps.registration_lab \
  --source camera.ply --target posed-model.ply
```

All coordinates must be in metres. Camera-coordinate cropping is available, e.g.
`--crop-camera -0.6 -0.6 0.25 0.6 0.6 2.0`; choose bounds from your actual scene.
The lab does not require a pre-existing calibration for cropping or initialization.
Background, missing depth, symmetry, occluded model surfaces, and poor overlap can
still produce an incorrect global match. The model sampler includes visual meshes
of the static OpenArm body and recorded arms; fingers are excluded. It does not
perform visibility filtering. This is a teaching tool, not a calibration acceptance gate.

## Reading the results

Outputs default to `recordings/registration_lab/`: input clouds and `stages.json`
with the source-to-target matrix for every stage. Use `--output` for separate runs.
Use `--no-gui` to compute without opening a window.

Metrics use the same evaluation clouds and distance cutoff at every stage:

- Fitness: fraction of source points within the cutoff of a target point.
- Inlier RMSE: Euclidean nearest-neighbour error of those accepted points, in mm.
  This is not the point-to-plane optimizer objective; no-overlap RMSE is null.
- All-source distance p95: nearest-target distance including rejected/background points.

A low RMSE with low fitness can mean only a small subset aligned. These are
single-view surface agreement metrics, not independent camera calibration accuracy.
The lab does not update `current.json` or reproduce the existing multi-view solver's
board constraints or initialization prior. Inspect several captures, then use a
shared multi-view solve and independent reference measurements for final calibration.

Reference algorithms:
[global registration](https://www.open3d.org/docs/release/tutorial/pipelines/global_registration.html)
and [ICP](https://www.open3d.org/docs/release/tutorial/pipelines/icp_registration.html).
