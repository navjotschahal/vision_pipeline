# Lab plate calibration versus the later global-registration replay

The saved right-arm lab figure is:
`recordings/openarm_pc_snapshot/calibrations/openarm_v1/registration_sessions/20260924_211043_right/view-00.residual.png`.

It shows a visually sensible model alignment. The later Mac experiment's roughly
2.03 m / 118 degree discrepancy must not be attributed to this lab figure.

## Evidence from the archived code and files

- `current.json` and `current_plate.json` identify the method as plate-against-base
  calibration using marker PnP and known plate placement.
- `run_residuals()` loads a supplied calibration. It does not run ICP.
- It samples the static body and recorded right arm, transforms measured depth into
  the base frame, and computes Euclidean nearest-model-sample distances.
- Points within 30 mm are coloured with JET (blue near zero, red near the cutoff).
  Sampled model points are projected into RGB and painted white last. There is no
  occlusion buffer. Uncoloured pixels are not necessarily calibration errors.
- Recomputing this image with the archived sampler, saved plate matrix and measured
  joint angles gives the identical count of **24,179** points within 30 mm. Only
  **one pixel out of 307,200** differs from the saved image on this Mac.
- No `result.json`, `current_registration.json`, or `model_registration_*.json`
  exists in the copied calibration snapshot. The figure therefore does not establish
  the magnitude of an ICP refinement; the plate transform already reproduces it.

Reproduction script and machine-readable result:

```sh
.venv/bin/python recordings/registration_lab_audit/reproduce_residual.py
```

Outputs are in `recordings/registration_lab_audit/` and do not alter the snapshot.

## How the board supplies initialization

The front base contact is x = 0.095 m, nearest marker margin 0.010 m, lateral
pattern-centre offset zero, and printed-face height 0.032 m. Edge contact and flat
support fix orientation. Marker dimensions and camera intrinsics allow RGB PnP to
estimate `camera_from_plate`. Composition gives:

```text
base_from_camera = base_from_plate @ inverse(camera_from_plate)
```

The old tape/IMU initialization only selects the board-axis direction in the plate
routine. It does not set the final tilt. The saved plate result averages 59 image
observations and reports 0.794 px reprojection RMS. That pixel residual is not a
millimetre accuracy assessment of the entire workspace.

## What the ICP solve would then do

`solve_session()` loads an initial camera matrix, crops in estimated base
coordinates, poses the model with measured joints, and runs two rounds of robust
point-to-plane ICP. It includes a prior near each round's initial estimate and an
optional board-height plane term. This is separate from `run_residuals()`.

The right-arm session records board height **0 mm**, whereas the saved plate result
uses **32 mm**. If placement is unchanged, a future plane-constrained solve must
use 32 mm or disable that term. This does not affect the saved residual renderer,
which does not use the plane-height constraint. The original snapshot remains intact.

## What changed in the later board-free experiment

The teaching tool used geometric FPFH/RANSAC initialization, a different camera-frame
crop and model sampling, and Open3D ICP without the original plane constraint or
initialization prior. Its wrong global match remained wrong after ICP. This is not
a controlled initializer-only comparison and not an original lab optimization log.

The later 12% overlap uses a downsampled/cropped cloud and 10 mm threshold; the lab
image's 24,179 count uses full-resolution points and 30 mm. Do not compare these as
equivalent metrics. The `replay_clouds` folder contains reconstructed camera and
posed-model PLYs, not a separate saved calibration result.

A fair experiment should first hold cloud selection, model, sampling and ICP
settings fixed and vary only initialization. Board-plane and prior removal can
then be evaluated separately. Independent targets are needed to claim task-space
accuracy for either method.
