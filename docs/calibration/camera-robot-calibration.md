# Camera-to-robot calibration: how it works and how we will automate it

Status: plan, 2026-09-16. Nothing here moves the robot yet.

## 1. What is being calibrated

Every 3D point the perception pipeline outputs is in `realsense_color_optical`, the
camera's frame. To act on it, the robot needs the same point in its own base frame. The
chain is:

```
pixel ──(intrinsics K)──▶ camera optical frame ──(X = T_base_cam)──▶ arm base frame ──(FK: URDF + encoders)──▶ link / contact frames
```

| Link in the chain | Source | Status |
|---|---|---|
| Intrinsics `K` (color + depth) | RealSense factory calibration, read per frame | Have it; verify (step 0) |
| Depth scale / stereo extrinsics | D400 on-chip calibration | Have it; health-check (step 0) |
| **`T_base_cam` per arm** | **Hand-eye calibration** | **Missing; this document** |
| Forward kinematics | OpenArm URDF + joint encoders | Have it; its error shows up in the residuals |

Our case is **eye-to-hand** (also called eye-on-base): the D435I is fixed to the bench
and the arms move in front of it. The OpenArm is bimanual, and both arm bases hang off
`openarm_body_link0` at a nominal (CAD) `y = ±0.031 m`, `z = 0.698 m`. We calibrate
**each arm separately** and then check the two results against the URDF body offsets.
A mismatch there reveals mounting error before it becomes a missed bimanual squeeze.

## 2. The math, briefly

Mount a calibration target (ChArUco board) rigidly on the arm's last link. Its pose
relative to that link, `T_ee_target`, is unknown but constant. At every robot pose `i`:

```
T_cam_target(i) = T_cam_base · T_base_ee(i) · T_ee_target
       ▲                 ▲           ▲              ▲
   board PnP        unknown X    forward kin.    unknown, constant
```

Taking two poses `i`, `j` cancels `T_ee_target`, which leaves the classic **AX = XB**
equation. `A` is the robot's relative motion, `B` is the board's relative motion as the
camera saw it, and `X` is the transform we want.

- **Rotation carries the information.** `X` is recovered from how the rotation axes of
  `A` and `B` correspond. Pure translations tell you almost nothing about `X`'s rotation.
  At least two motions with non-parallel rotation axes are required (OpenCV), and many
  more are recommended.
- **Tsai & Lenz (1989):** make the rotation angle between stations as large as possible,
  and make the rotation axes of different station pairs as different as possible.
- **Angular error dominates at distance.** 0.5° of rotation error in `X` becomes
  `tan(0.5°) × 1 m ≈ 8.7 mm` of position error one metre from the camera. That is why a
  good calibration spends its effort on rotation diversity, not on more samples in the
  same orientation.

Solver families, all available in OpenCV `calibrateHandEye` / `calibrateRobotWorldHandEye`:

| Method | Idea |
|---|---|
| Tsai–Lenz 1989, Park–Martin 1994, Horaud–Dornaika 1995 | Solve rotation first, then translation (separable) |
| Andreff 1999, Daniilidis 1998 (dual quaternions) | Solve rotation and translation together |
| Shah 2013, Li 2010 (`calibrateRobotWorldHandEye`) | Solve `T_base_cam` and `T_ee_target` jointly (AX = ZB) |

**Eye-to-hand with OpenCV:** pass the inverse forward kinematics, `T_ee_base`, where the
API expects `gripper2base`. The output labelled `cam2gripper` is then `T_base_cam`.

The research and industry standard is a **closed-form solve for the initial estimate,
followed by nonlinear refinement**. The refinement minimizes the reprojection error of
every board corner over `T_base_cam` and `T_ee_target` together, with a robust loss.
Closed-form solvers weight all poses equally and ignore pixel noise; the refinement does
not.

## 3. Recommended process (marker-based, the standard)

`easy_handeye2` (ROS 2, `calibration_type: eye_on_base`) wraps this process for ROS
robots; we implement it directly so that pose selection is automatic and every step is
recorded.

0. **Check the camera itself.**
   - Run the D400 on-chip health check. Recalibrate only if it reports poor health.
   - Detect a ChArUco board at several distances and confirm the corner reprojection
     RMS is sub-pixel.
   - Use the 640×480 color profile the pipeline uses; intrinsics are per resolution.
1. **Mount the target.**
   - ChArUco rather than a plain chessboard: it survives partial views and gives
     identified corners.
   - About 5×7 squares of 30 mm, printed flat, on a stiff 3D-printed plate bolted to the
     last link.
   - Measure one printed square with calipers and use the measured size.
   - The mount must not flex, and its offset does not need to be known.
2. **Choose 15–30 poses per arm.** Each pose must:
   - keep the board fully in view and within ~60° of facing the camera;
   - differ in rotation from the others about at least two non-parallel axes, ideally
     20° or more between stations;
   - spread through the volume where objects will be grasped, not just one spot;
   - stay away from joint limits and singularities.
3. **Capture each pose statically.**
   - Move, then wait until joint velocities are ~0.
   - Record joint states and 10 frames; average the board pose.
   - Reject detections with too few corners or high reprojection error.
   - Store the raw images and joint states, so the solve can be rerun without the robot.
4. **Solve.**
   - Run all five `calibrateHandEye` methods plus `calibrateRobotWorldHandEye`. Healthy
     data makes them agree within a few mm and a fraction of a degree; disagreement
     means bad poses or a slipping mount.
   - Then refine by reprojection error, and remove outlier poses by their residuals.
5. **Validate on held-out poses** (never the poses used to solve):
   - reprojection RMS in pixels, and corner position error in mm;
   - overlay the robot's URDF meshes on the camera image at `T_base_cam`. Misalignment
     is visible immediately;
   - physical check: have the arm touch or hover over a point the camera measured, and
     measure the offset;
   - left/right consistency: the two arm-base estimates vs the URDF body offsets.
6. **Store and publish.**
   - Save a versioned calibration with every pose, residual, and the solver outputs. The
     repository's `CalibrationRef(calibration_id, revision)` contract exists for this.
   - Publish it as a ROS 2 static transform, and stamp perception outputs with that
     `CalibrationRef`.
7. **Recheck routinely.** A quick 3–5 pose check at startup, or after anyone bumps the
   camera, compares against the stored calibration and triggers a full recalibration
   above a threshold.

Proposed acceptance gate, to be confirmed: held-out corner error ≤ 5 mm and ≤ 0.5° across
the grasp volume. For reference, the Hydra paper reports ~5 mm task-space error for its
markerless RGB-D method and ~7 mm for classical baselines.

**OpenArm-specific caveat:** encoder zero offsets and backlash put error directly into
`T_base_ee`. If the residuals stay high even with good board detections, the error is in
the kinematics, not the camera. The next step would then be joint-offset calibration,
using the same captured data.

## 4. The auto-calibration system we will build

"Auto" means that no one hand-picks poses or runs solver scripts. An operator still
starts it, watches it, and holds an emergency stop.

```
calibration planner ──▶ safe executor ──▶ synchronized capture ──▶ solver + refinement ──▶ validation report ──▶ versioned calibration + TF
      ▲                (ROS 2 trajectory      (RealSense frames +      (OpenCV closed form,       (held-out error,
      │                 controller, slow,      joint states, settle     reprojection BA)           mesh overlay,
      └── next best pose (most new rotation, board visible, collision-free) ◀────────────────────── left/right check)
```

1. **Planner.**
   - Sample candidate joint configurations near a "board faces camera" posture, using a
     rough initial `T_base_cam` (measured by hand, or from the first 3 poses).
   - Keep the reachable, collision-free ones: MoveIt bimanual config, self-collision,
     table.
   - Greedily add the candidate that most increases rotation diversity (largest minimum
     angle to already-captured orientations), with the board predicted to be in view.
2. **Executor.**
   - The existing ROS 2 `joint_trajectory_controller`, velocity-limited, with an operator
     deadman and a workspace fence.
   - Proven on fake hardware and the MuJoCo sim first; runs on the real arms only with
     explicit sign-off.
3. **Capture.**
   - RealSense frames paired with `/joint_states` at the same instant, using the
     timestamp and provenance contracts already in `vision_pipeline`.
   - Captured only after motion has settled.
4. **Solver and report.**
   - Deterministic and runnable offline from the recorded dataset.
   - Produces a report with all method estimates, residuals, and validation numbers.
5. **Markerless recheck** (later).
   - Estimate `T_base_cam` from the robot's own meshes, with no board: **Hydra** (RGB-D,
     ICP of robot geometry to depth; ~90 % success from 3 configurations; Apache-2.0;
     ROS 2) or **EasyHeC / EasyHeC++** (RGB, differentiable rendering of the robot mask
     plus automatic joint-space exploration).
   - Good for detecting drift automatically between marker calibrations. It needs
     accurate URDF meshes, and ours are CAD models of a low-cost arm, so it is a check,
     not the primary calibration.

## 5. Sources

- OpenCV `calibrateHandEye` / `calibrateRobotWorldHandEye` API and eye-to-hand notes:
  [calib3d.hpp](https://github.com/opencv/opencv/blob/4.x/modules/calib3d/include/opencv2/calib3d.hpp)
- R. Tsai, R. Lenz, "A New Technique for Fully Autonomous and Efficient 3D Robotics
  Hand/Eye Calibration", IEEE T-RA 5(3), 1989: [PDF](https://kmlee.gatech.edu/me6406/handeye.pdf)
- R. Horaud, F. Dornaika, "Hand-Eye Calibration", IJRR 1995: [arXiv:2311.12655](https://arxiv.org/abs/2311.12655)
- easy_handeye2 (ROS 2): [github.com/marcoesposito1988/easy_handeye2](https://github.com/marcoesposito1988/easy_handeye2)
- EasyHeC, RA-L 2023: [arXiv:2305.01191](https://arxiv.org/abs/2305.01191), code [ootts/EasyHeC](https://github.com/ootts/EasyHeC)
- EasyHeC++, IROS 2024: [arXiv:2410.09293](https://arxiv.org/abs/2410.09293)
- Hydra: Marker-Free RGB-D Hand-Eye Calibration: [arXiv:2504.20584](https://arxiv.org/abs/2504.20584)
- Intel RealSense D400 self-calibration and health check: [RealSense docs](https://dev.realsenseai.com/docs/intel-realsense-self-calibration-for-d400-series-depth-cameras/)
