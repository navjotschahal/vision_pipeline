# Camera-to-robot calibration: how it works and how we will automate it

Status, 2026-09-16:
- **Done:** the robot-agnostic solver, validation, pose selection, and rig profiles are
  implemented and tested on synthetic data.
- **Not built yet:** the ROS 2 bridge and live capture.
- **Robot motion:** nothing here moves the robot.

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

## 4. Robot-agnostic design

The rig may be the bimanual OpenArm today and a single Franka, a bimanual Franka, or a
KUKA + Franka pair later. Nothing in the calibration core names a robot.

**What changes per robot is one profile file** (`configs/calibration/*.yaml`, loaded by
`calibration/robot_profile.py`). It holds, per arm:
- `base_frame` → `flange_frame`, the TF frames forward kinematics connects;
- joint names and limits, with a safety margin;
- the ROS 2 `FollowJointTrajectory` action and a calibration speed cap;
- the mount: `eye_to_hand` (fixed camera) or `eye_in_hand` (wrist camera, common on
  Franka setups).

A rig can also give each arm's nominal base pose in a shared reference frame. Profiles
exist for OpenArm v1.0 and v2.0 (checked against their URDFs and controller configs).
Profiles for other robots are written from the running system, not guessed:

```bash
ros2 run tf2_tools view_frames          # base and flange frame names
ros2 topic echo /joint_states --once    # joint names
ros2 action list -t                     # FollowJointTrajectory action names
```

Franka FR3 (`franka_ros2`) and KUKA LBR iiwa/Med (`lbr_fri_ros2_stack`) both use ROS 2,
ros2_control, and MoveIt 2, the same interface pattern as OpenArm.

**What stays the same** (`calibration/hand_eye.py`, ROS-free, numpy + OpenCV):
- ChArUco board pose: IPPE, then Levenberg-Marquardt refinement.
- The motion-diversity check, which refuses rotation-poor data.
- All five closed-form solvers, then joint refinement of the camera and board transforms
  on corner reprojection, with a Huber loss and outlier rejection.
- Held-out validation, and the greedy next-pose selector.
- Both mounts, for any number of arms.

**Mixed or bimanual rigs.** Each arm is calibrated independently against the shared
camera:
- With a common body link (OpenArm), `relative_base_error` compares the implied
  arm-to-arm transform with the URDF.
- With separately mounted arms (KUKA + Franka) there is no nominal transform. Validation
  is then cross-arm: both arms point at the same board corner or at each other's flange,
  and the camera-predicted positions must agree.

**Two integration constraints found on this machine:**
- ROS 2 Humble's `rclpy` only imports under the system Python 3.10. The `.venv` is
  3.12, and `vision_pipeline` requires ≥3.11. The ROS side is therefore a thin bridge
  process: it reads TF and joint states, executes trajectories, and checks collisions
  through MoveIt. It exchanges plain messages with the calibration core over localhost.
- The OpenArm bimanual trajectory controllers run with `interpolation_method: "none"`
  and expect high-frequency commands. A calibration move must therefore send a densely
  sampled, velocity-limited trajectory, never one distant waypoint.

**Measured on synthetic data** (`tests/test_hand_eye.py`; 20 poses, 0.2 px corner noise,
three seeds). This shows why refinement is standard:

| Estimate | Camera position error | Camera rotation error |
|---|---|---|
| Best closed-form method per seed (Daniilidis) | 0.6–6.8 mm | 0.20–0.60° |
| Worst closed-form method per seed | 7.0–32.1 mm | 0.33–0.79° |
| After reprojection refinement (vs ground truth) | 0.13–0.21 mm | ≤ 0.014° |

Closed-form rows are deviations from the refined result, which is itself within 0.21 mm
and 0.014° of the ground truth.

Held-out board corners after refinement: 0.5–1.1 mm mean, 1.1–3.2 mm p95. Real data adds
kinematic error, board flatness, and depth-independent PnP noise, so expect worse on the
bench.

## 5. When the camera moves, and what the D435I IMU can add

**Robot on the bench:** OpenArm v1.0, so `configs/calibration/openarm_v1_bimanual.yaml` is the
active rig. The camera may be repositioned.

**Why it matters.** Calibration measures `T_base_cam` for one physical camera placement.
Every perception output reaches the robot through that transform, so moving the camera
by 1 cm or tilting it by 0.5° moves every commanded contact by about that much (0.5° is
~8.7 mm at 1 m). How the camera may move decides how the transform is kept correct:

| Camera placement | What keeps `T_base_cam` correct | Effort |
|---|---|---|
| Rigidly on the robot's own frame | One calibration; it survives moving the whole rig | Lowest; **recommended** |
| On a stand that is moved between sessions | Re-run auto-calibration after every move; detect moves | Minutes per move |
| On the wrist (eye-in-hand) | One `flange_from_camera` calibration; kinematics gives the pose every frame | Low; also gives multiple viewpoints |
| Moving freely during a task (head, handheld) | Continuous pose tracking against the robot (markerless robot tracking, or a fiducial on the base) | Hard; not recommended now |

A fixed camera helps because the unknown is a constant. It can be solved once with many
poses and refinement, validated, and reused, and nothing has to be tracked or timestamped
at run time.

**What the IMU can and cannot do.** The D435I carries a Bosch BMI055: a 3-axis
accelerometer and gyroscope, with no magnetometer. Measured on this camera, held still
for 11 s (`pyrealsense2`, 200 Hz profiles):

| Quantity | Measured |
|---|---|
| Gravity direction noise, per sample | 0.16° RMS |
| Gravity direction, 1 s averages | stable within 0.08° over 11 s |
| Accelerometer magnitude | 9.737 m/s² (uncalibrated scale/bias, ~0.7 % low) |
| Gyro bias | up to 0.24°/s per axis; raw integration drifts 3.5° in 11 s |
| Gyro after subtracting the measured bias | 0.31° in 11 s |

The capture used a blocking frameset read, which delivered ~38 Hz of the 200 Hz streams.
The static statistics are still valid.

So the IMU observes 2 of the 6 numbers in `T_base_cam`: roll and pitch relative to
gravity. It cannot observe:
- **heading about gravity:** there is no magnetometer, which would be unreliable next to
  motors anyway, and the gyro drifts;
- **position:** double-integrated acceleration diverges within a fraction of a second.

It cannot replace calibration, but it closes part of the gap:

1. **Detecting a move.** A tilt change of ≥ 0.3° (four times the 1 s noise) or a gyro
   burst marks the stored calibration stale and triggers a quick recheck. A slow pure
   slide with no rotation is invisible to the IMU, so the startup visual recheck is
   still required.
2. **Constraining a recalibration.** If the robot base's "up" is known (a level base, per
   the URDF), gravity fixes roll and pitch. Vision then only has to solve yaw and
   position, which makes a quick recheck better conditioned.
3. **A camera moving during a task** would need visual-inertial fusion. The IMU steadies
   short-term rotation, but vision still has to anchor the camera to the robot.

The camera-to-IMU rotation comes from librealsense's motion-module extrinsics. The
accelerometer's scale error only matters for absolute tilt, not for detecting a change.

## 6. How the arms are moved

Nothing new drives the motors. Calibration uses the same stack as the existing
controllers, commanded more cautiously:

```
calibration core (.venv, Python 3.12)             robot bridge (system Python 3.10, rclpy)
  select next pose ───── joint targets ──────────▶  MoveIt /check_state_validity  (collision: both arms, body, table)
  predict board in view                             MoveIt planning, velocity/acceleration scaling ~0.1
                                                    re-time to <= 0.3 rad/s, resample to ~100 Hz  (controllers do not interpolate)
                                                    FollowJointTrajectory on /left|right_joint_trajectory_controller
  capture board + joints ◀── settled joints, TF ──  /joint_states, TF base -> flange (robot_state_publisher)
                                                    abort on: deadman released, tracking error, limit margin, timeout
                        openarm_bringup (ros2_control, CAN) ──▶ motors
```

**Launch** (arguments checked against the v1.0 launch files; the CAN interfaces come up
first with the OpenArm SocketCAN setup script):

```bash
ros2 launch openarm_bringup openarm.bimanual.launch.py arm_type:=openarm_v1.0 use_fake_hardware:=true   # stage 1
ros2 launch openarm_bimanual_moveit_config move_group.launch.py arm_type:=openarm_v1.0
```

Pass `openarm_v1.0` explicitly. The bringup defaults to v2.0, and the MoveIt launch uses
`arm_type` as its config folder name.

**Where poses come from** (no hand-written pose lists):

1. **Seed.** Each arm is placed once in a "presentation pose" with the board facing the
   camera; the bridge records the joints.
2. **Bootstrap.** Six small wrist moves (joints 5–7, ±10°) around the seed, each checked
   by MoveIt, give a rough calibration.
3. **Explore.** Candidates are sampled within the profile limits over wrist and elbow
   joints (±35°). A candidate is kept only if MoveIt says it is collision-free, and if
   the rough calibration predicts the board inside the image, facing the camera within
   60°. `select_diverse_pose` then picks the one adding the most new rotation.
4. **Stop** once rotation diversity and the held-out error meet the acceptance gate.

One arm moves at a time. The other holds position and is part of the collision model.

**Safety staging**, each stage gating the next:

| Stage | Setup | What moves |
|---|---|---|
| 1 | Fake hardware (`use_fake_hardware:=true`) + RViz | Nothing physical; checks planning, timing, abort paths |
| 2 | The existing MuJoCo hardware sim | Simulated arms, with a rendered camera if available |
| 3 | Real arms at 10 % speed | Each pose approved before execution; e-stop in hand |
| 4 | Real arms, automatic sequence | Deadman held for the whole run; e-stop in reach |

## 7. The auto-calibration loop

```
calibration planner ──▶ safe executor ──▶ synchronized capture ──▶ solver + refinement ──▶ validation report ──▶ versioned calibration + TF
      ▲                (ROS 2 bridge:          (RealSense frames +      (hand_eye.py)              (held-out error,
      │                 dense, slow             joint states, settle                               mesh overlay,
      │                 trajectories)           check)                                             arm-to-arm check)
      └── next best pose (most new rotation, board predicted in view, collision-free) ◀────────
```

1. **Planner.**
   - Sample joint configurations within the profile's limits and margin.
   - Keep candidates that MoveIt reports collision-free (including the other arm) and
     where the board is predicted to face the camera, using a rough first estimate from
     3 hand-chosen poses.
   - Choose among them with `select_diverse_pose`.
2. **Executor.**
   - A slow, dense trajectory on the profile's action, with an operator deadman.
   - Proven on fake hardware or the sim first; runs on real arms only with explicit
     sign-off.
3. **Capture.**
   - After motion settles: joint states, TF `base_from_flange`, and 10 RealSense frames.
   - Stored as a replayable dataset, so the solve can be rerun without the robot.
4. **Solve, validate, and publish.** `calibrate_hand_eye`, then `validate_hand_eye` on
   held-out poses, then the arm-to-arm check, then a versioned result stamped with a
   `CalibrationRef`.
5. **Markerless recheck** (later): Hydra (RGB-D ICP of robot geometry) or
   EasyHeC / EasyHeC++ (differentiable rendering), to detect drift between marker
   calibrations.

## 8. Sources

- OpenCV `calibrateHandEye` / `calibrateRobotWorldHandEye` API and eye-to-hand notes:
  [calib3d.hpp](https://github.com/opencv/opencv/blob/4.x/modules/calib3d/include/opencv2/calib3d.hpp)
- R. Tsai, R. Lenz, "A New Technique for Fully Autonomous and Efficient 3D Robotics
  Hand/Eye Calibration", IEEE T-RA 5(3), 1989: [PDF](https://kmlee.gatech.edu/me6406/handeye.pdf)
- R. Horaud, F. Dornaika, "Hand-Eye Calibration", IJRR 1995: [arXiv:2311.12655](https://arxiv.org/abs/2311.12655)
- easy_handeye2 (ROS 2): [github.com/marcoesposito1988/easy_handeye2](https://github.com/marcoesposito1988/easy_handeye2)
- EasyHeC, RA-L 2023: [arXiv:2305.01191](https://arxiv.org/abs/2305.01191), code [ootts/EasyHeC](https://github.com/ootts/EasyHeC)
- EasyHeC++, IROS 2024: [arXiv:2410.09293](https://arxiv.org/abs/2410.09293)
- Hydra: Marker-Free RGB-D Hand-Eye Calibration: [arXiv:2504.20584](https://arxiv.org/abs/2504.20584)
- franka_ros2 (Franka FR3, ROS 2): [github.com/frankarobotics/franka_ros2](https://github.com/frankarobotics/franka_ros2)
- LBR-Stack, ROS 2 for KUKA LBR iiwa/Med: [github.com/lbr-stack/lbr_fri_ros2_stack](https://github.com/lbr-stack/lbr_fri_ros2_stack), [arXiv:2311.12709](https://arxiv.org/abs/2311.12709)
- D435I IMU (Bosch BMI055; accel/gyro rates): [librealsense d435i.md](https://github.com/realsenseai/librealsense/blob/master/doc/d435i.md)
- Intel RealSense D400 self-calibration and health check: [RealSense docs](https://dev.realsenseai.com/docs/intel-realsense-self-calibration-for-d400-series-depth-cameras/)
