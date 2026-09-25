# Camera-to-robot calibration on the bench: hand-guided ChArUco hand-eye through CPF

Status, 2026-09-24: implemented and verified offline (synthetic sessions, forward
kinematics against CPF's own environment, printed board scale). First run on the real arm
pending. This replaces the tape + IMU extrinsic of 2026-09-16 for `cpf_box_handoff`.

Tool: `python -m vision_pipeline.apps.openarm_hand_eye {board,identify,capture,solve,check}`.
Core: `calibration/hand_eye.py` (solver, unchanged), `calibration/openarm_cpf.py` (CPF
joint state, MuJoCo forward kinematics, result files).

## Why hand-guided rather than autonomous

Hand-eye needs rotations about at least two different axes between poses, 15 deg or more.
CPF's control law commands hand **position only**; wrist orientation is whatever the
nullspace posture task leaves. An autonomous attractor sweep therefore cannot guarantee
the rotation diversity the solver needs, and CPF has no collision model to plan against.
Hand-guiding the arm in `--mode damp` (gravity compensation + task damping + joint-limit
barrier, validated for hand-guiding on 2026-09-08) gives arbitrary wrist rotations in a
few minutes, and the capture itself is automatic: the tool decides when a pose is still,
detected, and new, records it, and solves. An autonomous variant (publish attractor
targets in `--mode hold`, hand-twist the wrist at each station) is a follow-up.

## What the tool needs from CPF

- The RT driver started **with `--shm`** so its `ShmBridge` publishes `q`, `dq`, and the
  hand position every cycle. In `damp`/`gravity` mode the bridge never accepts a target,
  and this tool never writes the command block.
- `cpf/src/openarm_hw/tools/shm_layout.py` (imported directly; it verifies its offsets
  against `build/shm_probe`), and the MuJoCo scene the driver loads
  (`scene_sim_openarm_bimanual.xml`). Both paths live in the `cpf:` section of
  `configs/calibration/openarm_v1_bimanual.yaml`.
- The MuJoCo world body is `openarm_body_link0` at identity, so FK of `openarm_<arm>_hand`
  is already in CPF's world frame and the solver's `base_from_camera` **is**
  `world_from_camera`. On start the tool compares its FK hand position with the driver's
  and refuses to run if they differ by more than 5 mm.

## Step by step

1. **Camera placement first.** Put the camera where it will be used and do not touch the
   tripod afterwards. Plug it into the USB 3 port before calibrating (a replug can nudge
   the tripod). Check with `rs-enumerate-devices -s` / the tool's start-up line: it prints
   `usb 3.2`. On USB 2 the tool still works (1280x720 colour at 6 fps) but the grasp
   pipeline's 640x480 depth + colour does not.
2. **Pattern.** Two kinds are supported, chosen with `--pattern`:
   - `charuco` (default, most precise): `python -m vision_pipeline.apps.openarm_hand_eye board`
     writes `calibrations/openarm_v1/charuco_board.{png,pdf}` (7x5 squares, 30 mm, 22 mm
     markers, `DICT_4X4_50`, 254 dpi, 234x174 mm page). Print at 100 %, glue flat to
     something stiff, and measure the square: span six squares with a ruler and divide.
     A purchased board works too; pass its `--squares`, `--square-mm`, `--marker-mm`,
     `--dictionary`, and `--legacy-pattern` if it was made with OpenCV older than 4.6.
   - `aruco-grid`: a plain grid of ArUco markers, e.g. six markers in 3x2. Point the camera
     at it and run `openarm_hand_eye identify`: it tries every predefined dictionary and
     prints the one that decodes, the ids row by row, the layout, an annotated image, and
     the capture command to use. Then measure with a ruler the black edge of one marker
     (`--marker-mm`) and the white gap between two neighbours (`--gap-mm`); the print scale
     is unknown otherwise. Marker corners are less precise than chessboard corners, so
     expect a somewhat higher reprojection RMS; if the held-out error misses the gate, print
     the ChArUco board.
3. **Mount.** Clamp a tab of the board in the gripper or tape it flat to the back of the
   hand. It must not move relative to the hand during the session; its offset is solved
   for, not measured. Keep it small enough not to hit the body or the other arm.
4. **Driver.** In a terminal, with the e-stop in reach:

   ```bash
   cd ~/Desktop/work/cpf/src/openarm_hw
   ./build/hold_pose_demo --config config/hold_pose.yml --arm right --mode damp \
       --duration 0 --shm /openarm_cpf
   ```

   The driver homes the arm first (3 s ramp to `q_home`), then holds it compliant. It
   exits at once if `/openarm_cpf` does not exist yet: start the capture tool first (it
   creates the segment and prints this command), or `cpf_node.py --create`.
5. **Capture.**

   ```bash
   cd ~/work/vision_pipeline && source .venv/bin/activate
   python -m vision_pipeline.apps.openarm_hand_eye capture --arm right --square-mm 30.0
   # or, with a 3x2 marker grid (values from `identify` and the ruler):
   python -m vision_pipeline.apps.openarm_hand_eye capture --arm right --pattern aruco-grid \
       --markers 3 2 --dictionary DICT_4X4_50 --first-id 0 --marker-mm 40.0 --gap-mm 10.0
   ```

   Move the arm slowly by hand and pause for a second at each station. The banner says
   why a pose is not captured (`MOVING`, `SETTLING`, `TOO SIMILAR`, `BOARD TOO OBLIQUE`,
   `NO DRIVER STATE`) and, after two poses, which world axis the rotations have covered
   least. Aim for 15-25 poses over the volume the boxes will sit in, with 20-40 deg of
   wrist rotation between stations about different axes, board facing the camera within
   60 deg. `space` captures anyway, `u` undoes, `s` solves, `q` quits and solves.
   Slow motions matter: `OVERSPEED` and `JOINT_LIMIT` trips latch and disable the
   motors, and the arm drops. Hold it at all times.
6. **Read the result.** The solve prints the camera position and heading, reprojection
   RMS, per-sample RMS, the spread of the five closed-form methods, motion diversity,
   held-out corner error (every 4th pose is held out, then the final solve uses all), and
   the difference to the tape + IMU estimate and to the other arm's calibration if it
   exists. It writes `calibrations/openarm_v1/hand_eye_<arm>_<stamp>.json`,
   `current_<arm>.json`, and `current.json`, plus the full session (images, joints,
   corners) under `calibrations/openarm_v1/sessions/`, which `solve --session DIR`
   re-solves without the robot.
7. **Check it visually.** `python -m vision_pipeline.apps.openarm_hand_eye check --arm right`
   projects the arm's link chain from FK through the calibration onto the live image
   (cyan); it must sit on the real arm as you move it. With the board still mounted it
   also draws the predicted board corners (magenta) over the detected ones and prints the
   gap in pixels, mm, and degrees. `--calibration recordings/cpf_handoffs/last_extrinsic.json`
   shows the tape estimate the same way, for comparison.
8. **Use it.** `cpf_box_handoff` now defaults to `calibrations/openarm_v1/current.json`
   when it exists (`--extrinsic tape` forces the old path). Repeat 3-7 with the board on
   the left hand when convenient; the two results should agree to a few mm, and their gap
   is reported.

## Without a marker on the hand: plate against the base, depth registration as a check

Added 2026-09-24 evening after the hand-mounted plate was judged too cumbersome.
`apps/openarm_model_calibration.py` provides:

- **`plate`** (primary, no driver needed). The marker plate lies flat on the table with
  one long edge pushed against the base plate's front edge (model: x = 0.095 m; the
  column face is at 0.030 m) and its pattern centred on the column. Contact fixes its
  yaw and x, the table fixes roll, pitch and z, and a ruler gives the rest: the distance
  from the contact edge to the nearest black tag edge (`--margin-mm`), any lateral
  offset of the pattern centre (`--lateral-mm`), and the plate thickness
  (`--thickness-mm`). PnP of the plate averaged over a few seconds then gives the
  camera pose directly; the previous estimate is used only to pick which way the long
  axis points, and the tool reports how far that estimate disagreed.

  ```bash
  python -m vision_pipeline.apps.openarm_model_calibration plate --pattern aruco-grid \
      --markers 2 3 --dictionary DICT_APRILTAG_36h11 --ids 5 4 3 2 1 0 \
      --marker-mm 60 --gap-mm 22 --margin-mm <ruler> --thickness-mm <ruler> --init tape
  ```

  Expected accuracy is set by the placement: about 0.5-1 deg of yaw from the edge
  contact and a few mm from the ruler, better than the tape and worse than hand-eye.
- **`capture` / `solve` / `residuals`** register RealSense depth to the MuJoCo visual
  meshes posed with the shared-memory joint state (robust point-to-plane ICP with the
  plate plane and a prior on the start). Measured on the bench the same evening: the
  D435i returns almost no depth on the black links and column (no preset helps; the
  high-accuracy preset halves the returns), so a static capture matches only table
  points around the base plate and the solve is unconstrained in x, y and yaw
  (observability 0.0006, leave-one-out spread 180 mm). Treat it as a **diagnostic**:
  `residuals` draws the posed model (white) over the depth points coloured by distance
  to it, which is a driver-free visual check of any calibration; with arm poses it
  can refine, but it must start from the plate or hand-eye result, never from tape.

## Acceptance gate

Proposed (from the design doc, to be confirmed on the bench): held-out corner error
<= 5 mm mean and <= 0.5 deg, reprojection RMS around 1 px at 1280x720, closed-form
methods within ~10 mm / 1 deg of the refined result, `check` overlay visibly on the arm
across the grasp volume. Kinematic error (encoder zero offsets, backlash, the 5.7 cm
homing droop is irrelevant since measured `q` is used) shows up as residuals that do not
improve with more poses.

## Camera swap notes (D455, L515)

- The result is a pose of the **colour optical frame**, so it depends on the physical
  camera and its placement, not on resolution; capture at 1280x720 and use at 640x480.
- **D455**: supported by the installed librealsense 2.56.5; same procedure. Its colour
  stream reports non-zero distortion coefficients, which the tool handles by undistorting
  the corners through librealsense before PnP (the D435I reports zeros). Its IMU also
  works with the tape fallback. Colour and depth are global shutter and the 95 mm
  baseline halves depth noise at 1 m, both good for the grasp pipeline.
- **L515**: **not supported** by librealsense 2.55 and later; the installed 2.56.5 will
  report "no device". Using it means downgrading `pyrealsense2` to 2.54.x for the whole
  pipeline, and its RGB sensor is a different, lower-resolution unit. Prefer the D455.
- Any camera change or tripod move invalidates `current.json`; recalibrate (10 minutes).
