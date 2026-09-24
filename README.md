# vision-pipeline

This project implements reusable sensor-to-world-belief infrastructure for robotics
experiments.

The first vertical slice is:

```text
RGB-D measurements
  -> YOLO detections
  -> DINOv2 region features
  -> persistent 3D box track
  -> robot-frame target for a dynamical-system controller
```

The models are replaceable capabilities. The reusable foundation is the meaning around
their data: clocks, timestamps, coordinate frames, calibration, provenance, memory
placement, uncertainty, recording, and replay.

Hardware placement is a standing architectural concern, not an afterthought. See
[`docs/architecture/compute-placement.md`](docs/architecture/compute-placement.md) for
the sensor/C++/GPU/Python boundary.

## Laptop webcam quick start

Install the package with its webcam, YOLO, and development dependencies:

```bash
python3 -m pip install -e ".[webcam,yolo,dev]"
```

Open the webcam pipeline:

```bash
vision-pipeline webcam
```

With no width, height, or FPS arguments, the camera keeps its native default instead of
receiving guessed settings. On macOS, inspect every AVFoundation format first, or save
it as a durable hardware note:

```bash
vision-pipeline webcam --probe
vision-pipeline webcam \
  --probe-report docs/hardware/macos-camera-capabilities.md
```

Then copy one supported row into raw parameters. No selection policy chooses a mode on
your behalf:

```bash
vision-pipeline webcam --device 0 --width 1920 --height 1080 --fps 30
```

The same parameters can live in the checked-in YAML file:

```bash
vision-pipeline webcam --config configs/webcam.yaml
```

The checked-in configuration enables live YOLO detection on the Apple GPU through
the explicit PyTorch backend on MPS. It owns preprocessing, the raw model forward pass,
decoding, and NMS. Select the high-level reference backend for parity comparisons:

```bash
vision-pipeline webcam \
  --config configs/webcam.yaml \
  --detector-backend ultralytics
```

The first inference initializes accelerator work, so inspect p50/p95 timing over
multiple frames instead of treating the first frame as steady-state performance.
Override the compute device without editing YAML:

```bash
vision-pipeline webcam \
  --config configs/webcam.yaml \
  --inference-device cpu
```

Live operation defaults to the bounded `latest` scheduler. Camera capture remains at
the requested rate while one worker runs inference; if that worker falls behind, the
single pending slot is replaced with the newest frame and the stale-frame drop is
reported. This avoids an ever-growing latency backlog. Use synchronous scheduling when
comparing inference backends under the simpler capture-then-infer loop:

```bash
vision-pipeline webcam \
  --config configs/webcam.yaml \
  --detection-mode synchronous
```

Detection overlays in `latest` mode are always drawn on the exact source frame used for
inference. The preview reports both capture and displayed sequence numbers so latency
is visible instead of silently applying old boxes to a new image.

Both implementations are isolated from the model-independent detection contracts. The
research backend currently uses Ultralytics only to reconstruct the checkpoint's
`torch.nn.Module`; all runtime inference stages after loading are explicit in our code.
Review Ultralytics' AGPL-3.0 and enterprise licensing choices before using it outside an
appropriate research or open-source setting.

Explicit CLI arguments override YAML. For example, this keeps the YAML configuration but
runs a bounded headless measurement:

```bash
vision-pipeline webcam \
  --config configs/webcam.yaml \
  --headless \
  --max-frames 300
```

Each explicit CLI value overrides the corresponding YAML value. After opening the
camera, the application prints the requested and negotiated values and warns if the
driver substituted a width, height, or FPS.

Hardware discovery follows the concrete sensor and operating system being used. We add
capability notes only after probing that hardware or reviewing its exact documentation.
See [`docs/hardware/capability-inventory.md`](docs/hardware/capability-inventory.md) for
the evidence workflow and currently known hardware.

Press `q` or Escape to stop. For a bounded, display-free probe:

```bash
vision-pipeline webcam --headless --max-frames 30
```

Without installing the console entry point, run:

```bash
PYTHONPATH=src python3 -m vision_pipeline webcam
```

On macOS, grant Camera permission to the terminal or IDE when prompted. Device index `0`
is the default; if Continuity Camera changes ordering, try `--device 1`.

The adapter implementation lives in
[`src/vision_pipeline/sources/opencv_webcam.py`](src/vision_pipeline/sources/opencv_webcam.py).
The foundational reasoning remains in
[`docs/lesson-01-measurements.md`](docs/lesson-01-measurements.md), separate from runtime
code.

Run verification with:

```bash
pytest
ruff check .
mypy
```

## Linux NVIDIA + RealSense setup

The Linux workstation bootstrap creates a repository-local `.venv`, installs its own
managed Python (so it does not alter the system Python), installs a CUDA PyTorch build,
and installs the RealSense Python SDK plus this project:

```bash
cd ~/work/vision_pipeline
./scripts/bootstrap_linux_gpu.sh
source .venv/bin/activate
```

The default versions are deliberately pinned for a reproducible first deployment:
Python 3.12, PyTorch 2.7.1 with CUDA 12.6 wheels, and `pyrealsense2` 2.56.5 to match the
2.56.5 librealsense tools on the workstation. The script finishes by allocating a CUDA
tensor and enumerating the attached RealSense devices. It is safe to rerun.

Version pins and required hardware checks can be overridden explicitly, for example:

```bash
VISION_REQUIRE_REALSENSE=0 VISION_PYTHON_VERSION=3.12 \
  ./scripts/bootstrap_linux_gpu.sh
```

The base environment is for capture, segmentation/detection, point-cloud geometry, and
the current tabletop estimator. Research grasp-generation systems with compiled CUDA
extensions should use a separate environment or container so their narrow dependency
pins cannot destabilize camera capture.

The implementation handoff for the on-host Codex agent is in
[`LINUX_AGENT_BRIEF.md`](LINUX_AGENT_BRIEF.md).

# Human pose links for arm retargeting

The `human-pose` command runs an optional YOLO COCO-17 pose checkpoint on a webcam,
draws selected links, and can write timestamped JSONL observations. Install with
`python3 -m pip install -e '.[webcam,yolo]'`. The first run may download the
checkpoint.

```sh
vision-pipeline human-pose --chain left_arm:left_shoulder,left_elbow,left_wrist \
  --chain right_arm:right_shoulder,right_elbow,right_wrist
vision-pipeline human-pose --headless --max-frames 100 --output pose-links.jsonl
```

`HumanPoseLinkPipeline` combines any `HumanPoseEstimator` with selected `LimbChain`
definitions. With aligned depth in metres and matching camera intrinsics it emits
metric `HumanPose3D` joints and valid 3D links. Without aligned depth it emits only
image coordinates. A single RGB camera does not supply metric 3D position through
this backend.

`SingleArmRetargeter` or `BimanualRetargeter` consumes the metric pose and maps each configured
`ArmRetargetingConfig.operator_chain` from its first to last joint. The default
chain is shoulder, elbow, wrist. Engage captures a neutral operator and robot TCP
pose. The target includes a base frame, workspace and speed limits, confidence,
and an expiry time. A downstream controller must implement IK, joint and collision
limits, interpolation, and an independent stop/deadman mechanism. The current
bimanual retargeter requires both arms to be valid. The webcam viewer never sends commands.

## Webcam to KUKA MuJoCo demo

The optional demo reuses the KUKA iiwa14 MJCF scene, home pose, torque limits,
`MujocoRobotView`, and `JointImpedanceController` from the sibling CPF PoC. It
does not modify that checkout or run the CPF contact filter. Install the optional
dependencies, then launch from this repository:

```sh
python3 -m pip install -e '.[kuka-sim]'
./.venv/bin/mjpython -m vision_pipeline kuka-webcam-sim \
  --poc-root ../Navjot_IS/poc --camera 0
```

On macOS, MuJoCo's passive viewer requires `mjpython`. The demo opens **two
windows**: MuJoCo and a camera preview with the tracked arm drawn over it. In
either window, press `c` to capture a neutral clutch pose, `h` to disarm and hold,
or `q` to quit.
Exactly one person must be visible to engage. If you use another environment,
invoke its `mjpython` instead of the path above. Run a bounded no-window check
with `--headless --max-seconds 10`. Pose status also appears in the terminal.

This webcam-only mapping uses the selected right shoulder, elbow, and wrist.
Relative wrist movement across the image controls the simulated KUKA tool's
**Y** coordinate; vertical image movement controls **Z**. The **X target stays
fixed** at the clutch pose because one webcam cannot measure metric depth. Tool
orientation is not controlled by this demo.
The target is bounded and speed-limited; a damped positional IK solver converts
it to KUKA joint targets, which the PoC impedance controller tracks. Missing,
stale, low-confidence, or multi-person observations hold the current joint
position. This is a simulation experiment and is not wired to physical hardware.

The live YOLO backend selects a detection index and does **not** maintain person
identity when people enter, leave, or cross. Do not use its `person_id` alone to
engage a physical robot. Add an operator tracker or explicit selection and reject
identity uncertainty before forwarding any target.

Meta research options (checked September 2026):

- [Sapiens2](https://github.com/facebookresearch/sapiens2) (ICLR 2026) supplies
  308 whole-body 2D keypoints, including detailed hands. Its documented inference
  uses 1024 x 768 crops and large 0.4B to 5B models, so measure latency on the
  deployment computer before choosing it for teleoperation.
- [SAM 3D Body](https://github.com/facebookresearch/sam-3d-body) (CVPR 2026)
  recovers a 3D body mesh and joints from a single image. Its model weights require
  Hugging Face access and its predictions remain model estimates, not calibrated
  metric RGB-D measurements. It is useful for an offline/slow 3D backend, subject
  to its [SAM License](https://github.com/facebookresearch/sam-3d-body/blob/main/LICENSE).
- [RTMW](https://arxiv.org/abs/2407.08634) is an open real-time whole-body 2D/3D
  alternative available in [MMPose](https://github.com/open-mmlab/mmpose).

The included live backend is deliberately small and replaceable. None of these
papers establishes end-to-end safe Franka control; that requires testing the full
camera, calibration, network, and robot loop.
