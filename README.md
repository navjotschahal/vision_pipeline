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
