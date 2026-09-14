# Lesson 1: A measurement is more than an array

## Learning goal

By the end of this lesson, you should be able to explain why this is unsafe:

```python
frame = camera.read()
```

and why a robotics pipeline instead needs an explicit measurement envelope:

```text
payload
+ sample and stream identity
+ capture and receipt timestamps
+ clock domains
+ coordinate frame
+ calibration revision
+ producer and compute placement
+ current memory placement
```

The envelope is reusable across RGB, depth, IR, LiDAR, IMU, force, tactile, joint,
and pose measurements. Their payload schemas will remain different.

## 1. Measurement, observation, and estimate

These terms are deliberately separate:

- A **measurement** is emitted by physical hardware: an image or IMU reading.
- An **observation** is derived: a YOLO box or a depth-deprojected 3D point.
- An **estimate** persists through time: a tracked box pose and velocity with uncertainty.

YOLO detections and DINOv2 features therefore do not belong in `MeasurementKind`.
They will receive their own derived-observation contracts in a later lesson.

## 2. Capture time is not receive time

A camera may timestamp a frame with its hardware clock. The host receives it later using
a different monotonic clock. Subtracting those two numbers is meaningless until a clock
mapping has been estimated.

`ClockDomain` identifies one uninterrupted clock epoch. Two cameras need distinct domain
IDs. A camera reset also starts a new domain, even if the device name stays the same.

When genuine capture time is unavailable, store `None`. Never rename receive time as
capture time just to make a field look complete.

## 3. A frame ID is not an image frame

Here, `FrameId("front_depth_optical")` names a 3D coordinate system. It lets later code
ask questions such as:

```text
Where is this depth point in the camera frame?
Where was that camera frame relative to the robot base at capture time?
```

The transform graph and `SE(3)` mathematics arrive in a later lesson. For now, we only
make the coordinate frame impossible to omit accidentally.

## 4. Calibration is versioned data

Calibration is not a permanent property of a camera name. It can depend on resolution,
stream profile, focus, temperature, mechanical mounting, and software configuration.
`CalibrationRef` therefore points to an immutable revision. We will later define the
referenced intrinsic and extrinsic calibration bundle.

## 5. Sensor compute, GPU compute, memory, and C++

Four ideas must not be conflated:

1. **Compute placement:** where a result was calculated.
2. **Memory placement:** where its bytes currently reside.
3. **Implementation language:** Python, C++, a device SDK, or something else.
4. **Scheduling policy:** where a future stage should run.

For example, a RealSense depth ASIC may produce depth, while the delivered SDK frame is
in host memory. Later, a C++ adapter may own that frame and expose a view to Python. A GPU
stage may upload it once and keep resize, normalization, inference, and compatible
post-processing on the GPU.

The contract records facts; it does not pretend that every RealSense camera is a general
purpose accelerator. Device capabilities will be queried by the adapter for the exact
model, firmware, stream profile, and host platform.

Our placement policy is:

| Resource | Intended responsibility |
|---|---|
| Camera/ASIC/ISP | Native depth, ISP, synchronization, or compression when supported |
| C++ | Driver callbacks, buffer ownership, bounded queues, high-rate transformations |
| GPU | Measured high-volume image/point-cloud operations and neural inference |
| Python | Reference semantics, orchestration, experiments, datasets, and evaluation |
| MCU | Hard real-time control and safety—not the perception research loop |

Moving data can cost more than a small computation. We will measure transfer count,
queue time, dropped frames, and end-to-end latency before choosing an optimized backend.

## 6. Run the real webcam source

Run:

```bash
PYTHONPATH=src python3 -m vision_pipeline webcam
```

Press `q` or Escape to stop. The application reports the requested and negotiated camera
profile, BGR pixel format, row stride, payload size, timestamp semantics, compute
placement, memory placement, and measured host-receipt rate.

OpenCV does not provide a trustworthy live-webcam acquisition timestamp, so these samples
correctly use `captured_at=None`. The host monotonic receipt time is retained separately.
The source does not invent camera intrinsics, so calibration also remains `None`.

For a bounded run without a display:

```bash
PYTHONPATH=src python3 -m vision_pipeline webcam --headless --max-frames 30
```

Do not guess a camera's best profile. Query AVFoundation's native format descriptions and
frame-rate ranges:

```bash
PYTHONPATH=src python3 -m vision_pipeline webcam --probe
```

Copy one complete supported row into raw capture parameters:

```bash
PYTHONPATH=src python3 -m vision_pipeline webcam \
  --device 0 --width 1920 --height 1080 --fps 30
```

To preserve the capability inventory with the experiment notes:

```bash
PYTHONPATH=src python3 -m vision_pipeline webcam \
  --probe-report docs/hardware/macos-camera-capabilities.md
```

The application still reports OpenCV's negotiated mode. Native capability support and
successful backend negotiation are separate facts.

Camera parameters can be versioned with an experiment in `configs/webcam.yaml`:

```bash
PYTHONPATH=src python3 -m vision_pipeline webcam --config configs/webcam.yaml
```

Configuration precedence is `built-in defaults < YAML < explicit CLI arguments`.

Then run:

```bash
pytest
```

The tests are executable statements of the contract: clocks cannot be mixed, clock
resets create new epochs, identifiers are explicit, timestamps fit signed 64-bit values,
and wrapping a payload causes no hidden copy.

## Checkpoint questions

Before Lesson 2, you should be able to answer:

1. Why can two timestamps with equal numeric values still be incomparable?
2. Why does transferring a frame to a GPU not change its capture timestamp?
3. Why can `produced_on=sensor_device` coexist with `payload_memory=host`?
4. Why is a YOLO detection an observation rather than a physical measurement?
5. Why should missing capture time be `None` rather than host receive time?

Lesson 2 will define typed image payloads and deterministic recording/replay without yet
adding YOLO, DINOv2, or multi-sensor fusion.
