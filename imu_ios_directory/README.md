# iPhone RGB-D + IMU Gateway

Native SwiftUI sensor gateway for streaming live iPhone motion data and ARKit RGB-D
frames to a host over two TCP connections.

## Payload

The IMU port sends one UTF-8 JSON object per line. Timestamps are monotonic device uptime
in nanoseconds. Core Motion and ARKit samples use this device clock, so the receiver can
associate them before mapping the iPhone clock to host time.

```json
{
  "schema": "iphone_imu.v1",
  "source_id": "iphone-...",
  "session_id": "run-uuid",
  "sequence_number": 42,
  "device_timestamp_ns": 123456789,
  "attitude_quaternion": {"x": 0, "y": 0, "z": 0, "w": 1},
  "gravity_compensated_acceleration_mps2": {"x": 0, "y": 0, "z": 0},
  "angular_velocity_radps": {"x": 0, "y": 0, "z": 0},
  "gravity_mps2": {"x": 0, "y": 0, "z": -9.80665}
}
```

The app uses `CMMotionManager` device motion at 100 Hz. iOS interrupts rear-camera ARKit
capture when the phone is locked or the app leaves the foreground, so RGB-D cannot be
streamed while locked. The app marks both links as paused and restarts capture after unlock
with a new `session_id` and ARKit world epoch. The host receiver must keep running across a
TCP disconnect or pause; it must not treat EOF as a fatal process condition.

While streaming, the app disables iOS automatic screen locking and restores the normal
idle-lock setting when streaming stops. Manually pressing the side button can still lock the
phone; use the app in the foreground for continuous RGB-D capture.

The next TCP port (`IMU port + 1`) carries RGB-D packets at up to 10 Hz. Each packet is:

1. Four-byte unsigned big-endian JSON-header length.
2. A UTF-8 `iphone_rgbd.v1` JSON header.
3. JPEG RGB bytes.
4. Tightly packed little-endian Float32 LiDAR depth in metres.
5. A tightly packed ARKit UInt8 depth-confidence plane when available.

The header also carries the raw-image camera intrinsics and ARKit camera-to-world transform,
both in column-major order, plus `rgb_orientation: "camera_buffer_native"`. The JPEG keeps
the native captured-image buffer orientation and is not rotated or resized by the app;
intrinsics therefore match the transmitted RGB dimensions. The visual-inertial transform is
the pose to use when moving each depth cloud into the ARKit world frame. Do not integrate the
raw accelerometer a second time on top of this pose.

Every IMU sample and RGB-D header contains the same random `session_id` for one app streaming
run. A new ID is generated on each start, including after an ARSession reset; receivers must
treat a changed ID as a new clock/world-frame epoch. Sequence numbers are only monotonic
within that session.

ARKit exposes LiDAR depth and confidence, but the public API does not expose a separate raw
rear-infrared intensity image. The IR sensing hardware is used internally to produce depth.

## Build and install

1. Open `IMUGateway.xcodeproj` in Xcode.
2. Select the connected iPhone 15 Pro Max as the run destination.
3. In the app target's Signing & Capabilities, select your Apple development team.
4. Run the app and allow motion access if prompted.
5. Start host listeners on consecutive ports, for example 5001 for IMU and 5002 for RGB-D.
6. Enter the Mac's reachable IP address and the IMU port, then tap **Start streaming**.

USB does not automatically make an arbitrary TCP listener reachable. For a first run, use
the Mac and iPhone on the same Wi-Fi network. A USB network path can be used when macOS
exposes a reachable interface and the receiver binds to that interface.

The receiver must treat `device_timestamp_ns` as measurement time, not TCP arrival time.
If the iPhone moves, ARKit has already fused camera and IMU observations into the streamed
camera transform. Raw IMU remains useful for inspection, recording, time-alignment tests,
and developing a separate estimator.

## Python receiver

From the repository root, install the receiver dependencies and start both listeners:

```shell
python3 -m pip install -e ".[iphone]"
vision-pipeline iphone-receiver --imu-port 5001
```

To record an analysis dataset while previewing RGB and depth:

```shell
vision-pipeline iphone-receiver \
  --imu-port 5001 \
  --record-dir recordings/iphone_run_001
```

The record directory must be new or empty. It contains lossless NumPy depth/confidence
arrays, original JPEG frames, IMU NDJSON, frame metadata, ARKit poses, both device and
host-receipt timestamps, and the nearest-IMU timestamp delta. Use `--headless` on a system
without a display.
