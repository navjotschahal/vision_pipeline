# RealSense D435i: USB link and achievable depth+color rate

Probed: `2026-09-15`, host `figueroa-lab11` (Ubuntu, kernel `5.11.0-rt7` PREEMPT_RT).
Device: RealSense D435I, serial `146322071961`, firmware `5.17.3.10`.

## USB link

The camera started this session on a USB 2.0 High-Speed port (Bus 003, root hub
`480M`): `pyrealsense2` reported `usb_type_descriptor = 2.1`, `lsusb -t` showed the
device at `480M` under the `xhci_hcd/12p` USB-2 root hub, and `dmesg` recorded
`usb 3-6: new high-speed USB device` at connect time. A first replug into a
USB-C cable/port did not change this — same bus, same `2.1` descriptor, same
`new high-speed USB device` in `dmesg`. USB-C connector shape does not imply
SuperSpeed wiring; the first cable/port was USB 2 only.

After a second replug (USB-C cable into a different USB-C port):

- `pyrealsense2`: `usb_type_descriptor = 3.2`
- `lsusb -t`: device on Bus 004 (root hub `10000M`), port 2, negotiated `5000M`
  on every interface
- `dmesg`: `usb 4-2: new SuperSpeed Gen 1 USB device number 42 using xhci_hcd`

The link is now USB 3 (SuperSpeed Gen 1, 5 Gbps). Confirmed the same physical
device throughout by `camera_info.name` / `serial_number` / `firmware_version`
matching on every probe (the USB `iSerialNumber` string surfaced by `dmesg`,
e.g. `150423062086`, differs from the `librealsense` camera serial
`146322071961` — a known quirk on this device, not a different unit; the
`pyrealsense2` serial matched the brief's stated `146322071961` on every check).

## Achievable depth+color rate and latency

Method: `RealSenseSource` (`src/vision_pipeline/sources/realsense.py`) opened
directly, so the numbers include the driver's own copy-out and depth-to-color
alignment, not a raw SDK loop. 15 frames warm-up, then a tight `read()` loop for
6 s per mode. `global_time_enabled` was confirmed `True` (the default) on both
the Stereo Module and RGB Camera sensors, which puts `frame.get_timestamp()` in
the `global_time` clock domain — synced to the host wall clock — so
capture-to-host latency is `time.time_ns()` at `read()` return minus the
frame's `captured_at` timestamp, no separate clock offset calibration needed.

Matched depth+color resolution/fps (both in the brief's valid-mode lists):

| Depth | Color | Requested fps | Measured fps (6s) | Errors | Latency mean / median / min / max |
|---|---|---:|---:|---:|---|
| 640x480 | 640x480 | 6 | 5.99 | 0 | 166.0 / 165.9 / 165.6 / 167.3 ms |
| 640x480 | 640x480 | 15 | 14.99 | 0 | 67.7 / 68.3 / 62.7 / 71.1 ms |
| 640x480 | 640x480 | 30 | 29.98 | 0 | 33.8 / 36.0 / 24.5 / 40.5 ms |

All three sustain their full requested rate with zero read errors over the 6 s
window — a sharp contrast to the USB 2 link (6 fps measured 2.0 fps, 15 fps
timed out). Latency tracks one frame period at each rate (readout + USB
transfer dominate, not queuing), so it scales down as fps goes up.

One asymmetric combo was tried and rejected by the SDK before any timing was
taken: depth `848x480@8` + color `640x480@6` → `RuntimeError("Couldn't resolve
requests")`. Not pursued — librealsense does not consider it a valid joint
profile on this device, and the matched-resolution modes above already give a
clean answer.

## Recommended mode for phase 1

**Depth 640x480@30 + color 640x480@30** — already `RealSenseConfig`'s default.
Sustains 29.98 fps measured with 0 errors, mean end-to-end latency 33.8 ms.
That is far inside the "centimetre position, a few degrees of yaw" bar for a
box that is static or hand-moved before contact.

## Gate

- Link reports 3.x: yes, `usb_type_descriptor = 3.2`, SuperSpeed Gen 1 (5000M) confirmed by `pyrealsense2`, `lsusb -t`, and `dmesg`.
- Stated mode with measured fps and latency: depth+color 640x480@30, 29.98 fps sustained, 33.8 ms mean capture-to-host latency.
