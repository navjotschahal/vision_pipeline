# Hardware capability notes

This directory records only hardware that we have actually connected and probed, or
whose exact model documentation we have reviewed. It is a reference for the parameters
and processing capabilities available to our experiments, not a catalogue of devices we
might use later.

For each newly connected sensor:

1. Identify the exact model, firmware, host operating system, and connection type.
2. Use the device or operating-system API to probe its live capabilities when possible.
3. Review the exact vendor documentation for anything the probe cannot report.
4. Save the observed modes, device-side compute features, constraints, source links, and
   probe date in a Markdown note here.
5. Copy a chosen raw mode into the experiment YAML and verify the runtime-negotiated
   values.

Do not add assumed specifications for hardware we have not inspected. A product-family
maximum from a web page is not enough to describe a connected device: firmware, drivers,
host ports, cables, and operating systems can change what is actually available.

## Known hardware

- [Current macOS camera capability snapshot](macos-camera-capabilities.md), generated
  from the cameras currently exposed by AVFoundation.
- [RealSense D435i USB link and achievable depth+color rate](d435i-usb3-fps-latency.md),
  probed on `figueroa-lab11` with `pyrealsense2`.

## Current probe command

Generate the static inventory:

```bash
vision-pipeline webcam \
  --probe-report docs/hardware/macos-camera-capabilities.md
```

Read the table, choose one row, then configure the exact values:

```yaml
webcam:
  device_index: 0
  width: 1920
  height: 1080
  fps: 30
```

This command is macOS-specific. When the repository is run with a sensor on another
operating system, we will inspect that concrete setup and add the appropriate probe then.
The common capture configuration remains raw `device_index`, `width`, `height`, and
`fps` values.

At runtime, verify the printed `requested=...` and `reported=...` values. A mode exposed
by the native API does not guarantee that decoding, display, model inference, or
downstream copies will sustain the same frame rate.
