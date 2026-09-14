# macOS camera capability snapshot

Generated: `2026-09-11T23:10:25.746805+00:00`

Probe method: macOS AVFoundation device formats and their supported frame-rate ranges.

This is an inventory, not a recommendation. Choose one row and copy its `width`, `height`, and a supported `fps` into `configs/webcam.yaml`. OpenCV still reports the mode actually negotiated by the device.

## Device 0: FaceTime HD Camera

- OpenCV/AVFoundation index: `0`
- Stable native ID: `1FD4B3A2-236E-492B-8CE5-255DD288CE50`
- Position: `unspecified`
- Status: connected

| Width | Height | Minimum FPS | Maximum FPS | Native format |
| ---: | ---: | ---: | ---: | :--- |
| 640 | 480 | 15 | 30 | `420v` |
| 1080 | 1920 | 15 | 30 | `420v` |
| 1280 | 720 | 15 | 30 | `420v` |
| 1328 | 1760 | 15 | 30 | `420v` |
| 1552 | 1552 | 15 | 30 | `420v` |
| 1760 | 1328 | 15 | 30 | `420v` |
| 1920 | 1080 | 15 | 30 | `420v` |

## Device 1: iPhone 15 Pro Max Camera

- OpenCV/AVFoundation index: `1`
- Stable native ID: `CAE4B489-B27E-4DB9-AE8E-BC2C00000001`
- Position: `unspecified`
- Status: connected

| Width | Height | Minimum FPS | Maximum FPS | Native format |
| ---: | ---: | ---: | ---: | :--- |
| 640 | 480 | 1 | 30 | `420v` |
| 640 | 480 | 1 | 60 | `420v` |
| 1280 | 720 | 1 | 30 | `420v` |
| 1280 | 720 | 1 | 60 | `420v` |
| 1920 | 1080 | 1 | 30 | `420v` |
| 1920 | 1080 | 1 | 60 | `420v` |
| 1920 | 1440 | 1 | 30 | `420v` |
| 1920 | 1440 | 1 | 60 | `420v` |

## Raw configuration fields

```yaml
webcam:
  device_index: 0  # choose a device above
  width: 1920      # copy from the same table row
  height: 1080
  fps: 30
```

Regenerate this snapshot after an OS update, camera firmware change, or hardware change. The native format identifies capture transport; the current OpenCV adapter emits decoded BGR8 frames in host memory.
