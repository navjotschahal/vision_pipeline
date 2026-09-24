"""Live pose and selected limb-link preview; emits no robot commands."""

from __future__ import annotations

import json
from pathlib import Path

import cv2

from vision_pipeline.perception.pose.chains import LimbChain, visible_links
from vision_pipeline.perception.pose.yolo import YoloPoseConfig, YoloPoseEstimator
from vision_pipeline.sources.opencv_webcam import OpenCvWebcam, WebcamConfig


def run_human_pose_viewer(
    *,
    camera: int,
    model: str,
    device: str | None,
    person_index: int,
    chains: tuple[LimbChain, ...],
    max_frames: int,
    headless: bool,
    output: Path | None,
) -> int:
    if max_frames < 0:
        raise ValueError("max_frames must be non-negative")
    estimator = YoloPoseEstimator(
        YoloPoseConfig(model=model, device=device, person_index=person_index)
    )
    stream = output.open("w", encoding="utf-8") if output is not None else None
    try:
        with OpenCvWebcam(WebcamConfig(device_index=camera)) as webcam:
            count = 0
            while max_frames == 0 or count < max_frames:
                sample = webcam.read()
                pose = estimator.estimate(sample.payload, sample.header)
                preview = sample.payload.data.copy() if not headless else None
                record: dict[str, object] = {
                    "sample_id": sample.header.sample_id,
                    "received_ns": sample.header.received_at.nanoseconds,
                    "frame_id": sample.header.frame_id.value,
                    "person_id": pose.person_id if pose is not None else None,
                    "chains": {},
                }
                selected: dict[str, object] = {}
                for chain in chains:
                    links = visible_links(pose, chain) if pose is not None else ()
                    entries = []
                    if pose is not None:
                        for start, end in links:
                            a = pose.get(start)
                            b = pose.get(end)
                            assert a is not None and b is not None
                            entries.append(
                                {
                                    "from": start.value,
                                    "to": end.value,
                                    "from_xy": a.image_xy,
                                    "to_xy": b.image_xy,
                                }
                            )
                            if preview is None:
                                continue
                            p = tuple(round(v) for v in a.image_xy)
                            q = tuple(round(v) for v in b.image_xy)
                            cv2.line(preview, p, q, (0, 220, 255), 3)
                            cv2.circle(preview, p, 5, (0, 255, 0), -1)
                            cv2.circle(preview, q, 5, (0, 255, 0), -1)
                    selected[chain.name] = entries
                record["chains"] = selected
                if stream is not None:
                    stream.write(json.dumps(record) + "\n")
                if preview is not None:
                    cv2.imshow("human pose links (q to quit)", preview)
                    if cv2.waitKey(1) & 0xFF in (27, ord("q")):
                        break
                count += 1
    finally:
        if stream is not None:
            stream.close()
        if not headless:
            cv2.destroyAllWindows()
    return 0
