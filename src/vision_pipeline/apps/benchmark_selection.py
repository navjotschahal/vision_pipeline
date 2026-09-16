"""Reproducible EfficientTAM / SAM 2.1 comparison for click-to-mask selection.

``quality`` runs the deterministic tracking suite
(:mod:`vision_pipeline.perception.objects.selection_evaluation`) on saved recordings,
synchronously and with every frame processed, and reports mask agreement, loss and
substitution behavior, compute time, and peak VRAM per backend.

``stream`` runs the same threaded capture -> mask -> geometry pipeline as
``select_object`` headlessly on the live camera (or a paced replay), seeds it with a
scripted click on the first processed frame, and reports sustained output rate,
capture-to-mask latency, stale and dropped frames, geometry rate, and GPU memory.

Measurements are workstation facts for this host only; paper FPS numbers are not used.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import yaml

from vision_pipeline.apps.object_selection_config import (
    DEFAULT_CONFIG_PATH,
    REPOSITORY_ROOT,
    load_object_selection_config,
)
from vision_pipeline.perception.objects.selected_geometry import (
    SelectedGeometryError,
    SelectedGeometryEstimator,
)
from vision_pipeline.perception.objects.selection import (
    Click,
    SelectionSession,
    SelectionState,
)
from vision_pipeline.perception.objects.selection_backend import (
    BACKENDS,
    build_research_segmenter,
)
from vision_pipeline.perception.objects.selection_evaluation import run_sequence, summarize
from vision_pipeline.runtime.latest_rgbd import LatestRgbdCapture
from vision_pipeline.runtime.selection_pipeline import (
    CommandKind,
    GeometryOutput,
    MaskOutput,
    OperatorCommand,
    SelectionPipeline,
)
from vision_pipeline.sources.realsense import RealSenseSource
from vision_pipeline.sources.rgbd_recording import ReplayRgbdSource, replay_rgbd

_SEQUENCE_FRAMES = 150
_MONTAGE_FRAMES = (0, 50, 85, 118, 140)
_STALE_MS = 100.0


def _synchronize() -> None:
    torch.cuda.synchronize()


def _process_gpu_mib() -> float | None:
    """This process's total GPU memory as the driver reports it (includes CUDA context)."""

    try:
        output = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError):
        return None
    for line in output.splitlines():
        pid, _, used = line.partition(",")
        if pid.strip() == str(os.getpid()):
            return float(used.strip())
    return None


def _environment() -> dict[str, Any]:
    return {
        "torch": torch.__version__,
        "cuda_device": torch.cuda.get_device_name(0),
        "numpy": np.__version__,
        "opencv": cv2.__version__,
        "backends": {name: asdict(spec) for name, spec in BACKENDS.items()},
    }


def _montage(kept: dict[int, Any], path: Path, title: str) -> None:
    tiles = []
    for index in _MONTAGE_FRAMES:
        if index not in kept:
            continue
        generated, mask = kept[index]
        image = generated.frame.color.payload.data.copy()
        image[mask] = (0.45 * image[mask] + np.array((0, 140, 0))).astype(np.uint8)
        for region, color in (
            (generated.reference, (255, 255, 0)),
            (generated.distractor, (0, 0, 255)),
        ):
            contours, _ = cv2.findContours(
                region.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            cv2.drawContours(image, contours, -1, color, 2)
        label = f"{index} {generated.script.phase.value}"
        cv2.putText(image, label, (8, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 4)
        cv2.putText(image, label, (8, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
        tiles.append(cv2.resize(image, (320, 240), interpolation=cv2.INTER_AREA))
    if not tiles:
        return
    strip = np.hstack(tiles)
    header = np.zeros((28, strip.shape[1], 3), np.uint8)
    cv2.putText(header, title, (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    cv2.imwrite(str(path), np.vstack((header, strip)))


def run_quality(args: argparse.Namespace) -> int:
    scenarios = yaml.safe_load(Path(args.scenarios).read_text())
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    policy = load_object_selection_config(args.config).policy
    seeds: dict[tuple[str, str], np.ndarray] = {}
    report: dict[str, Any] = {
        "command": " ".join(sys.argv),
        "environment": _environment(),
        "policy": asdict(policy),
        "recent_memory_frames": args.recent_memory_frames,
        "sequence_frames": _SEQUENCE_FRAMES,
        "backends": {},
    }
    sequences = []
    for sequence in scenarios["sequences"]:
        path = REPOSITORY_ROOT / sequence["recording"]
        frames = []
        for frame in replay_rgbd(path):
            frames.append(frame)
            if len(frames) == _SEQUENCE_FRAMES:
                break
        sequences.append((sequence, frames))

    for backend_name in args.backends:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        backend = build_research_segmenter(
            backend_name,
            models_root=Path(args.models_root),
            recent_memory_frames=args.recent_memory_frames,
        )
        _synchronize()
        load_s = time.perf_counter() - started
        first_frame = sequences[0][1][0]
        first_target = sequences[0][0]["targets"][0]["click"]
        first_click = Click(float(first_target[0]), float(first_target[1]))
        started = time.perf_counter()
        backend.seed(first_frame, (first_click,))
        _synchronize()
        cold_seed_ms = (time.perf_counter() - started) * 1e3
        started = time.perf_counter()
        backend.seed(first_frame, (first_click,))
        _synchronize()
        warm_seed_ms = (time.perf_counter() - started) * 1e3
        results: dict[str, Any] = {}
        for sequence, frames in sequences:
            recording = Path(sequence["recording"]).name
            for target in sequence["targets"]:
                click = Click(float(target["click"][0]), float(target["click"][1]))
                key = f"{recording}/{target['name']}"
                scores, kept = run_sequence(
                    backend,
                    frames,
                    click,
                    synchronize=_synchronize,
                    keep_frames=_MONTAGE_FRAMES,
                )
                seeds[(backend_name, key)] = kept[0][1]
                summary = summarize(scores, policy)
                results[key] = {"click": target["click"], **summary}
                _montage(
                    kept,
                    output_dir / f"{backend_name}__{recording}__{target['name']}.jpg",
                    f"{backend_name} {key}: green=mask cyan=reference red=distractor",
                )
                if args.rows:
                    results[key]["rows"] = [asdict(score) for score in scores]
                print(
                    f"{backend_name:18s} {key:40s} seed_px={summary['seed_mask_pixels']:6d} "
                    f"iou={summary['mean_iou_visible']:.3f} "
                    f"p10={summary['p10_iou_visible']:.3f} "
                    f"hidden_present={summary['hidden_frames_reported_present']}/"
                    f"{summary['hidden_frames']} "
                    f"on_distractor={summary['hidden_frames_on_distractor']} "
                    f"lost@{summary['session_first_lost_index']}"
                    f"({summary['session_first_lost_phase']}) "
                    f"track_ms={summary['track_ms_p50']:.1f}/{summary['track_ms_p95']:.1f}",
                    flush=True,
                )
        report["backends"][backend_name] = {
            "model_load_s": load_s,
            "cold_first_seed_ms": cold_seed_ms,
            "warm_seed_ms": warm_seed_ms,
            "peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
            "peak_reserved_mib": torch.cuda.max_memory_reserved() / 2**20,
            "process_gpu_mib_at_end": _process_gpu_mib(),
            "targets": results,
        }
        del backend
    if len(args.backends) == 2:
        agreement = {}
        for sequence, _ in sequences:
            recording = Path(sequence["recording"]).name
            for target in sequence["targets"]:
                key = f"{recording}/{target['name']}"
                left = seeds[(args.backends[0], key)]
                right = seeds[(args.backends[1], key)]
                union = np.count_nonzero(left | right)
                agreement[key] = float(np.count_nonzero(left & right) / union) if union else 1.0
        report["seed_mask_iou_between_backends"] = agreement
    path = output_dir / "quality.json"
    path.write_text(json.dumps(report, indent=2))
    print(f"wrote {path}")
    return 0


def run_geometry(args: argparse.Namespace) -> int:
    """M2 repeatability: track each target through a static recording and fit geometry.

    The scene does not move, so the spread of the centroid and bounds over all frames is
    the combined repeatability of mask, depth, filtering, and PCA bounds.
    """

    config = load_object_selection_config(args.config)
    scenarios = yaml.safe_load(Path(args.scenarios).read_text())
    backend = build_research_segmenter(
        args.backend,
        models_root=Path(args.models_root),
        recent_memory_frames=args.recent_memory_frames,
    )
    results: dict[str, Any] = {}
    for sequence in scenarios["sequences"]:
        path = REPOSITORY_ROOT / sequence["recording"]
        frames = list(replay_rgbd(path))
        for target in sequence["targets"]:
            key = f"{path.name}/{target['name']}"
            session = SelectionSession(backend, config.policy)
            estimator = SelectedGeometryEstimator(config.geometry)
            click = Click(float(target["click"][0]), float(target["click"][1]))
            update = session.select(frames[0], (click,))
            centroids, sizes, points, valid, milliseconds = [], [], [], [], []
            rejections: dict[str, int] = {}
            planes: dict[str, int] = {}
            lost_at = None
            for index, frame in enumerate(frames):
                if index:
                    update = session.update(frame)
                if not update.is_tracking:
                    lost_at = index
                    break
                assert update.segmentation is not None
                started = time.perf_counter()
                try:
                    estimate = estimator.estimate(
                        frame, update.segmentation, selection_id=update.selection_id
                    )
                except SelectedGeometryError as error:
                    rejections[error.reason.value] = rejections.get(error.reason.value, 0) + 1
                    continue
                milliseconds.append((time.perf_counter() - started) * 1e3)
                status = estimate.quality.support_plane_status.value
                planes[status] = planes.get(status, 0) + 1
                centroids.append(estimate.centroid_metres)
                sizes.append(sorted(estimate.observation.bounds.size_metres))
                points.append(estimate.quality.final_points)
                valid.append(estimate.quality.valid_depth_fraction)
            centroid_array = np.array(centroids)
            results[key] = {
                "click": target["click"],
                "frames": len(frames),
                "lost_at_frame": lost_at,
                "geometry_frames": len(centroids),
                "rejections": rejections,
                "support_plane_status": planes,
                "centroid_mean_metres": centroid_array.mean(axis=0).tolist() if centroids else None,
                "centroid_std_mm": (centroid_array.std(axis=0) * 1e3).tolist()
                if centroids
                else None,
                "sorted_size_median_metres": np.median(sizes, axis=0).tolist() if sizes else None,
                "sorted_size_std_mm": (np.std(sizes, axis=0) * 1e3).tolist() if sizes else None,
                "final_points_median": float(np.median(points)) if points else None,
                "valid_depth_fraction_median": float(np.median(valid)) if valid else None,
                "geometry_ms_p50_p95": (
                    np.percentile(milliseconds, [50, 95]).tolist() if milliseconds else None
                ),
            }
            summary = results[key]
            print(
                f"{key:40s} geometry={summary['geometry_frames']}/{len(frames)} "
                f"rejected={rejections} plane={planes} "
                f"centroid_std_mm={np.round(summary['centroid_std_mm'] or [], 2).tolist()} "
                f"size={np.round(summary['sorted_size_median_metres'] or [], 3).tolist()}",
                flush=True,
            )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(
            {
                "command": " ".join(sys.argv),
                "backend": args.backend,
                "geometry_config": asdict(config.geometry),
                "targets": results,
            },
            indent=2,
            default=str,
        )
    )
    print(f"wrote {output}")
    return 0


def run_startup(args: argparse.Namespace) -> int:
    """Cold start in this fresh process: model build, first seed, first track, warm seed."""

    frames = []
    for frame in replay_rgbd(args.replay):
        frames.append(frame)
        if len(frames) == 3:
            break
    click = Click(float(args.click[0]), float(args.click[1]))
    timings: dict[str, float] = {}
    started = time.perf_counter()
    backend = build_research_segmenter(
        args.backend,
        models_root=Path(args.models_root),
        recent_memory_frames=args.recent_memory_frames,
    )
    _synchronize()
    timings["model_build_s"] = time.perf_counter() - started
    steps: tuple[tuple[str, Callable[[], object]], ...] = (
        ("first_seed_ms", lambda: backend.seed(frames[0], (click,))),
        ("first_track_ms", lambda: backend.track(frames[1])),
        ("warm_seed_ms", lambda: backend.seed(frames[0], (click,))),
        ("warm_track_ms", lambda: backend.track(frames[2])),
    )
    for name, action in steps:
        started = time.perf_counter()
        action()
        _synchronize()
        timings[name] = (time.perf_counter() - started) * 1e3
    report = {
        "command": " ".join(sys.argv),
        "backend": args.backend,
        # /proc/<pid> is created with the process, before Python or torch imports.
        "process_start_to_ready_s": time.time() - os.stat(f"/proc/{os.getpid()}").st_ctime,
        **timings,
    }
    print(json.dumps(report))
    if args.output:
        with Path(args.output).open("a") as stream:
            stream.write(json.dumps(report) + "\n")
    return 0


def run_stream(args: argparse.Namespace) -> int:
    config = load_object_selection_config(args.config)
    policy = config.policy
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    backend = build_research_segmenter(
        args.backend,
        models_root=Path(args.models_root),
        recent_memory_frames=args.recent_memory_frames,
    )
    _synchronize()
    load_s = time.perf_counter() - started
    live = args.replay is None
    source: RealSenseSource | ReplayRgbdSource
    if live:
        source = RealSenseSource(config.realsense)
    else:
        source = ReplayRgbdSource(args.replay, pacing=True, preload=True)
    source.open()
    device: dict[str, Any] | None = (
        asdict(source.info) if isinstance(source, RealSenseSource) else None
    )
    clicks = tuple(Click(x, y, True) for x, y in args.click) + tuple(
        Click(x, y, False) for x, y in args.negative_click or ()
    )
    session = SelectionSession(backend, policy)
    geometry = SelectedGeometryEstimator(config.geometry)
    capture = LatestRgbdCapture(source)
    mask_rows: list[dict[str, Any]] = []
    geometry_rows: list[dict[str, Any]] = []
    selected_at: list[float] = []
    saved: list[MaskOutput] = []
    pipeline: SelectionPipeline

    def on_mask(output: MaskOutput) -> None:
        update = output.update
        if not selected_at and update.state is SelectionState.UNSELECTED:
            selected_at.append(time.monotonic())
            pipeline.submit(OperatorCommand(CommandKind.SELECT, output.frame.frameset_id, clicks))
        segmentation = update.segmentation
        mask_rows.append(
            {
                "index": output.index,
                "t": time.monotonic(),
                "frameset_id": output.frame.frameset_id,
                "color_sequence": output.frame.color.header.sequence_number,
                "state": update.state.value,
                "selection_id": update.selection_id,
                "loss_reason": update.loss_reason.value if update.loss_reason else None,
                "worker_ms": output.worker_ms,
                "host_receive_to_mask_ms": output.host_receive_to_mask_ms,
                "capture_to_mask_ms": output.capture_to_mask_ms,
                "object_score": segmentation.object_score if segmentation else None,
                "mask_pixels": segmentation.mask_pixels if segmentation else 0,
                "command_errors": list(output.command_errors),
            }
        )
        if update.is_tracking and output.index % 60 == 5:
            saved.append(output)

    def on_geometry(output: GeometryOutput) -> None:
        estimate = output.geometry
        geometry_rows.append(
            {
                "index": output.mask.index,
                "t": time.monotonic(),
                "geometry_ms": output.geometry_ms,
                "capture_to_geometry_ms": output.capture_to_geometry_ms,
                "rejection": output.rejection.value if output.rejection else None,
                "centroid_metres": list(estimate.centroid_metres) if estimate else None,
                "box_size_metres": list(estimate.observation.bounds.size_metres)
                if estimate
                else None,
                "points": estimate.quality.final_points if estimate else None,
                "valid_depth_fraction": estimate.quality.valid_depth_fraction if estimate else None,
            }
        )

    pipeline = SelectionPipeline(
        capture,
        session,
        geometry,
        live_timing=live,
        on_mask=on_mask,
        on_geometry=on_geometry,
    )
    gpu_samples: list[float] = []
    try:
        pipeline.start()
        deadline = time.monotonic() + args.duration
        while time.monotonic() < deadline and not pipeline.finished:
            time.sleep(1.0)
            sample = _process_gpu_mib()
            if sample is not None:
                gpu_samples.append(sample)
    finally:
        pipeline.stop()
        source.close()
    if pipeline.error is not None and not isinstance(pipeline.error, EOFError):
        print(f"pipeline error: {pipeline.error!r}", file=sys.stderr)
    stats = pipeline.stats()

    tracking = [row for row in mask_rows if row["state"] == SelectionState.TRACKING.value]
    steady_start = (tracking[0]["t"] + args.settle_seconds) if tracking else 0.0
    steady = [row for row in tracking if row["t"] >= steady_start]
    steady_geometry = [
        row for row in geometry_rows if row["t"] >= steady_start and row["centroid_metres"]
    ]

    def percentiles(rows: list[dict[str, Any]], key: str) -> list[float] | None:
        values = [row[key] for row in rows if row[key] is not None]
        return [float(v) for v in np.percentile(values, [50, 95])] if values else None

    def rate(rows: list[dict[str, Any]]) -> float | None:
        if len(rows) < 2:
            return None
        return float((len(rows) - 1) / (rows[-1]["t"] - rows[0]["t"]))

    capture_latencies = [row["capture_to_mask_ms"] for row in steady if row["capture_to_mask_ms"]]
    centroids = np.array([row["centroid_metres"] for row in steady_geometry])
    report = {
        "command": " ".join(sys.argv),
        "environment": _environment(),
        "backend": args.backend,
        "recent_memory_frames": args.recent_memory_frames,
        "source": "live" if live else f"paced-replay:{args.replay}",
        "device": device,
        "clicks": [asdict(click) for click in clicks],
        "duration_s": args.duration,
        "settle_seconds_excluded": args.settle_seconds,
        "model_load_s": load_s,
        "pipeline_stats": asdict(stats),
        "tracking_outputs": len(tracking),
        "steady_tracking_outputs": len(steady),
        "sustained_tracking_fps": rate(steady),
        "worker_ms_p50_p95": percentiles(steady, "worker_ms"),
        "host_receive_to_mask_ms_p50_p95": percentiles(steady, "host_receive_to_mask_ms"),
        "capture_to_mask_ms_p50_p95": percentiles(steady, "capture_to_mask_ms"),
        "stale_outputs_over_100ms": sum(value > _STALE_MS for value in capture_latencies),
        "stale_fraction_over_100ms": (
            sum(value > _STALE_MS for value in capture_latencies) / len(capture_latencies)
            if capture_latencies
            else None
        ),
        "lost_events": stats.lost_events,
        "first_loss": next((row for row in mask_rows if row["loss_reason"]), None),
        "sustained_geometry_fps": rate(steady_geometry),
        "geometry_ms_p50_p95": percentiles(steady_geometry, "geometry_ms"),
        "capture_to_geometry_ms_p50_p95": percentiles(steady_geometry, "capture_to_geometry_ms"),
        "centroid_std_metres": centroids.std(axis=0).tolist() if len(centroids) > 1 else None,
        "peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
        "peak_reserved_mib": torch.cuda.max_memory_reserved() / 2**20,
        "process_gpu_mib_max": max(gpu_samples) if gpu_samples else None,
        "latency_note": (
            "capture_to_mask uses librealsense global_time (SDK device-to-host-realtime "
            "mapping, error unmeasured); host_receive_to_mask starts after USB transfer"
        ),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps({**report, "mask_rows": mask_rows, "geometry_rows": geometry_rows}, indent=1)
    )
    for position, item in enumerate(saved[:4]):
        image = item.frame.color.payload.data.copy()
        segmentation = item.update.segmentation
        if segmentation is not None:
            image[segmentation.mask] = (
                0.45 * image[segmentation.mask] + np.array((0, 140, 0))
            ).astype(np.uint8)
        cv2.imwrite(str(output.with_suffix("")) + f"-{position}.jpg", image)
    print(
        json.dumps(
            {
                key: value
                for key, value in report.items()
                if key not in ("environment", "pipeline_stats", "device")
            },
            indent=2,
        )
    )
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--models-root", default=str(REPOSITORY_ROOT.parent))
    parser.add_argument(
        "--recent-memory-frames",
        type=int,
        help="attend to only this many recent tracked-frame memories (default: model's)",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    quality = commands.add_parser("quality", help="deterministic suite on recordings")
    quality.add_argument(
        "--backends", nargs="+", choices=sorted(BACKENDS), default=sorted(BACKENDS)
    )
    quality.add_argument(
        "--scenarios", default=str(REPOSITORY_ROOT / "configs" / "selection_benchmark.yaml")
    )
    quality.add_argument("--output-dir", required=True)
    quality.add_argument("--rows", action="store_true", help="include per-frame scores")
    geometry = commands.add_parser("geometry", help="M2 geometry repeatability on recordings")
    geometry.add_argument("--backend", choices=sorted(BACKENDS), default="efficienttam-ti")
    geometry.add_argument(
        "--scenarios", default=str(REPOSITORY_ROOT / "configs" / "selection_benchmark.yaml")
    )
    geometry.add_argument("--output", required=True)
    startup = commands.add_parser("startup", help="cold start of one backend in this process")
    startup.add_argument("--backend", choices=sorted(BACKENDS), required=True)
    startup.add_argument("--replay", required=True)
    startup.add_argument("--click", type=float, nargs=2, required=True)
    startup.add_argument("--output", help="append the JSON result to this NDJSON file")
    stream = commands.add_parser("stream", help="threaded pipeline on live camera or replay")
    stream.add_argument("--backend", choices=sorted(BACKENDS), required=True)
    stream.add_argument("--replay", help="paced replay recording instead of the live camera")
    stream.add_argument("--click", type=float, nargs=2, action="append", required=True)
    stream.add_argument("--negative-click", type=float, nargs=2, action="append")
    stream.add_argument("--duration", type=float, default=30.0)
    stream.add_argument("--settle-seconds", type=float, default=2.0)
    stream.add_argument("--output", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if not torch.cuda.is_available():
        print("CUDA is required for the selection benchmark", file=sys.stderr)
        return 2
    torch.set_num_threads(4)
    if args.command == "startup":
        return run_startup(args)
    if args.command == "geometry":
        return run_geometry(args)
    return run_quality(args) if args.command == "quality" else run_stream(args)


if __name__ == "__main__":
    raise SystemExit(main())
