# Click selection and selected-object geometry (M0–M2)

| Document | Contents |
|---|---|
| [M0.md](M0.md) | Environment, camera link, test/lint baselines, geometric estimator, recordings |
| [M1-backend-decision.md](M1-backend-decision.md) | EfficientTAM-Ti vs SAM 2.1 Hiera Tiny measurements and decision |
| [M2-selected-geometry.md](M2-selected-geometry.md) | Mask + depth to metric cloud, centroid, bounds, quality, visualization |
| [preliminary/](preliminary/README.md) | Superseded first-session runs, kept for traceability |

## Commands

All commands run from the repository root with `source .venv/bin/activate`.

```bash
# One-time: pinned EfficientTAM/SAM 2 sources and SHA-256-verified checkpoints next to
# the repository (verifies only, when already present).
./scripts/setup_selection_models.sh

# Live selection UI: left click selects / adds positive, right click adds negative,
# c clears, q quits. Positions are color-optical-frame metres; no robot frame.
python -m vision_pipeline.apps.select_object
python -m vision_pipeline.apps.select_object --ndjson /tmp/selected.ndjson   # dry-run records
python -m vision_pipeline.apps.select_object --replay recordings/d435i_table_static
python -m vision_pipeline.apps.select_object --no-display --click 218 432 --duration 15

# CPF bimanual box grasp handoff (tape + IMU extrinsic; see configs/calibration/camera_world_tape.yaml).
python -m vision_pipeline.apps.cpf_box_handoff --measure   # crosshair to tape the heading aim point
python -m vision_pipeline.apps.cpf_box_handoff             # click box, press s when READY
#   -> recordings/cpf_handoffs/cpf_box_*.json and the grasp_planner.py --object-pos/--half-width args

# Deterministic RGB-D recording (never overwrites) and its diagnostics.
python -m vision_pipeline.apps.record_rgbd recordings/<new-name> --frames 300

# Backend comparison and geometry measurements.
python -m vision_pipeline.apps.benchmark_selection quality --output-dir docs/selection/m1-quality --rows
python -m vision_pipeline.apps.benchmark_selection startup --backend efficienttam-ti \
  --replay recordings/d435i_table_static --click 218 432 --output docs/selection/m1-startup.ndjson
python -m vision_pipeline.apps.benchmark_selection stream --backend efficienttam-ti \
  --click 218 432 --duration 30 --output docs/selection/m1-stream/live-<link>-efficienttam-ti.json
python -m vision_pipeline.apps.benchmark_selection stream --backend efficienttam-ti \
  --replay recordings/d435i_table_static --click 218 432 --duration 12 \
  --output docs/selection/m1-stream/replay-efficienttam-ti.json
python -m vision_pipeline.apps.benchmark_selection geometry \
  --output docs/selection/m2-geometry/geometry-efficienttam-ti.json

# Tests. The CUDA adapter test is opt-in.
pytest -q tests/test_rgbd_recording.py tests/test_selection.py tests/test_selected_geometry.py \
  tests/test_selection_pipeline.py tests/test_selection_evaluation.py
VISION_SELECTION_GPU_TESTS=1 pytest -q tests/test_selection_backend_gpu.py
```

Recordings live under the git-ignored `recordings/` directory. The quality suite and the
repeatability benchmark expect `recordings/d435i_table_static` and `recordings/d435i_m0`
(see `configs/selection_benchmark.yaml` for the click targets).

## Code map

| Module | Role |
|---|---|
| `sources/rgbd_recording.py` | `RgbdRecorder`, `replay_rgbd`, `ReplayRgbdSource` (optionally paced) |
| `runtime/latest_rgbd.py` | `LatestSlot` one-value handoff, `LatestRgbdCapture` thread |
| `perception/objects/selection.py` | `PromptableVideoSegmenter` contract, `Click`/`map_click`, `SelectionSession` UNSELECTED/TRACKING/LOST policy |
| `perception/objects/selection_backend.py` | Pinned EfficientTAM/SAM 2 streaming adapters (imported only on demand) |
| `perception/objects/selected_geometry.py` | Mask + aligned depth → `SelectedObjectGeometry` |
| `perception/objects/selection_evaluation.py` | Deterministic tracking-quality perturbation suite |
| `runtime/selection_pipeline.py` | Capture → mask → geometry worker threads, operator commands, latency stats |
| `apps/select_object.py` | One-command live/replay UI and headless runner |
| `apps/benchmark_selection.py` | `quality`, `startup`, `stream`, `geometry` measurements |
| `apps/object_selection_config.py`, `configs/object_selection.yaml` | Configuration |

The existing geometric baseline (`tabletop.py`, `box_estimator.py`,
`tabletop_estimator.py`, `tabletop_viewer.py`) and its contracts are unchanged.
