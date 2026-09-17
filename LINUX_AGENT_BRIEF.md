# Linux RGB-D Object Selection and Manipulation Perception Brief

## Mission

Build and validate a stable, research-backed perception path for selecting one object
on a tabletop from an Intel RealSense RGB-D stream and producing geometry suitable for
robot manipulation. Run this work on the Linux workstation with the NVIDIA GPU and
physical camera. Do not spend time enabling RealSense on macOS; the Mac development
path uses the iPhone or webcam sources.

The first operator interface is deliberately simple: select an object with mouse
clicks. Text-conditioned selection is a later optional capability, not a prerequisite.

## Finalized scope for this agent run

Deliver M0, M1, and M2 only:

1. Measure the existing RealSense baseline and create deterministic RGB-D replay.
2. Compare EfficientTAM-Ti with SAM 2.1 Hiera Tiny, then retain the better measured
   click-to-mask tracker behind a backend-independent contract.
3. Turn its mask and aligned depth into a timestamped metric object point cloud,
   centroid, oriented bounds, confidence diagnostics, and live visualization.

Do not implement YOLO, text prompting, FoundationPose, grasp generation, or robot
motion in this run. Those are later gated stages. Do not purge the existing geometric
baseline or measurement contracts.

## Workstation and repository

- SSH host alias: `figueroa_openarmpc_airpenn`
- Repository: `/home/nschahal/work/vision_pipeline`
- GPU: NVIDIA GeForce RTX 3070, 8 GiB, compute capability 8.6
- Attached and verified camera: Intel RealSense D435I, serial `146322071961`, firmware
  `5.17.3.10`
- Host librealsense tools: 2.56.5
- Repo environment: Python 3.12.14, PyTorch 2.7.1+cu126, torchvision 0.22.1+cu126,
  pyrealsense2 2.56.5.9235

Start with:

```bash
cd /home/nschahal/work/vision_pipeline
source .venv/bin/activate
nvidia-smi
python -c 'import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name())'
python -c 'import pyrealsense2 as rs; print([d.get_info(rs.camera_info.name) for d in rs.context().query_devices()])'
```

If the environment is missing or damaged, rerun `./scripts/bootstrap_linux_gpu.sh`.
It must remain repo-local and must not replace the system Python.

Baseline recorded on 2026-09-16 after the environment install:

- `pytest -q`: 129 passed, 1 failed. The inherited failure is
  `tests/test_human_pose.py::test_depth_lifter_rejects_discontinuous_and_missing_depth`.
- `ruff check .`: 2 inherited findings, both in `sources/realsense.py` (`B008` and
  `SIM105`).
- `mypy`: 12 inherited errors in `runtime/shm_seqlock.py` and
  `runtime/box_view_channel.py`.

Do not attribute those known baseline findings to new selection work, but do not add
new failures. Fixing one while touching the same code is welcome if behavior remains
covered.

## Existing code to preserve and evaluate

Do not purge useful contracts merely because an estimator is replaced. Inspect these
first:

- `src/vision_pipeline/sources/realsense.py`: aligned RGB-D acquisition, intrinsics,
  timestamps, provenance, and stream negotiation.
- `src/vision_pipeline/perception/objects/contracts.py`: object observation contracts.
- `src/vision_pipeline/perception/objects/geometry.py`: mask/box-selected depth
  deprojection and 3D geometry.
- `src/vision_pipeline/perception/objects/tabletop.py`: detector-free plane removal,
  clustering, oriented bounds, and current geometric baseline.
- `src/vision_pipeline/perception/objects/box_estimator.py`: estimator seam.
- `src/vision_pipeline/apps/tabletop_estimator.py` and `tabletop_viewer.py`: live
  producer/viewer split and shared-memory diagnostics.
- `configs/tabletop_box.yaml`: currently hard-coded for the verified D435I; workspace
  bounds are explicitly uncalibrated placeholders.
- Relevant tests: `test_object_perception.py`, `test_tabletop_box.py`,
  `test_box_estimator.py`, `test_pointcloud.py`, and `test_spatial.py`.

The current largest-cluster estimator is a useful non-learned baseline but is not a
reliable object identity mechanism in clutter. Keep it as a fallback and diagnostic,
not as the final selection policy.

## Recommended architecture

Use the smallest dependable stack that solves each distinct problem:

```text
RealSense aligned color + depth
  -> operator positive/negative mouse clicks
  -> promptable mask and temporal propagation on CUDA
  -> mask/depth validity/ROI intersection
  -> selected metric point cloud in the color optical frame
  -> table-plane removal + outlier filtering
  -> object geometry and confidence
  -> either object pose or object-conditioned grasp candidates
  -> camera-to-robot calibrated transform + safety checks
```

### Selection: EfficientTAM first, SAM 2.1 as the reference

Implement the integration behind a small `PromptableVideoSegmenter` protocol. Benchmark
**EfficientTAM-Ti** and **SAM 2.1 Hiera Tiny** on the same saved RealSense sequences and
live 640x480 stream. EfficientTAM-Ti is the default candidate because it was designed
for low-latency tracking; SAM 2.1 Tiny is the reference/fallback because it has the
larger official ecosystem. Do not integrate YOLO into this path: a user click already
identifies the target, while a box detector does not supply the precise mask needed for
depth selection.

A left click adds a positive prompt; right click adds a negative prompt; clear/reselect
must be available. Initialize from a click on the live color image and propagate the
mask over subsequent frames. Re-prompt after tracking loss instead of silently
switching objects.

Primary sources:

- Efficient Track Anything (ICCV 2025):
  <https://openaccess.thecvf.com/content/ICCV2025/html/Xiong_Efficient_Track_Anything_ICCV_2025_paper.html>
- EfficientTAM implementation: <https://github.com/yformer/EfficientTAM>
- Paper (ICLR 2025): <https://openreview.net/pdf?id=Ha6RTeWMd0>
- SAM 2 implementation and benchmark: <https://github.com/facebookresearch/sam2>
- Meta research overview: <https://ai.meta.com/research/sam2/>

Keep external repositories and checkpoint acquisition outside core contracts, record
exact revisions and checkpoint checksums, and avoid importing either backend when the
capability is disabled. Install compiled research dependencies in an isolated
environment/service if they conflict with the capture `.venv`.

The M1 benchmark must report warm and cold startup, capture-to-mask p50/p95 latency,
sustained output FPS, dropped/stale frames, peak VRAM, and tracking failures. Target at
least 15 Hz sustained output with p95 capture-to-mask latency below 100 ms and no
unbounded queue on the RTX 3070. Prefer segmentation quality and explicit `LOST`
behavior over a small FPS advantage. These are project gates, not claims from a paper.

Text-conditioned selection is out of scope for M1/M2. If it becomes necessary later,
evaluate SAM 3 as an optional backend; do not add GroundingDINO or YOLO merely to claim
text selection now.

### 3D output: distinguish geometry, object pose, and grasp pose

These are not interchangeable:

- For an unknown object with no canonical model, produce the selected point cloud,
  centroid, oriented bounds, surface normals, and covariance. A unique semantic 6D
  object pose is generally undefined because there is no agreed object coordinate
  frame and symmetric objects have ambiguous rotations.
- For a known object with a CAD model, use FoundationPose for 6D pose estimation and
  tracking, seeded by the selected mask. It supports novel model-based objects given a
  CAD model; its model-free mode requires reference capture and is a separate workflow.
- If the robot's immediate goal is pickup, prefer object-conditioned 6-DoF grasp
  proposals over inventing an object coordinate frame. Filter candidate contacts by
  the selected mask/point cloud, table collision, gripper width, reachability, and IK.

Primary sources:

- FoundationPose, CVPR 2024: <https://github.com/NVlabs/FoundationPose>
- M2T2, CoRL 2023: <https://github.com/NVlabs/M2T2>
- GraspGen, ICRA 2026: <https://github.com/NVlabs/GraspGen>
- AnyGrasp, IEEE Transactions on Robotics 2023:
  <https://ieeexplore.ieee.org/document/10167687>
- Contact-GraspNet, ICRA 2021: <https://github.com/NVlabs/contact_graspnet>
- GraspNet-1Billion, CVPR 2020:
  <https://openaccess.thecvf.com/content_CVPR_2020/papers/Fang_GraspNet-1Billion_A_Large-Scale_Benchmark_for_General_Object_Grasping_CVPR_2020_paper.pdf>

FoundationPose and grasp-generation repositories have compiled, narrowly pinned CUDA
dependencies. Run each as a separate environment/container or local service. Do not
install them into the capture `.venv`. For later unknown-object pickup, evaluate
GraspGen first only if its supported gripper and NVIDIA license fit the project;
otherwise evaluate M2T2. AnyGrasp has strong journal results but a machine-licensed
binary SDK, and Contact-GraspNet has an old TensorFlow/CUDA stack, so neither is the
default integration target.

## Implementation milestones

### M0 — establish the measured baseline

1. Record `git status`, current commit, `nvidia-smi`, librealsense version, camera
   identity, USB descriptor, and negotiated stream modes.
2. Run the focused tests and then the full suite. Record existing failures separately
   from regressions.
3. Run the current estimator headlessly for a bounded number of frames:

   ```bash
   python -m vision_pipeline.apps.tabletop_estimator --no-publish --max-frames 300
   ```

4. Save measured capture rate, dropped/failed frames, latency if available, and a short
   RGB-D recording for deterministic replay. Do not tune against only one live scene.

### M1 — click-to-mask live capability

1. Define the model-independent `PromptableVideoSegmenter` contract and deterministic
   replay benchmark.
2. Benchmark EfficientTAM-Ti and SAM 2.1 Hiera Tiny on `cuda:0`; record the metrics
   specified above and select the backend from evidence.
3. Add a standalone live RGB-D selection app before coupling it to robot control.
4. Display the exact color frame associated with the aligned depth frame.
5. Collect positive/negative clicks and show a translucent mask overlay.
6. Propagate the selected instance through video with the selected backend.
7. Expose states such as `UNSELECTED`, `TRACKING`, and `LOST`; never substitute a new
   object automatically after loss.
8. Keep capture fresh with a bounded latest-frame queue; do not build an inference
   backlog.

### M2 — selected object in metric 3D

1. Reuse `extract_object_point_cloud` and the RealSense calibration contracts.
2. Reject invalid/zero depth and intersect the SAM mask with configured workspace and
   table-clearance constraints.
3. Apply robust outlier cleanup without erasing thin object parts.
4. Emit `ObjectObservation3D` with mask provenance, timestamps, source frame IDs,
   point count, uncertainty/quality metrics, centroid, and oriented bounds.
5. Add visual diagnostics for mask, valid selected depth, point cloud, table plane, and
   final bounds. Diagnostics must consume outputs without blocking acquisition.

### M3 — pose/grasp branch

Choose based on the actual manipulation contract:

- Known CAD inventory: prototype FoundationPose in an isolated service and return
  camera-frame SE(3), score, timestamp, and model identifier.
- Unknown everyday objects: prototype grasp inference in an isolated service using the
  selected cloud/mask. Return several ranked camera-frame gripper poses, not only one.

In both cases, reject stale results, transform with an explicitly calibrated
camera-to-robot transform, collision-check against the table and scene, and require a
robot-side reachability/IK check. Perception confidence alone must never authorize
motion.

## Acceptance criteria for the first handoff

- One command starts RealSense capture and the click-selection UI on Linux.
- A positive click selects the intended tabletop object; negative clicks refine it.
- The same object remains selected during modest camera/object motion and brief partial
  occlusion, or the system visibly enters `LOST`.
- Mask pixels and aligned depth refer to the same captured RGB-D pair.
- The app publishes/prints the selected object's metric 3D centroid and bounds at a
  measured rate, with no unbounded queue growth.
- Unit tests cover click coordinate mapping, mask/depth alignment, empty/invalid depth,
  tracking loss, reselection, and frame/timestamp provenance.
- A replay fixture allows the critical path to be tested without the live camera.
- CUDA use and peak VRAM are reported; the process fits within the RTX 3070's 8 GiB.
- The EfficientTAM/SAM 2 comparison and backend decision are recorded with reproducible
  commands; paper FPS numbers are not presented as workstation measurements.
- Existing source/contracts remain usable, and unrelated code is not deleted.

## Engineering constraints

- Preserve the repo's coordinate-frame, timestamp, calibration, and provenance
  semantics. Never label a camera-frame pose as a robot-frame pose.
- Keep learned models behind optional capability boundaries and fail with actionable
  dependency/checkpoint errors.
- Pin external repository revisions and checkpoint hashes; record licenses before
  redistribution or product use.
- Prefer focused adapters over copying an entire research repository into this source
  tree.
- Do not connect perception output to autonomous robot motion during the perception
  milestones. Use visualization and dry-run pose publication first.
- Make small commits with commands/results in the commit message or accompanying docs.

## First action for the Linux agent

Read this brief and the files listed under "Existing code," then report the M0 baseline
before editing. After that, implement M1 and M2 end to end, beginning with the
EfficientTAM-Ti versus SAM 2.1 Hiera Tiny replay benchmark. Do not add YOLO to the
critical path. Do not begin FoundationPose, SAM 3, or grasp-network integration until
click-selected 3D output is repeatable from a saved recording and the live D435I.
