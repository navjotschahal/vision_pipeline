# M1 — click-to-mask backend decision

**Decision: EfficientTAM-Ti** (`selection.backend: efficienttam-ti`), using the model's
trained memory length. SAM 2.1 Hiera Tiny stays available behind the same
`PromptableVideoSegmenter` contract as the reference.

EfficientTAM-Ti won every measured criterion except the static mask agreement on the
largest objects, where both backends score ≥ 0.968 IoU:

- **Identity:** in the loss/substitution suite, EfficientTAM-Ti reported the hidden
  object as present on 23 of 236 hidden frames and never on the pasted distractor.
  SAM 2.1-Tiny was present on 60 of 230, and on the distractor 21 times. It locked onto
  the distractor for the black remote and stayed there after the real object returned.
- **Mask agreement on the same seeds** (four targets where both backends chose the same
  object): mean IoU 0.970 vs 0.907.
- **Speed and memory:** 38 vs 53 ms per tracked frame; 24.1 vs 18.0 Hz sustained live
  output; capture→mask p50/p95 94/114 vs 110/134 ms (USB 2.1); 798 vs 988 MiB process
  GPU memory.

These are measurements on this workstation (RTX 3070 8 GiB, driver 570.211.01,
PyTorch 2.7.1+cu126), not paper numbers.

## Project gates

| Gate | EfficientTAM-Ti (live D435I) | Status |
|---|---|---|
| ≥ 15 Hz sustained output | 24.05 Hz masks, 24.05 Hz geometry (30 s) | met |
| p95 capture→mask < 100 ms | 114.0 ms p95, 93.6 ms p50 | **not met on the current USB 2.1 link; USB 3 run pending** |
| No unbounded queue | one-value slots only; 162 of 878 captured frames overwritten, 0 geometry inputs overwritten | met |
| Within 8 GiB | 798 MiB process peak (451 MiB allocated) | met |

Every live measurement below was taken with the D435I enumerated at **USB 2.1** (Bus 01,
`480M`), because that is how it was connected during this session. On that link the
camera alone has capture→host p50/p95 of 43/78 ms (`M0.md` addendum), versus a 33.8 ms
mean on the USB 3.2 link measured earlier. The pipeline's own share is visible in
host-receive→mask: p50 54.8, p95 73.6 ms. The latency gate must be re-measured on USB 3
with the stream command below before it can be called met or failed.

## Pinned sources

| Backend | Repository @ revision | Checkpoint (SHA-256) | License |
|---|---|---|---|
| EfficientTAM-Ti | `yformer/EfficientTAM` @ `abcd061ebd3cc6e7527d152d75b890126aaa53f6` | `efficienttam_ti.pt` `acbb17b2…3175fa7d` (Hugging Face `yunyangx/efficient-track-anything`) | Apache-2.0 |
| SAM 2.1 Hiera Tiny | `facebookresearch/sam2` @ `2b90b9f5ceec907a1c18123530e92e794ad901a4` | `sam2.1_hiera_tiny.pt` `7402e0d8…34be69` (`dl.fbaipublicfiles.com/segment_anything_2/092824`) | Apache-2.0; `cc_torch` kernel BSD-3 |

`scripts/setup_selection_models.sh` clones, pins, downloads, and verifies both. The
adapter refuses a repository at another revision or a checkpoint with another hash.
Neither repository is pip-installed, and no CUDA extension is built. The `.venv` gains
only `hydra-core 1.3.2` and `iopath 0.1.10`.

## Integration notes (both backends, same adapter)

`ResearchVideoSegmenter` (`perception/objects/selection_backend.py`) drives the official
video predictor one live frame at a time. It holds the seed frame plus up to three
correction frames, the last 16 tracked frames, and one frame of cached features, so
memory stays at ≤ 17 frames however long the stream runs (checked by
`tests/test_selection_backend_gpu.py`). Images are resized and normalized on the GPU,
and inference runs under `bfloat16` autocast with TF32 enabled.

Findings in the pinned upstream code:

- `clear_non_cond_mem_around_input=true` calls a method that does not exist in either
  repository. The adapter clears stale memory around a correction itself.
- The builders apply their post-processing overrides after the caller's, so a caller's
  `fill_hole_area=0` is silently replaced by 8. That makes every frame attempt the unbuilt
  hole-filling CUDA extension, and it affected the first session's runs
  (`preliminary/`). The adapter passes `apply_postprocessing=False` and restates those
  defaults without hole filling.
- The `efficienttam_ti.yaml` checkpoint uses standard RoPE memory cross-attention; the
  pooled "efficient" attention classes are used only by the `_1`/`_2` variants. Memory
  attention therefore costs ~23 ms in both models (profiled: EfficientTAM image encoder
  8.9 ms, memory attention 22.9 ms, heads 3.4 ms, memory encoder 2.5 ms; SAM 2.1 image
  encoder 24.3 ms, memory attention 22.5 ms).

## Measurement 1 — deterministic tracking quality and loss behavior

```
python -m vision_pipeline.apps.benchmark_selection quality --output-dir docs/selection/m1-quality --rows
```

**Method** (`perception/objects/selection_evaluation.py`): targets are single clicks on the
first frame of the real recordings `d435i_table_static` (lit, 5 targets) and `d435i_m0`
(dark, 2 targets) — `configs/selection_benchmark.yaml`. For each backend, the seeded
object is inpainted out of later frames and its real pixels are pasted back along a
150-frame script:

| Frames | Phase |
|---|---|
| 0–29 | static |
| 30–69 | motion (±40 px) |
| 70–99 | occluder sweep, 8 frames fully covered |
| 100–139 | exit to out of view and return, with an identical copy pasted elsewhere as a distractor |
| 140–149 | static |

Each frame's reference mask is known exactly. It is derived from each backend's own
seed, so the part-vs-whole ambiguity of a click does not bias tracking scores. Session
behavior is the `SelectionPolicy` (score ≥ 0.5, ≥ 50 px, area ratio ≤ 4) replayed over
the raw outputs.

This is a synthetic stress test built from real frames. It does not replace recorded
object motion, which was not available this session.

| Backend | Mean IoU, 7 targets | Mean IoU, 4 same-seed targets | Worst p10 IoU | Hidden frames reported present | Hidden frames on distractor | TRACKING while hidden (session) | Re-acquired after return (raw) | Track ms p50/p95 | Peak allocated / process MiB |
|---|---|---|---|---|---|---|---|---|---|
| **EfficientTAM-Ti** | 0.933 | **0.970** | **0.748** | **23/236** | **0** | 19 | **119/119** | **37.7/38.2** | **449 / 792** |
| SAM 2.1 Hiera Tiny | 0.918 | 0.907 | 0.000 | 60/230 | 21 | 24 | 96/115 | 53.2/54.1 | 644 / 996 |

| Target | Seed IoU between backends | Seed px ETAM / SAM2 | IoU ETAM / SAM2 | Hidden present, on distractor: ETAM / SAM2 | Session LOST frame: ETAM / SAM2 |
|---|---|---|---|---|---|
| table/taped box | 0.97 | 40924 / 39947 | 0.968 / **0.992** | 1/33, 0 / 1/32, 0 | 77 / 79 (area jump, 97 % covered) |
| table/white ball | 0.99 | 1800 / 1811 | **0.981** / 0.979 | 1/33, 0 / 0/32, 0 | 79 / 79 (low score) |
| table/side-table box | 0.99 | 5002 / 4971 | 0.988 / **0.991** | 3/34, 0 / 7/34, **2** | 80 / 79 |
| table/black remote | 0.89 | 883 / 833 | **0.941** / 0.665 | 0/33, 0 / **33/33, 18** | 78 / 86 |
| table/gripper fingertip | 0.23 | 2364 / 9957 | 0.929 / 0.973 | 1/34, 0 / 2/33, 0 | 78 / 79 |
| m0/dark case side | 0.37 | 7596 / 20196 | 0.910 / 0.957 | 2/35, 0 / 2/32, 1 | 79 / 79 |
| m0/dark equipment panel | 0.27 | 6581 / 24138 | 0.812 / 0.868 | 15/34, 0 / 15/34, 0 | 111 / 111 |

- For the last three targets the two backends chose different part/whole segments from
  the same click, so their IoUs are not directly comparable. SAM 2.1 tended to choose
  the larger whole.
- Both backends declared LOST only once the object was ≥ 97 % covered (reference
  fraction 0.00–0.03 at the loss frame), never during partial occlusion, and never
  re-acquired automatically.
- **SAM 2.1 on the black remote** (montage
  `m1-quality/sam2.1-hiera-tiny__d435i_table_static__black_remote.jpg`): it segmented the
  occluder while the object was covered and the pasted copy while the object was out of
  view, and it stayed on the copy after the real object returned. The session's
  area-jump check stopped it at frame 86. EfficientTAM-Ti reported nothing while the
  object was hidden and re-acquired it on return.
- The 15 "present while hidden" frames on the dark panel are an artifact of the method,
  shared by both backends: inpainting the dark scene leaves a dark patch resembling the
  panel.

Cold start in a fresh process (`m1-startup.ndjson`, two runs each):

```
python -m vision_pipeline.apps.benchmark_selection startup --backend <backend> \
  --replay recordings/d435i_table_static --click 218 432 --output docs/selection/m1-startup.ndjson
```

| Backend | Process start → ready | Model build | First seed (cold) | Warm seed | Warm track (2nd frame) |
|---|---|---|---|---|---|
| EfficientTAM-Ti | 1.66–1.74 s | 0.44–0.48 s | 241–252 ms | 16.2–17.5 ms | 26.1–28.5 ms |
| SAM 2.1 Hiera Tiny | 1.94–1.96 s | 0.66 s | 251–256 ms | 32.3–34.5 ms | 43.5–43.8 ms |

A track call before memory fills is cheaper than a steady-state one (38 / 53 ms above).

## Measurement 2 — threaded pipeline on paced replay and the live D435I

Same pipeline as the app: capture thread → one-value slot → mask worker → one-value slot
→ geometry worker. A scripted click on the white ball (218, 432) is applied to the first
processed frame. The first 2 s after selection are excluded as settling.

```
python -m vision_pipeline.apps.benchmark_selection stream --backend <backend> \
  --replay recordings/d435i_table_static --click 218 432 --duration 12 \
  --output docs/selection/m1-stream/replay-<backend>.json
python -m vision_pipeline.apps.benchmark_selection stream --backend <backend> \
  --click 218 432 --duration 30 --output docs/selection/m1-stream/live-usb2-<backend>.json
```

| Run | Sustained mask Hz | Worker ms p50/p95 | Host receive→mask p50/p95 ms | Capture→mask p50/p95 ms | Outputs > 100 ms old | Frames read / overwritten / seq. gaps | Geometry Hz | Capture→geometry p50/p95 ms | Lost | Peak alloc / process MiB |
|---|---|---|---|---|---|---|---|---|---|---|
| ETAM paced replay | 25.08 | 38.2/39.6 | — | — | — | 300 / 52 / 2 (recorded) | 25.09 | — | 0 | 451 / 798 |
| SAM2 paced replay | 18.46 | 53.6/55.4 | — | — | — | 300 / 117 / 2 (recorded) | 18.46 | — | 0 | 642 / 986 |
| **ETAM live, USB 2.1** | **24.05** | 38.6/41.6 | 54.8/73.6 | **93.6/114.0** | 214 (32 %) | 878 / 162 / 23 | **24.05** | 98.9/121.7 | 0 | 451 / 798 |
| SAM2 live, USB 2.1 | 17.95 | 54.2/56.4 | 72.1/94.8 | 110.0/133.8 | 358 (72 %) | 873 / 343 / 27 | 17.95 | 115.7/140.0 | 0 | 642 / 988 |

Capture→mask uses librealsense `global_time`, the SDK's device-to-host-realtime mapping;
the mapping error is not measured. Replay runs report compute only. On the static ball,
the centroid std-dev over the steady window was 0.47/1.07/2.08 mm (ETAM) and
0.55/1.05/1.93 mm (SAM2).

## Rejected tuning, with evidence

- **Fewer recent memories** (adapter option `recent_memory_frames`; the temporal
  encodings are remapped so the kept memories and the conditioning frame use their
  trained encodings). EfficientTAM-Ti track time dropped from 38.0 to 31.2 / 28.7 /
  26.3 ms with 3 / 2 / 1 memories, but quality collapsed:

  | Memories | Mean IoU (7) | Hidden on distractor | Re-acquired | Output |
  |---|---|---|---|---|
  | 3 | 0.783 | 30 | 31/119 | `m1-quality-etam-memory3/` |
  | 2 | 0.793 | 17 | 31/119 | `m1-quality-etam-memory2/` |
  | 6 (stock) | 0.933 | 0 | 119/119 | `m1-quality/` |

  Not used. SAM 2.1-Tiny stays above 41 ms even with one memory.

- **`torch.compile` of memory attention** (EfficientTAM-Ti): steady state 37.9 → 36.4 ms,
  after ~30 s of compilation and recompiles while the pointer memory fills. Not used.

## Selection behavior delivered

`SelectionSession` implements the brief's policy:

- **States:** `UNSELECTED` until a positive click. `TRACKING` while each mask passes the
  loss checks. `LOST` latches on low object score, a too-small mask, or a > 4× area jump,
  and it resets backend memory without automatic re-acquisition or substitution.
- **Clicks:** left click selects or adds a positive prompt; right click (or shift-click)
  adds a negative prompt while tracking; `c` clears. A left click while `LOST` reselects.
- **Provenance:** clicks name the displayed frameset and are applied to that exact
  frame, held in a 30-frame ring. A click on a frame that has aged out is rejected with a
  message. Every mask carries its frameset and color sample header, and a result for a
  different frame is refused.

Unit tests: `tests/test_selection.py`, `tests/test_selection_pipeline.py`,
`tests/test_selection_evaluation.py`, `tests/test_rgbd_recording.py`, and the opt-in
`tests/test_selection_backend_gpu.py`.

## Open items

1. Re-run the two live stream commands on a USB 3 link to settle the p95 < 100 ms gate.
2. Record a real sequence with object motion, hand occlusion, and an object leaving
   view, and review it with the app. The loss/substitution evidence above is synthetic.
3. Text prompting, SAM 3, FoundationPose, grasping, and robot motion remain out of scope.
