# Learning plan: text-prompted object tracking with unified 4D models (Pattern B)

Started 2026-09-24. Goal: say an object name ("book", "box", "apple", "orange"), find it,
attach 3D points to it, track them live, and turn those tracks into an object pose that a
robot arm (or two) can act on. The architecture studied is **Pattern B**: one feed-forward
4D model that encodes a video once and answers queries of the form "where is this pixel's
3D point at time t?".

This file is meant to be worked through **one day at a time**, including by a separate
Claude session. The protocol for that session is at the end, and the progress log is the
last section.

---

## 1. The target architecture

```
 "apple" ─► GROUNDING (SAM 3 / YOLOE-26) ─► mask of the apple in frame t0
                                              │ sample N pixels (u,v) inside the mask
                                              ▼
 video window ─► 4D ENCODER (ViT, once per window) ─► scene latent F
                                              │
       queries q = (u, v, t_src=t0, t_tgt=t, t_cam=t) ─► QUERY DECODER ─► 3D point P_i(t)
                                              ▼
                 3D point tracks of the apple, one trajectory per sampled pixel
                                              ▼
 D435i depth ─► METRIC SCALE: Umeyama with scale s aligns tracks to measured depth
                                              ▼
 POSE: weighted Umeyama + RANSAC ─► T_CO(t) ∈ SE(3), filtered on the Lie group
                                              ▼
 HEALTH CHECK: inlier ratio, residual ─► on failure, re-run GROUNDING (re-detect)
                                              ▼
 MANIPULATION: T_WG(t) = T_WC · T_CO(t) · T_OG   (one T_OG per arm)
```

Two facts shape this design:

1. **Pattern B models do not understand language.** D4RT, Point4D and Flow4R answer "where
   does this pixel go?", not "where is the apple?". Grounding picks *which* pixels to
   query. (IGGT4D, 2026-07, adds instance identity and open-vocabulary segmentation inside
   the 4D model; its code was not released as of 2026-09-24.)
2. **Pattern B models run on windows (chunks) of frames, not one frame at a time.** "Real
   time" therefore means a sliding window that advances by S frames: latency is at least
   S/fps plus one window's inference time, and compute grows as window/S because each
   step re-encodes the whole window. Measuring that trade-off is part of the plan.

## 2. What exists to run (checked 2026-09-24)

| Model | What it is | Code / weights | Input | Use in this plan |
|---|---|---|---|---|
| **D4RT** (DeepMind, CVPR 2026 Best Paper, [2512.08924](https://arxiv.org/abs/2512.08924)) | ViT-g encoder + cross-attention query decoder; query `(u,v,t_src,t_tgt,t_cam)` → 3D point | **Official: none.** Paper and project page only | RGB video | Read the paper in depth. It is the reference design. |
| **OpenD4RT** ([github Lijiaxin0111/Open-d4rt](https://github.com/Lijiaxin0111/Open-d4rt)) | Unofficial PyTorch reimplementation, VideoMAE2 ViT-G hybrid backbone | Apache-2.0, HF checkpoints (32- and 48-frame) | RGB video | Main model to run. Its size versus the RTX 3070's 8 GiB is unknown, so measure it. |
| **Point4D** (CMU, 2026-09, [2609.09145](https://arxiv.org/abs/2609.09145)) | 3D query motion decoder; re-queries 3D endpoints across chunks; 200+ frame tracks | MIT, weights `minsikj/point4d` on HF | RGB `(T,H,W,3)` | Second model. Its chunk-to-chunk handoff is the streaming mechanism to study. |
| **Flow4R** (TUM, 2026-02, [2602.14021](https://arxiv.org/abs/2602.14021)) | Two-view ViT predicting 3D points + scene flow + confidence per pixel | Release unclear; check the project page | image pair | Read. Scene flow as the one core representation. |
| **IGGT4D** (2026-07, [2607.19228](https://arxiv.org/abs/2607.19228)) | Streaming, causal, instance-grounded 4D transformer | "Coming soon" | video | Read. Watch for the code release; it could merge grounding into stage B. |
| **TAPIP3D** ([2504.14717](https://arxiv.org/abs/2504.14717)) / **SpatialTrackerV2** ([2507.12462](https://arxiv.org/abs/2507.12462)) | 3D point trackers; TAPIP3D accepts **RGB-D** | Public code | RGB(-D) | Baselines that can use the D435i depth directly. |
| **SAM 3 / 3.1** ([2511.16719](https://arxiv.org/abs/2511.16719)), **YOLOE-26** ([2602.00168](https://arxiv.org/abs/2602.00168)) | Text → masks (+ tracking for SAM 3) | Public | RGB | Grounding stage. |

Hardware: Mac (Apple MPS) for reading, math and small exercises; Linux RTX 3070 (8 GiB)
for model runs; Intel RealSense D435i for RGB-D recordings. All published speeds
(D4RT's on an A100, SAM 3.1's on an H100) are datacenter-GPU numbers. Record your own.

## 3. Rules for every day

- Each day has **Read**, **Math**, **Build**, **Check yourself** and **Done when**.
- Exercises live in `learning/pattern-b/dayNN/` (standalone scripts; do not modify
  `src/`). NumPy first, and PyTorch only once the idea has been written by hand.
- Record only **measured** numbers, with the hardware, resolution and window size.
- Before any day that reads a 2025–2026 paper, search for newer versions, code releases
  or follow-ups (arXiv, the project page, GitHub) and note what changed.

---

## Phase 0: Geometry you cannot skip (days 1–5)

### Day 1: Pinhole camera, back-projection, depth
- **Read:** Hartley & Zisserman ch. 6 (camera models); Szeliski *Computer Vision* 2nd ed. §2.1.
- **Math:** `u = fx·X/Z + cx`, `v = fy·Y/Z + cy`, and its inverse `X = (u−cx)·Z/fx`,
  `Y = (v−cy)·Z/fy`. Why depth resolution gets worse with distance for a stereo camera:
  `σ_Z ≈ Z²·σ_d / (f·b)`, where `σ_d` is the disparity error and `b` the baseline.
- **Build:** back-project one recorded D435i depth frame to a point cloud in NumPy, then
  project it back to pixels and check that the round-trip error is 0.
- **Check yourself:** What happens to `σ_Z` when Z doubles? Why does the D435i's 50 mm
  baseline matter?
- **Done when:** the round trip is exact and you can derive `σ_Z` on paper.

### Day 2: Rotations and SE(3)
- **Read:** Solà, Deray, Atchuthan, *A micro Lie theory for state estimation in robotics*
  ([1812.01537](https://arxiv.org/abs/1812.01537)), §I–III.
- **Math:** SO(3) (`RᵀR = I`, `det R = 1`), SE(3) as a 4×4 matrix, composition, the
  inverse `T⁻¹ = [Rᵀ, −Rᵀt]`, frame notation `T_AB` (maps points in B to A).
- **Build:** `se3.py` with compose, inverse and apply-to-points, plus tests
  (`T·T⁻¹ = I`, `T_AC = T_AB·T_BC`).
- **Check yourself:** Why can't you average two rotation matrices element-wise?
- **Done when:** the tests pass and you can read any `T_XY` chain aloud correctly.

### Day 3: Lie algebra, exp/log, perturbations
- **Read:** Solà §IV–V (exp/log, ⊕/⊖, Jacobians).
- **Math:** Rodrigues' formula `exp(θ[ω]×) = I + sinθ[ω]× + (1−cosθ)[ω]×²`; the log map;
  `T ⊕ τ = T·exp(τ^)`; why errors live in the 6-D tangent space.
- **Build:** exp/log for SO(3) and SE(3); verify `log(exp(τ)) = τ` on random samples,
  including θ near 0 and near π.
- **Check yourself:** What breaks numerically near θ = 0 and θ = π, and how do you fix it?
- **Done when:** round-trip tests pass at the edge cases.

### Day 4: Umeyama / Kabsch (the pose solver D4RT itself uses)
- **Read:** Umeyama 1991, *Least-squares estimation of transformation parameters between
  two point patterns*; Arun, Huang & Blostein 1987.
- **Math:** minimise `Σ wᵢ‖qᵢ − (sR pᵢ + t)‖²`. Weighted centroids `p̄, q̄`;
  `H = Σ wᵢ (pᵢ−p̄)(qᵢ−q̄)ᵀ = UΣVᵀ`; `R = V·diag(1,1,det(VUᵀ))·Uᵀ`;
  `s = tr(ΣD) / Σ wᵢ‖pᵢ−p̄‖²` (D = the diag above); `t = q̄ − sR p̄`.
  Derive it: the cross term reduces to maximising `tr(R H)`; why the SVD gives that
  maximum; why the `det` term prevents a reflection.
- **Build:** `umeyama(p, q, w, with_scale)` in NumPy. Test on synthetic points with known
  `s, R, t`, noise, and a planar (degenerate) case.
- **Check yourself:** How many points are needed in the minimal case? Why does a planar
  point set still work, while collinear points do not?
- **Done when:** the known transform is recovered to 1e-9 without noise, and the error
  grows smoothly with noise.

### Day 5: RANSAC and robust estimation
- **Read:** Fischler & Bolles 1981 (RANSAC); skim MAGSAC++ (Barath et al., CVPR 2020).
- **Math:** iterations needed `k = log(1−p) / log(1−wᵐ)` for inlier ratio `w`, sample
  size `m=3`, confidence `p`; choosing the inlier threshold from sensor noise (Day 1's σ_Z).
- **Build:** RANSAC around Day 4's solver, with 30 % of points moved onto a "table"
  (tracks that slid off the object). Compare against plain Umeyama.
- **Check yourself:** At `w = 0.5`, `p = 0.99`, how many iterations? What if `w = 0.2`?
- **Done when:** RANSAC recovers the pose with 30–50 % outliers and plain Umeyama fails.

## Phase 1: Transformer machinery behind query decoders (days 6–9)

### Day 6: Attention from scratch
- **Read:** Vaswani et al. 2017; Dosovitskiy et al. 2021 (ViT).
- **Math:** `softmax(QKᵀ/√d)V`; why the `√d` scaling; cost `O(N²d)` and what N is for a
  video (the frame count × patches per frame).
- **Build:** single-head and multi-head attention in NumPy, checked against
  `torch.nn.functional.scaled_dot_product_attention`.
- **Check yourself:** For 48 frames at patch size 2×16×16 on 256×256 frames, how many
  tokens does D4RT's encoder see?
- **Done when:** your output matches PyTorch to 1e-6.

### Day 7: Cross-attention and query-based decoders
- **Read:** Carion et al. 2020 (DETR); Jaegle et al. 2021 (Perceiver IO), the direct
  ancestor of "encode once, decode any query".
- **Math:** cross-attention: queries from the task, keys/values from the scene latent `F`.
  Why decoding N queries costs `O(N·|F|)` and is independent per query.
- **Build:** a toy query decoder in PyTorch: encode a 2D function on a grid, query it at
  arbitrary continuous `(x, y)` with Fourier-feature positional encoding.
- **Check yourself:** Why can D4RT decode 550 tracks at 60 FPS but not every pixel?
- **Done when:** the toy decoder interpolates between grid points it never saw.

### Day 8: Video backbones (VideoMAE, DINOv2, spatio-temporal patches)
- **Read:** Tong et al. 2022 (VideoMAE) and Wang et al. 2023 (VideoMAE V2), since OpenD4RT uses a
  VideoMAE2 ViT-G; Oquab et al. 2023 (DINOv2).
- **Math:** tube-patch embedding; masked-autoencoder pretraining objective; local
  (frame-wise) versus global attention and their costs.
- **Build:** count parameters and activation memory for ViT-B/L/g at your window size;
  predict whether ViT-G fits in 8 GiB at fp16.
- **Done when:** you have a written memory estimate to check against on Day 15.

### Day 9: Feed-forward 3D reconstruction lineage
- **Read:** DUSt3R (Wang et al., CVPR 2024), VGGT (Wang et al., CVPR 2025): pointmaps
  and why predicting 3D points directly replaced classical SfM pipelines.
- **Math:** a pointmap `X ∈ ℝ^{H×W×3}` in a reference camera frame; confidence-weighted
  regression loss; scale ambiguity for RGB-only input.
- **Check yourself:** Why does an RGB-only model's output need a scale (Day 4's `s`) before
  a robot can use it?
- **Done when:** you can draw DUSt3R → VGGT → D4RT and say what each one added.

## Phase 2: Point tracking, 2D to 3D (days 10–12)

### Day 10: Tracking Any Point (TAP) and its metrics
- **Read:** Doersch et al. 2022 (TAP-Vid); Karaev et al. 2024 (CoTracker3,
  [2410.11831](https://arxiv.org/abs/2410.11831)).
- **Math:** metrics `δ_avg`, Occlusion Accuracy, Average Jaccard (AJ); why visibility is
  predicted separately from position.
- **Build:** run CoTracker3 on a D435i recording of a moving box; visualise tracks.
- **Done when:** you can explain AJ and have a baseline video.

### Day 11: Online trackers
- **Read:** TAPNext++ ([2604.10582](https://arxiv.org/abs/2604.10582), CVPR Findings 2026);
  Track-On2 ([2509.19115](https://arxiv.org/abs/2509.19115)).
- **Math:** recurrent state versus sliding windows; the re-detection metric `AJ_RD`.
- **Check yourself:** When is an online (causal) tracker required and when is a windowed
  one acceptable for a robot?
- **Done when:** you have a written comparison: latency floor, memory, and re-detection.

### Day 12: 3D point tracking
- **Read:** TAPIP3D ([2504.14717](https://arxiv.org/abs/2504.14717)), SpatialTrackerV2
  ([2507.12462](https://arxiv.org/abs/2507.12462)).
- **Math:** world-centric versus camera-centric tracks; `P_world = T_WC(t)·P_cam`.
- **Build:** lift Day 10's 2D tracks with D435i depth (Day 1) into 3D tracks. This is the
  classical baseline every Pattern B result gets compared against.
- **Done when:** you have 3D tracks of the moving box and have noted where depth holes break them.

## Phase 3: Pattern B in depth (days 13–17)

### Day 13: D4RT, the paper
- **Read:** [2512.08924](https://arxiv.org/abs/2512.08924) in full, including the appendix.
- **Math:** query `q = (u, v, t_src, t_tgt, t_cam)` plus a local patch embedding; how
  tracking, depth, point clouds and camera pose are all *choices of query*; camera pose
  via Umeyama on two query grids (your Day 4 code).
- **Check yourself:** Which query gives "the apple's 3D trajectory in the frame of camera
  time 0"? Which one gives "depth map at frame 7"?
- **Done when:** you can write the query for each task without looking.

### Day 14: Point4D and Flow4R, two other routes
- **Read:** Point4D ([2609.09145](https://arxiv.org/abs/2609.09145)): a 3D query decoder
  independent of visibility, and re-querying endpoints across chunks. Flow4R
  ([2602.14021](https://arxiv.org/abs/2602.14021)): scene flow as the core output.
- **Check yourself:** How does Point4D carry a track from chunk k to chunk k+1 without
  re-matching? What does that imply for streaming?
- **Done when:** you have a one-page comparison of D4RT, Point4D and Flow4R.

### Day 15: Run OpenD4RT
- **Build:** install OpenD4RT on the 3070; run a checkpoint on a D435i RGB clip; query
  tracks for pixels sampled on one object.
- **Measure:** peak GPU memory, encoder time, decoder time per 1k queries, window length.
  Compare against the Day 8 estimate.
- **Done when:** the numbers are in the log (or a clear record of why it does not fit and
  what was tried: fp16, fewer frames, lower resolution).

### Day 16: Run Point4D
- **Build:** the same clip and object pixels; its chunked demo with overlap.
- **Measure:** the same metrics as Day 15, plus how track quality holds across chunk boundaries.
- **Done when:** a side-by-side video exists for OpenD4RT versus Point4D versus Day 12's baseline.

### Day 17: Metric scale and accuracy against the D435i
- **Math:** fit `s, R, t` (Day 4, `with_scale=True`) between model points and
  depth-back-projected points on the same pixels; residual statistics after alignment.
- **Build:** a per-window scale estimate; report the scale's drift across windows.
- **Done when:** you know each model's error in mm against the D435i on your scene.

## Phase 4: Build the system (days 18–23)

### Day 18: Grounding
- **Read:** SAM 3 ([2511.16719](https://arxiv.org/abs/2511.16719)): the detector and the
  tracker share one backbone, and the "what" loop is separate from the "where" loop.
- **Build:** text → mask with YOLOE-26 (real-time) and SAM 3 (accuracy reference) on
  "book", "box", "apple" and "orange".
- **Done when:** you have masks for all four and their per-frame time on the 3070.

### Day 19: Query sampling
- **Math:** how many points: pose error falls roughly like `σ/√N` until outliers
  dominate; spread across the mask (farthest-point sampling) versus random; avoid mask
  edges where depth mixes object and background.
- **Build:** a sampler returning N pixels from a mask with an edge margin.
- **Done when:** you have a plot of pose jitter against N on a static object.

### Day 20: Object frame and pose over time
- **Math:** at t0 define the object frame O (centroid + axes); store `pᵢ` in O; each
  window solve `T_CO(t)` with RANSAC + Umeyama (Days 4–5), weights from model confidence.
- **Build:** a streaming loop: sliding window → queries → 3D tracks → `T_CO(t)`.
  Draw the object axes on the video.
- **Done when:** the axes stay attached to a box moved by hand.

### Day 21: Filtering on SE(3)
- **Read:** Solà §VI (error-state Kalman filter on manifolds).
- **Math:** state `T`, constant-velocity model in the tangent space, update with
  `T ⊕ K·(z ⊖ T)`; the covariance is what downstream control trusts.
- **Build:** an ES-EKF over the Day 20 poses; compare jitter and lag against raw poses.
- **Done when:** measured jitter and lag are logged for both.

### Day 22: Failure detection and re-grounding
- **Read:** RRTrack ([2607.23669](https://arxiv.org/abs/2607.23669)); Robust 6-DoF
  Tracking with Built-In Recovery ([2607.23468](https://arxiv.org/abs/2607.23468)).
- **Build:** health signals (inlier ratio, residual, visibility fraction); when they drop,
  re-run grounding and re-initialise the queries with the object frame preserved (match
  the new points to `pᵢ` via Umeyama on the overlap).
- **Done when:** covering the object with a hand and uncovering it recovers the same
  object frame.

### Day 23: End-to-end latency budget
- **Measure:** capture → grounding → window fill → encoder → decoder → pose → filter.
  Plot latency against window length.
- **Done when:** you have a written answer to "what window length gives the best trade-off
  between latency and stability on the 3070?"

## Phase 5: Handing the pose to manipulation (days 24–26)

### Day 24: Retargeting as a composition of transforms
- **Math:** `T_WG(t) = T_WC · T_CO(t) · T_OG`; bimanual: `T_OG_left`, `T_OG_right` fixed
  in the object frame; how the pose covariance propagates through the chain (adjoint).
- **Build:** given a tracked box, output both grasp targets in the world frame and
  visualise them following the box.

### Day 25: Object flow as the interface
- **Read:** Dream2Flow ([2512.24766](https://arxiv.org/abs/2512.24766)), Dex4D
  ([2602.15828](https://arxiv.org/abs/2602.15828)), PointWorld
  ([2601.03782](https://arxiv.org/abs/2601.03782)).
- **Check yourself:** When is a single SE(3) pose enough, and when do you need the full
  point flow (deformable, articulated objects)?

### Day 26: Review and write-up
- Write a 2-page summary of the architecture, every measured number, what failed, and
  the next step (for example, IGGT4D if its code is out, or training a smaller encoder).

---

## Protocol for the daily learning session (for a separate Claude session)

1. Read this file. Find the first day in the progress log that is not marked done.
2. Search the web for updates on that day's papers (new versions, code releases,
   follow-ups from 2026) and list anything new before teaching.
3. Teach in order: **intuition → math derivation → worked numeric example → build**.
   Ask the learner to derive or predict before revealing.
4. Build the exercise in `learning/pattern-b/dayNN/`; run it; keep only measured numbers.
5. Go through "Check yourself" with the learner. Do not mark the day done until "Done when" holds.
6. Append a log entry below: date, day, what was learned, measured numbers with hardware,
   open questions. Do not edit earlier entries.

## Progress log

| Date | Day | Status | Measured / learned | Open questions |
|---|---|---|---|---|
| 2026-09-24 | — | plan written | Checked code availability: D4RT official none; OpenD4RT unofficial (Apache-2.0, HF ckpts); Point4D MIT + weights; IGGT4D pending | Does OpenD4RT's ViT-G fit in 8 GiB? |
