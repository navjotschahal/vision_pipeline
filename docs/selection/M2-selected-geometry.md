# M2 — selected object in metric 3D

Status on 2026-09-16: implemented and measured on replay and the live D435I. Every output
is in `realsense_color_optical` (X right, Y down, Z forward). Nothing is expressed in a
robot frame and nothing commands motion.

## What is computed

`SelectedGeometryEstimator.estimate(frame, segmentation, selection_id)` in
`src/vision_pipeline/perception/objects/selected_geometry.py` first refuses a mask that
does not name the frame's own frameset and color sample. It then runs these stages, and
each stage's surviving point count is reported:

| # | Stage | Removes | Keeps |
|---|---|---|---|
| 1 | Valid depth (`extract_object_point_cloud`, stride 2, 0.2–2.0 m) | zero/non-finite/out-of-range depth | measured mask pixels |
| 2 | Depth-continuous components (4-neighbour, step ≤ max(1 cm, 3 % of range)) | mask leakage onto surfaces far behind the object | largest surface, plus same-depth pieces split by an occluder; no boundary pixel is eroded |
| 3 | Workspace crop (explicit frame, **uncalibrated placeholder**) | points outside bounds | — |
| 4 | Support clearance | points within 1 cm of, or behind, an accepted support plane | — |
| 5 | Isolated points (3D neighbour count in a range-adaptive radius) | flying pixels | connected thin structures |

The support plane is fitted by RANSAC (`fit_support_plane`) to **non-mask** depth in a
ring around the object's image region, not to the whole workspace. The ring's radius
scales with object size, and a two-cell margin keeps mixed boundary pixels out. The
plane is oriented toward the camera. It is rejected (`support_plane_status=above_object`)
when the 5th percentile of the object's signed distances to it is below −5 cm, i.e.
when it passes above part of the object. Otherwise a clearance cut would erase real
object points. The plane is reused for up to 10 frames while that check still passes.

The first implementation fitted the plane over the whole workspace. On the live stream
it sometimes chose a parallel surface ~0.2 m above the table (the big box's top or the
side table), and it rejected 40–50 of ~700 live frames of the ball as
`at_or_below_support`. Those first live runs were overwritten by the reruns reported in
`M1-backend-decision.md`, which have no rejections. With the local ring and the
above-object check, the
same recording yields geometry on 300/300 frames with a stable table offset
(0.868–0.890 m).

Output (`SelectedObjectGeometry`):

- `observation: ObjectObservation3D` (existing contract): source = aligned **depth**
  sample header (capture time, frame ID, calibration), `SEGMENTATION_MASK` selection,
  `MEASURED_METRIC`, point count, PCA oriented box of the **visible surface**.
  `confidence` is the segmenter's object-presence probability, not a geometric
  confidence.
- `cloud`, `frameset_id`, `color_source` (the exact color sample the mask came from),
  `selection_id`, `mask_backend`, `object_score`.
- `centroid_metres`, `covariance_metres2` (row-major 3×3 spread of the surface points).
- `support_plane` (or `None`) and `quality: SelectedGeometryQuality`: per-stage counts,
  valid-depth fraction, components found, depth median/MAD, neighbour radius, plane
  status/RMS/inliers/age.
- Rejections raise `SelectedGeometryError` with a `GeometryRejection` reason
  (`no_valid_depth`, `outside_workspace`, `no_support_plane` if required,
  `at_or_below_support`, `too_few_points`, `degenerate`). The selection stays
  `TRACKING`; only the 3D output is withheld for that frame.

The oriented box is not an object pose. Its PCA axes are ambiguous for symmetric
objects, and it bounds only the observed surface.

## Repeatability on static recordings

```
python -m vision_pipeline.apps.benchmark_selection geometry \
  --output docs/selection/m2-geometry/geometry-efficienttam-ti.json
```

EfficientTAM-Ti masks, `configs/object_selection.yaml`, every recorded frame. The scene
is static, so the spread is the combined repeatability of mask, depth, filtering, and
bounds. It is not accuracy against a measured ground truth: no object was measured with
a tape or reference.

| Target (recording) | Frames with geometry | Centroid std x/y/z (mm) | Sorted visible-surface bounds, median (m) | Support plane |
|---|---|---|---|---|
| taped box (table_static) | 300/300 | 0.52 / 1.44 / 0.74 | 0.244 × 0.325 × 0.338 | fitted/cached 300 |
| white ball | 300/300 | 0.44 / 1.46 / 2.05 | 0.038 × 0.069 × 0.073 | fitted/cached 300 |
| box on side table | 300/300 | 1.83 / 1.12 / 4.05 | 0.060 × 0.221 × 0.295 | fitted/cached 300 |
| black remote | 300/300 | 3.15 / 2.29 / 7.68 | 0.037 × 0.061 × 0.116 | not found 32, fitted/cached 268 |
| gripper fingertip (hanging in air) | 300/300 | 0.45 / 1.55 / 1.56 | 0.065 × 0.103 × 0.157 | above object 299 (correctly unused) |
| dark case side (d435i_m0) | 180/180 | 3.19 / 1.52 / 6.76 | 0.068 × 0.373 × 0.529 | above object 180 |
| dark equipment panel (d435i_m0) | 83/180 | 16.5 / 11.7 / 16.2 | 0.019 × 0.144 × 0.247 | rejected 97 frames: `at_or_below_support` 80, `too_few_points` 17 |

Geometry time on the host CPU in the synchronous run: p50 3–8 ms, p95 8–17 ms, depending
on mask size (per-target values are in the JSON).

## Live and paced-replay rates

Geometry runs in its own worker behind a one-value slot, so it never delays masks
(stream commands and full tables are in `M1-backend-decision.md`):

| Run (EfficientTAM-Ti, white ball) | Geometry Hz | Geometry ms p50/p95 | Capture→geometry p50/p95 ms | Geometry inputs overwritten | Rejections |
|---|---|---|---|---|---|
| Paced replay, 12 s | 25.09 | 4.3/11.6 | — | 0 | 0 |
| Live D435I, USB 2.1, 30 s | 24.05 | 4.5/12.1 | 98.9/121.7 | 0 | 0 |

The live capture→geometry latency includes the USB 2.1 link's 43/78 ms capture→host
share (`M0.md` addendum).

## Visualization and dry-run publication

`python -m vision_pipeline.apps.select_object` shows two panels:

- **Color panel:** the frame the displayed geometry was computed from, with a translucent
  mask, click markers, the oriented bounds projected with that frame's color intrinsics,
  the centroid, state, score, capture→mask age, and the metric readout.
- **Cloud panel:** every valid-depth mask point colored by the stage that removed it
  (green final, pink isolated, yellow table clearance, brown workspace, red other
  surface), the support-plane sample (blue inliers), bounds, and centroid. It is
  rendered at most 15 Hz, from references published by the workers, so rendering never
  blocks capture, masks, or geometry.

Both panels also draw the box's principal axes from its centre: x red (longest visible
extent), y green, z blue (thinnest; the normal of a flat face). They are labelled as PCA
axes, not a pose. NDJSON records carry them as `bounds.principal_axes` (unit column
vectors in the camera frame).

The snapshots show where PCA axes can and cannot be trusted:

- **Flat box** (`side-box-tracking.jpg`, bounds 0.30 × 0.21 × 0.06 m): z is its top-face
  normal and x its long edge, so the axes line up with the faces.
- **Near-cubic box** (`box-tracking.jpg`, 0.325 × 0.326 × 0.248 m): x and y are almost
  equal, so the PCA axes are diagonal to the faces and can rotate between frames.

For grasping a box between two arms, face-aligned axes need a different estimate, e.g.
the table normal as "up" plus a minimum-area rectangle of the footprint for yaw.

Snapshots from replay: `m2-visualization/box-tracking.jpg`, `side-box-tracking.jpg`, and
`ball-tracking.jpg`.

`--ndjson PATH` appends one `selected-object-observation-v1` record per geometry output.
Each record carries the frameset, color and depth sample IDs, both device capture
stamps with their clock domain, the host receive stamp, state, loss reason, backend,
score, capture→mask and capture→geometry latency, centroid, covariance, bounds (with an
explicit "not an object pose" marker), support plane, quality counts, and method. A
15 s live run wrote 337 records.

## Limitations found

- Workspace bounds are an explicit placeholder, and there is no camera-to-table or
  camera-to-robot calibration.
- A support plane is assumed to be locally planar around the object. A thin object lying
  flat, like the black remote, loses most points to the 1 cm clearance, and its plane
  was not found in 32 frames. Its centroid spread (7.7 mm in z) is the largest among the
  lit targets.
- Dark, specular, or low-texture surfaces yield sparse depth (dark equipment panel:
  geometry on 83/180 frames).
- The support-plane cache assumes the camera does not move relative to the table
  within 10 frames.
- Bounds and covariance describe visible surfaces, so they change with viewpoint.
