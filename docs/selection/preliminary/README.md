# Preliminary selection runs (superseded)

These files were produced at about 05:20-05:25 EDT on 2026-09-16 by the first agent
session, with an earlier `benchmark_selection.py` and `selection_backend.py`. They are kept
for traceability only and are **not** used for the backend decision:

- the adapter's `++model.fill_hole_area=0` override was applied before the builder's own
  post-processing overrides, so `fill_hole_area=8` won and every frame attempted the
  unbuilt CUDA hole-filling extension;
- images were resized to 1024x1024 on the CPU instead of on the GPU;
- the live run seeded both backends from one hard-coded click on a dark scene, and SAM 2.1
  was never run live;
- there was no ground-truth tracking-quality or loss measurement.

The replacement measurements are in `../M1-backend-decision.md`.
