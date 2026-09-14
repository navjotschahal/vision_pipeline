# Compute placement policy

**Status:** initial architecture constraint, reviewed 2026-09-11

This project will exploit sensor-native processing, GPUs, C++, DSPs, and NPUs when they
improve measured latency, throughput, power, or data movement. It will not assume that a
brand name implies a particular accelerator capability.

## Independent facts

Every processing result eventually needs to preserve these independent facts:

- where the result was computed;
- where its payload currently lives;
- which implementation and configuration produced it;
- which source capture time it inherits;
- whether a transfer or copy occurred;
- which object owns the buffer and when it is safe to reuse.

Lesson 1 implements the first two facts and preserves the timestamp. Provenance chains,
copy accounting, ownership leases, and GPU completion events will be added with typed
buffer contracts.

Moving a measurement never changes its capture timestamp. Derived results inherit the
source acquisition time and add separate processing start/end times.

## Runtime boundary

```text
sensor firmware / ASIC
    native depth, ISP, hardware synchronization, supported embedded perception
        |
        v
C++ data plane
    callbacks, frame ownership, pools, bounded queues, timestamp extraction,
    zero/low-copy interop, measured hot geometry
        |
        +----------> recorder (original headers and payloads)
        |
        v
GPU processing
    color conversion, resize, normalization, inference, dense masks/features,
    compatible depth/point-cloud operations
        |
        v
Python experiment plane
    graph configuration, models, evaluation, visualization, replay, RL/IL datasets
```

C++ is an implementation language, not an execution site. C++ code can run on a host
CPU, launch GPU kernels, or manage a device-owned handle. Python frameworks can likewise
execute kernels on a GPU. Placement is chosen per stage from measured behavior.

## Hardware-specific boundaries

We add sensor-specific placement details only after connecting and probing the exact
hardware or reviewing its exact model documentation. No untested device capabilities are
assumed here.

## Scheduling principles

- Camera callbacks only capture metadata and transfer ownership into a bounded queue.
- Live control prefers a deadline and a latest-wins/drop-oldest policy over a growing
  latency backlog.
- Recording uses a separate provisioned queue and emits explicit gap/drop records.
- Temporal sensor synchronization and GPU inference batching remain separate operations.
- Each item in a GPU batch keeps its own measurement header and acquisition time.
- Upload/map once where practical; keep compatible dense operations together on the GPU.
- Return compact observations to the CPU instead of unnecessary dense intermediate data.
- Never claim zero-copy without instrumenting actual copies and buffer lifetimes.

Placement decisions will use capture-to-result latency, p50/p95/p99 jitter, effective
rate, drop reasons, transfer count, memory pressure, power, and task-level quality.

## Deliberately deferred

Lesson 1 does not introduce CMake, pybind/nanobind, CUDA kernels, DLPack, DMA-BUF, ROS 2
loaned messages, buffer pools, or scheduling code. Those mechanisms should implement
stable semantics rather than define them.
