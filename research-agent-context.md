# Research Context: Reusable World-State and Action Infrastructure

**Prepared:** 2026-09-10  
**Project:** `vision_pipeline`  
**Purpose:** Give an AI research agent enough context to investigate prior art, papers, open-source systems, and public technical disclosures relevant to a reusable multimodal perception and world-state infrastructure for embodied AI.

## 1. Project Context

We are designing a reusable perception and world-state substrate for humanoid robots, manipulation, teleoperation, drones, UGVs, autonomous cars, and other embodied systems.

The intended flow is:

```text
RGB / stereo / depth / IR / LiDAR / IMU / force / touch / proprioception
    -> timestamping and recording
    -> calibration and coordinate frames
    -> synchronization and registration
    -> perception and tracking
    -> state estimation and sensor fusion
    -> uncertainty-aware world-state belief
    -> RL / IL / world models / planning / control
```

The system should support both:

1. **Dataset and reconstruction mode:** recording aligned sensor episodes for RL, imitation learning, world-action models, NeRF, Gaussian splatting, digital twins, and evaluation.
2. **Runtime mode:** producing the same semantic observation contract online for policies, dynamical systems, planners, safety filters, and controllers.

The first application target is a manipulation experiment:

```text
Detect and track a box
    -> estimate its 3D pose and velocity
    -> attach a dynamical-system attractor
    -> move a robot hand toward it
    -> detect contact using force, touch, geometry, or proprioception
    -> update the controller
```

The broader target is a structured belief state containing objects, people, robot state, geometry, contact, forces, velocities, actions, timestamps, provenance, and uncertainty.

The Python package uses `pyproject.toml`. Python is intended for research orchestration, model integration, datasets, evaluation, and RL. C++ is intended for high-rate device drivers, ROS 2 nodes, sensor ingestion, real-time queues, zero-copy paths, GPU preprocessing, and latency-sensitive interfaces. CUDA, TensorRT, vendor SDKs, device DSPs, and edge accelerators should be used where they improve throughput or latency.

Current project contracts include:

- timestamped `SensorSample`
- `SensorKind` for RGB, depth, infrared, LiDAR, IMU, pose, detections, and features
- explicit rigid transforms and a frame tree
- nearest-neighbour synchronization with a timing budget
- a capability registry for optional model and hardware adapters

These are only the initial contracts. The research should help determine what abstractions, schemas, middleware, and algorithms are missing.

## 2. Core Research Questions

Investigate these questions with primary sources wherever possible:

### A. Common embodied perception abstraction

1. Has anyone published a broadly reusable abstraction that turns heterogeneous sensors into an uncertainty-aware world-state belief for many robot platforms?
2. Which systems separate raw measurements, calibration, synchronization, state estimation, learned perception, world representation, and control?
3. What are the best existing patterns from ROS 2, Open Robotics, NVIDIA Isaac, Autoware, OpenVINS, VINS-Fusion, Kimera, Nav2, MoveIt 2, Drake, Pinocchio, MuJoCo, and similar ecosystems?
4. Which systems support replay, deterministic evaluation, provenance, missing modalities, confidence, covariance, and time alignment?
5. What should be a universal interface, and what should remain platform-specific?
6. How are dynamic entities represented separately from persistent maps, meshes, NeRFs, and Gaussian splats?

### B. World-action models and embodied foundation models

Investigate:

- Dyna Robotics and the exact meaning of its public terms such as **world action model**, **Dyna** stack, robot foundation model, or related product names. Verify terminology from official sources before using it.
- Field AI and its public work on robot foundation models, field robotics, autonomy, multimodal perception, world models, navigation, manipulation, and deployment.
- Other relevant systems from Google DeepMind, Physical Intelligence, Figure AI, 1X, Sanctuary AI, Agility Robotics, NVIDIA, Tesla, Toyota Research Institute, Boston Dynamics, Covariant, and Berkeley/MIT groups.

For every system, determine:

1. What is publicly documented versus inferred?
2. Does it use an explicit state representation, latent state, video tokens, 3D scene representation, or a hybrid?
3. Does it predict images, features, object state, contact, rewards, actions, or full trajectories?
4. How are action, proprioception, force, touch, and exteroception aligned?
5. Is the system model-free RL, offline RL, imitation learning, behavior cloning, diffusion policy, transformer policy, model predictive control, classical control, or a hybrid?
6. What is used during training and what is used at deployment?
7. How are safety, latency, uncertainty, failures, and sensor dropouts handled?

### C. Humanoid control, especially Optimus

Research public evidence about Tesla Optimus and comparable humanoids. Do not claim private implementation details as fact.

Investigate:

- Public Tesla presentations, talks, patents, engineering posts, job descriptions, demonstrations, and interviews
- Public research or talks from Figure, 1X, Agility, Boston Dynamics, Sanctuary AI, Unitree, and Fourier Intelligence
- Whole-body control, inverse kinematics, operational-space control, impedance control, force control, model predictive control, RL, imitation learning, teleoperation, motion retargeting, and learned visuomotor policies
- How visual perception, proprioception, tactile sensing, force/torque, and contact estimation may be combined
- Which components are likely classical safety-critical controllers versus learned policies
- Whether the system uses demonstrations, simulation, reinforcement learning, fleet data, or combinations

Separate confirmed public facts, credible technical interpretation, and speculation in the report.

### D. Drones and autonomous flight

Investigate public and academic systems related to Skydio, DJI, PX4, ArduPilot, Auterion, Qualcomm Flight, NVIDIA Isaac, and research flight stacks.

Questions:

- What are the standard visual-inertial odometry, SLAM, state-estimation, planning, and control layers?
- How are stereo cameras, monocular cameras, depth sensors, LiDAR, IMU, GPS, barometers, optical flow, and event cameras fused?
- Which workloads run on flight-controller MCUs, DSPs, NPUs, GPUs, or companion computers?
- How are timing, failsafes, obstacle avoidance, and degraded modes handled?
- Which abstractions transfer to humanoids and UGVs?

Do not infer Skydio or DJI proprietary internals from marketing language. Use open flight stacks and papers for technical mechanisms, and label commercial details as unknown when necessary.

### E. Autonomous cars

Investigate Tesla Autopilot/FSD, Waymo, Mobileye, NVIDIA DRIVE, Autoware, Apollo, and related systems.

Questions:

- Camera-only versus camera-plus-LiDAR/radar architectures
- Bird's-eye-view and occupancy representations
- Object tracking, lane and map representations, motion forecasting, and planning
- Sensor calibration and fleet-scale data pipelines
- Online and offline mapping
- End-to-end policies versus modular stacks
- Safety validation and redundancy
- How representations generalize to robots, drones, and manipulation

Again, distinguish official claims, peer-reviewed results, open-source implementations, patents, and speculation.

### F. Human video, imitation learning, and world models

Investigate how human videos are used for robot learning:

- visual representation pretraining
- action and affordance discovery
- task segmentation
- human pose and hand-object interaction
- inverse dynamics and action inference
- cross-embodiment learning
- teleoperation and demonstration collection
- video-language-action models
- diffusion policies and transformer policies
- offline RL and dataset curation
- learning from human videos without robot actions

Clarify which information ordinary human video lacks: metric scale, robot actions, force, contact, calibrated pose, and embodiment. Identify methods that bridge that gap.

### G. NeRF, Gaussian splatting, and digital twins

Investigate how reconstruction systems consume and expose:

- calibrated images
- camera trajectories
- depth and LiDAR
- IMU and visual-inertial pose
- object masks and dynamic-scene segmentation
- uncertainty and temporal updates
- interaction and semantic annotations

Relevant topics include NeRF, 3D Gaussian Splatting, dynamic Gaussian splatting, SLAM with Gaussian splats, SplaTAM, Gaussian-SLAM, neural scene representations, semantic mapping, and digital twins for robotics.

Determine whether these representations are suitable for:

- persistent geometry
- localization
- planning
- manipulation
- simulation
- policy observation
- visualization

Do not assume a rendering representation is automatically a good control representation.

## 3. Candidate Research Institutions

Search for work from:

- MIT CSAIL, LIDS, Improbable AI, Robot Locomotion Group, Computer Vision Group
- Caltech AMBER Lab, CAST, computer vision and control groups
- UC Berkeley BAIR, AUTOLAB, Embodied Intelligence, Berkeley Robotics and Intelligent Machines
- ETH Zurich Robotics Systems Lab, Autonomous Systems Lab, Computer Vision and Geometry Lab
- Harvard SEAS robotics, computer vision, and machine learning groups
- University of Oxford robotics, autonomous systems, and computer vision groups
- Stanford robotics and autonomous systems groups
- Carnegie Mellon Robotics Institute
- University of Washington, University of Toronto, University of Michigan, Georgia Tech, TUM, Max Planck institutes, and EPFL where relevant

Search lab pages, institutional repositories, arXiv, Semantic Scholar, IEEE, ACM, RSS, ICRA, IROS, CoRL, NeurIPS, ICML, CVPR, ICCV, ECCV, SIGGRAPH, 3DV, and RA-L.

## 4. Search Strategy

Use combinations of these query families:

### Architecture and state

- `multimodal robot perception world state belief sensor fusion calibration synchronization`
- `robotics perception abstraction timestamped observations uncertainty provenance`
- `embodied AI world model action conditioned state representation`
- `robotics unified scene representation perception planning control`
- `robot sensor data platform replay multimodal dataset schema`

### Calibration and fusion

- `camera lidar imu extrinsic calibration robot survey`
- `multi camera synchronization heterogeneous sensors robotics`
- `visual inertial lidar fusion real time robotics`
- `camera depth lidar registration uncertainty robotics`
- `force vision contact state estimation manipulation`

### Humanoid and manipulation

- `humanoid robot visuomotor policy imitation learning reinforcement learning contact`
- `whole body teleoperation human pose retargeting humanoid`
- `vision force tactile sensor fusion manipulation`
- `dynamical systems attractor robot manipulation visual servoing`
- `world model robot manipulation action conditioned`

### Reconstruction and digital twins

- `Gaussian splatting robotics SLAM manipulation`
- `NeRF robot mapping planning digital twin`
- `dynamic Gaussian splatting robotics sensor fusion`
- `semantic 3D scene representation robot policy`

### Vehicles and drones

- `autonomous driving occupancy world model sensor fusion`
- `drone visual inertial lidar navigation onboard GPU`
- `Skydio perception autonomy technical paper`
- `PX4 visual inertial odometry companion computer`
- `Waymo world model perception planning paper`

### Companies

For each company search both:

- official technical material: `site:company-domain research perception world model`
- external evidence: `company name paper robotics perception control`

## 5. Source and Evidence Rules

Prioritize sources in this order:

1. Peer-reviewed papers and official proceedings
2. University lab pages and technical reports
3. Official company engineering papers, patents, talks, and documentation
4. Open-source repositories and design documents
5. Interviews and conference talks
6. Reputable technical journalism
7. Social media, forums, and unsourced summaries only as leads

For every important claim, record:

- exact title
- authors or organization
- year and version
- URL or DOI
- source type
- relevant section/page
- what the source actually demonstrates
- confidence: high, medium, or low
- whether it is directly applicable to this project

Never present a vendor marketing statement as an implementation fact. Never fill gaps in proprietary systems with confident guesses.

## 6. Comparison Schema

Produce one row per system or paper with these fields:

| Field | Description |
|---|---|
| System / paper | Name and link |
| Institution / company | Origin |
| Year | Publication or public release year |
| Platform | Humanoid, arm, drone, UGV, car, general |
| Sensors | RGB, stereo, depth, LiDAR, IR, IMU, force, tactile, proprioception |
| Time model | Hardware sync, software timestamps, event-based, unknown |
| Calibration | Intrinsic, extrinsic, hand-eye, online, unknown |
| State representation | Objects, occupancy, BEV, point cloud, latent, 3D scene, splats, etc. |
| Perception | Detection, segmentation, pose, tracking, features |
| Fusion | Filter, factor graph, learned fusion, late fusion, early fusion |
| Learning | RL, IL, offline RL, diffusion, transformer, self-supervised, classical |
| World model | Predicted output and horizon |
| Control | IK, impedance, MPC, dynamical system, policy, hybrid |
| Compute | CPU, GPU, DSP, NPU, edge device, cloud |
| Runtime rate | Published rate/latency if available |
| Replay/data | Dataset, logging, annotation, provenance |
| Safety | Constraints, fallback, uncertainty, validation |
| Open source | What is actually available |
| Evidence level | High, medium, low |
| Relevance | Direct, adjacent, background |

## 7. Deliverables Expected from the Research Agent

Produce the following in order:

### Deliverable 1: Executive synthesis

Answer:

- Is this reusable abstraction already published?
- Which parts are established engineering practice?
- Which parts are active research?
- Which parts are likely proprietary?
- What should this project build instead of duplicating existing infrastructure?

### Deliverable 2: Annotated bibliography

Provide 25-50 high-value papers and technical sources grouped by:

- sensor calibration and synchronization
- visual-inertial and LiDAR fusion
- 3D scene representation and mapping
- object and human pose tracking
- contact and force fusion
- humanoid control and imitation learning
- world models and action-conditioned prediction
- NeRF and Gaussian splatting for robotics
- drone and autonomous-vehicle stacks
- middleware, datasets, and deployment

For each item, include a one-paragraph explanation of why it matters and what to implement after reading it.

### Deliverable 3: System architecture comparison

Compare the proposed `vision_pipeline` abstraction with ROS 2, Isaac ROS/Isaac Sim, Autoware, Apollo, PX4, ArduPilot, OpenVINS, VINS-Fusion, Kimera, Nav2, MoveIt 2, and relevant open systems.

Identify where to integrate rather than reimplement.

### Deliverable 4: Company evidence matrix

Create separate evidence tables for:

- Dyna Robotics
- Field AI
- Tesla Optimus
- Figure
- 1X
- Agility Robotics
- Boston Dynamics
- Skydio
- DJI
- Tesla autonomous driving
- Waymo

For each, list confirmed public facts, evidence links, unknowns, and cautious technical interpretations.

### Deliverable 5: Learning curriculum

Create a step-by-step curriculum that alternates reading and implementation:

1. camera geometry and calibration
2. timestamped recording and replay
3. frame graphs and transformations
4. RGB-D and LiDAR registration
5. IMU and visual-inertial estimation
6. YOLO detection and tracking
7. DINOv2 embeddings
8. human pose and hand tracking
9. force/contact fusion
10. world-state schema and uncertainty
11. box manipulation with an attractor dynamical system
12. Gaussian-splat or NeRF reconstruction
13. RL and imitation-learning datasets
14. action-conditioned world models
15. C++/CUDA/ROS 2 deployment

Each stage should specify a paper, a small coding task, an evaluation metric, and a failure mode to inspect.

### Deliverable 6: Build recommendations

Recommend:

- package boundaries
- Python/C++ boundaries
- ROS 2 integration points
- storage and replay formats
- timestamp and clock policy
- frame naming conventions
- uncertainty representation
- model adapter interfaces
- GPU and accelerator strategy
- simulation interfaces
- test strategy
- minimum viable hardware setup

## 8. Required Conclusions

The final report must explicitly answer:

1. Is the proposed world-state pipeline technically coherent?
2. Which abstractions are common across humanoids, drones, cars, and UGVs?
3. Which abstractions cannot be universal?
4. What must be represented for RL and imitation learning?
5. What is needed to train action-conditioned world models?
6. What information is required at deployment?
7. Which parts should be classical, learned, or hybrid?
8. What should run on sensor hardware, DSP, MCU, GPU, edge computer, or cloud?
9. What should the first six implementation milestones be?
10. Which papers should be read first, and in what order?

## 9. Output Style

Be technically rigorous and concrete. Prefer tables and diagrams where helpful. Use equations only when they clarify a mechanism. Separate:

- established fact
- result reported by a paper
- engineering recommendation
- plausible interpretation
- unknown or proprietary detail

Do not produce generic AI hype. Do not claim a company uses RL, IL, a world model, a particular sensor, or a particular architecture unless a reliable public source supports it. Include dates because the systems and terminology change quickly.

The final recommendation should be practical for a researcher who wants to learn step by step while building a real, extensible Python/C++ perception and world-state package.
