"""Offline Open3D global-registration and ICP teaching viewer (all inputs in metres)."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np


def load_session(directory, view, o3d):
    """Load the actual model-calibration capture format, not a live camera."""
    import cv2

    header = json.loads((directory / "session.json").read_text())
    record = json.loads((directory / f"{view}.json").read_text())
    depth = np.load(directory / f"{view}.depth.npy")
    k = np.asarray(header["camera"]["camera_matrix"])
    v, u = np.indices(depth.shape)
    valid = np.isfinite(depth) & (depth > 0.25) & (depth < 2.5)
    z = depth[valid]
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(np.column_stack((
        (u[valid] - k[0, 2]) * z / k[0, 0],
        (v[valid] - k[1, 2]) * z / k[1, 1], z,
    )))
    image = cv2.imread(str(directory / f"{view}.color.png"))
    if image is not None and image.shape[:2] == depth.shape:
        cloud.colors = o3d.utility.Vector3dVector(image[valid, ::-1] / 255.0)
    return cloud, header, record


def model_cloud(xml, record, o3d):
    """Sample visual meshes in base/world coordinates at this capture's joint angles."""
    import mujoco

    model = mujoco.MjModel.from_xml_path(str(xml))
    data = mujoco.MjData(model)
    joints = {key: value for arm in record.get("joints", {}).values()
              for key, value in arm.items()}
    for name, value in joints.items():
        index = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if index < 0:
            raise ValueError(f"Recorded joint missing from model: {name}")
        data.qpos[model.jnt_qposadr[index]] = value
    mujoco.mj_forward(model, data)
    result = o3d.geometry.PointCloud()
    for geom in range(model.ngeom):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom) or ""
        body = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY,
                               int(model.geom_bodyid[geom])) or ""
        if "_visual" not in name or "finger" in name:
            continue
        static = body.startswith("openarm_body") or body.endswith("_link0")
        if not static and not any(body.startswith(f"openarm_{arm}")
                                  for arm in record.get("joints", {})):
            continue
        if model.geom_type[geom] != mujoco.mjtGeom.mjGEOM_MESH:
            continue
        mesh_id = model.geom_dataid[geom]
        va, vn = model.mesh_vertadr[mesh_id], model.mesh_vertnum[mesh_id]
        fa, fn = model.mesh_faceadr[mesh_id], model.mesh_facenum[mesh_id]
        mesh = o3d.geometry.TriangleMesh(
            o3d.utility.Vector3dVector(model.mesh_vert[va:va + vn].astype(float)),
            o3d.utility.Vector3iVector(model.mesh_face[fa:fa + fn]),
        )
        mesh.compute_vertex_normals()
        transform = np.eye(4)
        transform[:3, :3] = data.geom_xmat[geom].reshape(3, 3)
        transform[:3, 3] = data.geom_xpos[geom]
        mesh.transform(transform)
        result += mesh.sample_points_uniformly(max(100, int(mesh.get_surface_area() * 80000)))
    return result


def prepare(cloud, voxel, o3d):
    cloud = cloud.voxel_down_sample(voxel)
    if len(cloud.points) < 20:
        raise ValueError("Too few points; check units, crop, and voxel size")
    cloud.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=2 * voxel, max_nn=30))
    feature = o3d.pipelines.registration.compute_fpfh_feature(
        cloud, o3d.geometry.KDTreeSearchParamHybrid(radius=5 * voxel, max_nn=100))
    return cloud, feature


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--demo", action="store_true", help="download Open3D example clouds")
    mode.add_argument("--source", type=Path, help="camera cloud PLY/PCD, in metres")
    mode.add_argument("--session", type=Path, help="saved registration session directory")
    mode.add_argument("--replay", type=Path, help="replay an output directory without recomputing")
    parser.add_argument("--comparison", type=Path,
                        help="saved world_from_camera JSON for visual comparison, not ground truth")
    parser.add_argument("--target", type=Path, help="posed model cloud in base frame, metres")
    parser.add_argument("--model-xml", type=Path, help="local OpenArm MuJoCo scene and assets")
    parser.add_argument("--view", default="view-00")
    parser.add_argument("--voxel", type=float, default=0.02, help="feature voxel size, metres")
    parser.add_argument("--evaluation-distance", type=float, default=0.01)
    parser.add_argument("--iterations", type=int, default=15, help="ICP steps per scale")
    parser.add_argument("--crop-camera", nargs=6, type=float,
                        metavar=("XMIN", "YMIN", "ZMIN", "XMAX", "YMAX", "ZMAX"))
    parser.add_argument("--output", type=Path, default=Path("recordings/registration_lab"))
    parser.add_argument("--no-gui", action="store_true")
    args = parser.parse_args(argv)
    if args.voxel <= 0 or args.evaluation_distance <= 0 or args.iterations < 1:
        parser.error("voxel, evaluation distance, and iterations must be positive")
    try:
        import open3d as o3d
    except ImportError:
        parser.error("Install the registration extra: pip install -e '.[registration]'")
    o3d.utility.random.seed(7)
    if args.replay:
        report = json.loads((args.replay / "stages.json").read_text())
        source = o3d.io.read_point_cloud(str(args.replay / "source.ply"))
        target = o3d.io.read_point_cloud(str(args.replay / "target.ply"))
        view_stages(source, target, report["stages"], args.voxel,
                    report["evaluation_distance_m"], o3d, args.comparison)
        return
    if args.demo:
        dataset = o3d.data.DemoICPPointClouds()
        source, target = [o3d.io.read_point_cloud(p) for p in dataset.paths[:2]]
    elif args.session:
        source, header, record = load_session(args.session, args.view, o3d)
        xml = args.model_xml or Path(header["model_xml"])
        if args.target:
            target = o3d.io.read_point_cloud(str(args.target))
        elif xml.is_file():
            target = model_cloud(xml, record, o3d)
        else:
            parser.error("Model XML unavailable here; supply --model-xml or --target")
        print(f"Captured joints: {record.get('joints', {})}")
    else:
        if args.target is None:
            parser.error("--source requires --target")
        source = o3d.io.read_point_cloud(str(args.source))
        target = o3d.io.read_point_cloud(str(args.target))
    if args.crop_camera:
        source = source.crop(o3d.geometry.AxisAlignedBoundingBox(
            args.crop_camera[:3], args.crop_camera[3:]))
    if not len(source.points) or not len(target.points):
        parser.error("Empty source or target cloud")
    reg = o3d.pipelines.registration
    src, sf = prepare(source, args.voxel, o3d)
    dst, tf = prepare(target, args.voxel, o3d)
    # Every stage is measured on the SAME cloud resolution and distance threshold.
    evaluation_source = source.voxel_down_sample(args.voxel / 4)
    evaluation_target = target.voxel_down_sample(args.voxel / 4)
    stages = []

    def save_stage(label, transform):
        fit = reg.evaluate_registration(evaluation_source, evaluation_target,
                                        args.evaluation_distance, transform)
        moved = copy.deepcopy(evaluation_source).transform(transform)
        distances = np.asarray(moved.compute_point_cloud_distance(evaluation_target))
        stage = {"label": label, "source_to_target": transform.tolist(),
                 "fitness": fit.fitness,
                 "inlier_rmse_mm": fit.inlier_rmse * 1000 if fit.fitness else None,
                 "all_source_distance_p95_mm": float(np.percentile(distances, 95) * 1000)}
        stages.append(stage)
        print(f"{label}: fitness={fit.fitness:.3f}, "
              f"inlier RMSE={stage['inlier_rmse_mm']} mm", flush=True)

    save_stage("Unaligned", np.eye(4))
    print("Computing FPFH + RANSAC global alignment...", flush=True)
    global_fit = reg.registration_ransac_based_on_feature_matching(
        src, dst, sf, tf, True, args.voxel * 1.5,
        reg.TransformationEstimationPointToPoint(False), 3,
        [reg.CorrespondenceCheckerBasedOnEdgeLength(0.9),
         reg.CorrespondenceCheckerBasedOnDistance(args.voxel * 1.5)],
        reg.RANSACConvergenceCriteria(100000, 0.999))
    transform = global_fit.transformation
    save_stage("Global: FPFH + RANSAC", transform)
    if global_fit.fitness == 0:
        raise RuntimeError("Global registration found no overlap; inspect crop and geometry")
    for scale, voxel in enumerate((args.voxel, args.voxel / 2, args.voxel / 4), 1):
        a, _ = prepare(source, voxel, o3d)
        b, _ = prepare(target, voxel, o3d)
        estimator = reg.TransformationEstimationPointToPlane(reg.HuberLoss(voxel))
        for step in range(args.iterations):
            result = reg.registration_icp(a, b, 1.5 * voxel, transform, estimator,
                                          reg.ICPConvergenceCriteria(max_iteration=1))
            if result.fitness == 0:
                raise RuntimeError("ICP lost all correspondences; inspect global alignment")
            transform = result.transformation
            save_stage(f"ICP scale {scale}, iteration {step + 1}", transform)
    args.output.mkdir(parents=True, exist_ok=True)
    report = {"demo": args.demo, "session": str(args.session) if args.session else None,
              "view": args.view, "evaluation_distance_m": args.evaluation_distance,
              "note": "Single-view geometric fit; not independently validated calibration.",
              "stages": stages}
    (args.output / "stages.json").write_text(json.dumps(report, indent=2, allow_nan=False))
    o3d.io.write_point_cloud(str(args.output / "source.ply"), source)
    o3d.io.write_point_cloud(str(args.output / "target.ply"), target)
    print(f"Saved stages and input clouds to {args.output}")
    if args.no_gui:
        return

    view_stages(evaluation_source, evaluation_target, stages, args.voxel,
                args.evaluation_distance, o3d, args.comparison)


def view_stages(source, target, stages, voxel, evaluation_distance, o3d, comparison=None):
    """Replay exactly the saved transformations, optionally alongside a prior calibration."""
    src = source.voxel_down_sample(voxel / 4)
    dst = target.voxel_down_sample(voxel / 4)
    stages = list(stages)
    final_icp = len(stages) - 1
    if comparison:
        transform = np.asarray(json.loads(comparison.read_text())["world_from_camera"])
        fit = o3d.pipelines.registration.evaluate_registration(
            src, dst, evaluation_distance, transform)
        stages.append({"label": "Saved plate calibration (comparison, not ground truth)",
                       "source_to_target": transform.tolist(), "fitness": fit.fitness,
                       "inlier_rmse_mm": fit.inlier_rmse * 1000 if fit.fitness else None})
    print("N: next, B: back, 1: unaligned, 2: global, 3: final ICP, 4: comparison")
    print("C: RGB, D: distances; mouse: rotate/zoom. Metrics are not calibration accuracy.")
    print("Mouse: rotate/zoom. Metrics are geometric fit, not calibration accuracy.")
    display = copy.deepcopy(src)
    reference = copy.deepcopy(dst).paint_uniform_color([0.1, 0.8, 0.85])
    state = {"index": 0, "color": "uniform"}
    viewer = o3d.visualization.VisualizerWithKeyCallback()
    if not viewer.create_window("OpenArm registration | N/B steps | 1 raw 2 global 3 ICP 4 plate | C RGB D error"):
        raise RuntimeError("Could not create Open3D window; use --no-gui for offline computation")
    viewer.add_geometry(display)
    viewer.add_geometry(reference)
    viewer.add_geometry(o3d.geometry.TriangleMesh.create_coordinate_frame(size=voxel * 5))

    def update(vis, index=None, color=None):
        if index is not None:
            state["index"] = max(0, min(len(stages) - 1, index))
        if color:
            state["color"] = color
        stage = stages[state["index"]]
        moved = copy.deepcopy(src).transform(np.asarray(stage["source_to_target"]))
        display.points = moved.points
        display.colors = moved.colors
        if state["color"] == "distance":
            d = np.clip(np.asarray(moved.compute_point_cloud_distance(dst)) /
                        evaluation_distance, 0, 1)
            display.colors = o3d.utility.Vector3dVector(np.column_stack((d, 1 - d, d * 0)))
        elif state["color"] != "rgb" or not display.has_colors():
            display.paint_uniform_color([1.0, 0.55, 0.15])
        print(f"{state['index']}: {stage['label']} | overlap {stage['fitness']:.1%} | "
              f"inlier RMSE {stage['inlier_rmse_mm']} mm", flush=True)
        vis.update_geometry(display)
        vis.reset_view_point(True)
        return False

    viewer.register_key_callback(ord("N"), lambda v: update(v, state["index"] + 1))
    viewer.register_key_callback(ord("B"), lambda v: update(v, state["index"] - 1))
    for key, index in (("1", 0), ("2", 1), ("3", final_icp), ("4", len(stages) - 1)):
        viewer.register_key_callback(ord(key), lambda v, i=index: update(v, i))
    viewer.register_key_callback(ord("C"), lambda v: update(v, color="rgb"))
    viewer.register_key_callback(ord("D"), lambda v: update(v, color="distance"))
    update(viewer, len(stages) - 1 if comparison else 0)
    viewer.run()
    viewer.destroy_window()


if __name__ == "__main__":
    main()
