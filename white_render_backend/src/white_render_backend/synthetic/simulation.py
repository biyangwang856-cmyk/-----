"""A constrained, real-dynamics Franka episode generator.

Heavy simulation/media imports are deliberately lazy. The robot and cube are
initialized before t=0; during the episode only joint motors and physics steps
move them. All observations and GT come from the same final Bullet scene.
"""
from __future__ import annotations

from fractions import Fraction
import hashlib
from importlib.metadata import version
import json
import math
from pathlib import Path

from ..domain import EngineError, require


FPS, WIDTH, HEIGHT, COUNT, PHYSICS_HZ = 24, 864, 480, 120, 240
TEMPLATE_VERSION = "franka_table_pick_place_v1"
COLORS = {"red": [.85, .05, .025, 1], "blue": [.03, .15, .85, 1], "green": [.04, .65, .12, 1]}
JOINT_INDICES = list(range(7)) + [9, 10]


def _json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def _jsonl(path, rows):
    Path(path).write_text("".join(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n" for row in rows), encoding="utf-8")


def _hash(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def capabilities():
    """Availability only; importing this does not connect to a simulator."""
    from importlib.util import find_spec
    ready = all(find_spec(name) is not None for name in ("pybullet", "pybullet_data", "av", "PIL"))
    return {"available": ready, "reason": None if ready else "Install the simulation optional dependencies", "template_version": TEMPLATE_VERSION}


def validate_spec(spec):
    required = {"robot": "franka_panda", "template": "pick_place", "destination": "right", "scene": "table", "camera": "fixed", "fps": FPS, "width": WIDTH, "height": HEIGHT, "duration_seconds": 5, "seed": 7}
    for key, expected in required.items():
        actual = spec.get(key, expected)
        require(actual == expected and not isinstance(actual, bool), "SIMULATION_SPEC", f"{key} supports only {expected!r}")
    require(spec.get("object_color", "red") in COLORS, "SIMULATION_SPEC", "object_color must be red, blue or green")
    return {**required, "schema_version": 1, "object_color": spec.get("object_color", "red")}


def _lerp(a, b, t):
    t = min(1., max(0., t))
    w = t * t * (3. - 2. * t)
    return [(1 - w) * x + w * y for x, y in zip(a, b)]


def command_at(t, xy):
    """Open-loop template schedule; no future measured observations are used."""
    down, high = [*xy, .045], [*xy, .205]
    dest = [xy[0], xy[1] + .18]
    if t < .7:
        return _lerp(high, down, t / .7), .04, "approach"
    if t < 1.3:
        return down, max(0., .04 * (1 - (t - .7) / .5)), "close"
    if t < 1.9:
        return _lerp(down, high, (t - 1.3) / .6), 0., "lift"
    if t < 2.6:
        return _lerp(high, [*dest, .205], (t - 1.9) / .7), 0., "translate"
    if t < 3.3:
        return _lerp([*dest, .205], [*dest, .045], (t - 2.6) / .7), 0., "lower"
    if t < 3.9:
        return [*dest, .045], min(.04, .04 * (t - 3.3) / .4), "release"
    if t < 4.6:
        return _lerp([*dest, .045], [*dest, .205], (t - 3.9) / .7), .04, "retreat"
    return [*dest, .205], .04, "hold"


def _source_validation(states, destination, lower, upper):
    import numpy as np
    positions = np.asarray([s["object_position_m"] for s in states])
    lift = float(positions[:, 2].max() - positions[0, 2])
    error = float(np.linalg.norm(positions[-1, :2] - destination))
    bilateral = sum(s["finger_contact_links"] == [9, 10] for s in states)
    max_speed = float(max(np.linalg.norm(s["object_linear_velocity_m_s"]) for s in states[-10:]))
    q = np.asarray([s["joint_position"] for s in states])
    finite = bool(np.isfinite(q).all() and np.isfinite(positions).all())
    collisions = [s["state_index"] for s in states if s["unexpected_collision_count"]]
    # A milliradian/millimetre tolerance admits numerical solver excursions,
    # not a changed joint limit. Raw joint state and limits remain exported.
    criteria = {"finite_states": finite, "joint_limits_within_0_001_tolerance": bool(((q >= np.asarray(lower) - .001) & (q <= np.asarray(upper) + .001)).all()),
                "lift_over_10cm": lift > .10, "bilateral_finger_contact_observed": bilateral > 5,
                "final_target_error_under_3cm": error < .03, "final_on_table": abs(float(positions[-1, 2]) - .025) < .012,
                "final_gripper_released": all(s["contact_count"] == 0 for s in states[-10:]),
                "final_object_speed_under_2cm_s": max_speed < .02, "no_unexpected_collisions": not collisions}
    metrics = {"lift_m": lift, "destination_error_m": error, "bilateral_contact_states": bilateral,
               "final_object_position_m": positions[-1].tolist(), "final_speed_window_max_m_s": max_speed,
               "unexpected_collision_state_indices": collisions}
    return {"status": "pass" if all(criteria.values()) else "fail", "source_task_pass": all(criteria.values()),
            "criteria": criteria, "metrics": metrics, "scope": "One constrained scripted physics episode; not a learned-policy or real-world benchmark", "h3_fidelity": "not_evaluated"}


def generate_scene(output_dir: Path, spec: dict, progress=None) -> dict:
    """Generate a new bundle in an empty directory and return relative metadata.

    Cancellation is provided by a progress callback that may raise. Exceptions
    close the encoder and disconnect this physics client; no global connection
    is reused. Partial output is for the job layer to retain/remove safely.
    """
    spec = validate_spec(spec)
    try:
        import av
        import numpy as np
        from PIL import Image
        import pybullet as p
        import pybullet_data
    except ImportError as exc:
        raise EngineError("SIMULATION_UNAVAILABLE", "Install white-render-backend[simulation] in the worker environment", status=503) from exc
    output = Path(output_dir)
    require(not output.exists() or not any(output.iterdir()), "SIMULATION_OUTPUT", "Output directory must be empty")
    for subdir in ("media", "depth", "instance_seg", "semantic_seg"):
        (output / subdir).mkdir(parents=True, exist_ok=True)
    pulse = progress or (lambda event: None)
    pulse({"stage": "simulation_initializing", "frames_done": 0, "frames_total": COUNT})
    cid = p.connect(p.DIRECT)
    require(cid >= 0, "SIMULATION_UNAVAILABLE", "Could not connect to a DIRECT physics client", status=503)
    encoder = None
    try:
        data_root = Path(pybullet_data.getDataPath())
        p.setAdditionalSearchPath(str(data_root), physicsClientId=cid)
        p.setGravity(0, 0, -9.81, physicsClientId=cid)
        p.setTimeStep(1 / PHYSICS_HZ, physicsClientId=cid)
        p.setPhysicsEngineParameter(numSolverIterations=100, deterministicOverlappingPairs=1, physicsClientId=cid)
        objects = []
        def box(name, center, half, *, mass=0., color=(.78, .78, .78, 1.), instance_id, class_id, role):
            collision = p.createCollisionShape(p.GEOM_BOX, halfExtents=half, physicsClientId=cid)
            visual = p.createVisualShape(p.GEOM_BOX, halfExtents=half, rgbaColor=color, physicsClientId=cid)
            body = p.createMultiBody(mass, collision, visual, basePosition=center, physicsClientId=cid)
            objects.append({"name": name, "instance_id": instance_id, "class_id": class_id, "role": role,
                            "body_id": body, "half_extents_m": list(half), "initial_position_m": list(center), "mass_kg": mass, "static": mass == 0})
            return body
        floor = box("floor", [.25, 0, -.87], [2, 2, .03], instance_id=1, class_id=1, role="background")
        table = box("worktable", [.25, 0, -.04], [.8, .55, .04], instance_id=2, class_id=2, role="support")
        for index, (x, y) in enumerate(((-.43, -.44), (-.43, .44), (.93, -.44), (.93, .44))):
            box(f"table_leg_{index}", [x, y, -.46], [.035, .035, .38], instance_id=3 + index, class_id=2, role="support")
        robot = p.loadURDF("franka_panda/panda.urdf", useFixedBase=True, physicsClientId=cid)
        robot_instance = 10
        objects.append({"name": "franka_panda", "instance_id": robot_instance, "class_id": 3, "role": "robot", "body_id": robot,
                        "asset": "pybullet_data/franka_panda/panda.urdf", "asset_sha256": _hash(data_root / "franka_panda/panda.urdf"), "fixed_base": True})
        xy = [.45, float(np.random.default_rng(spec["seed"]).uniform(-.035, .035))]
        destination = [xy[0], xy[1] + .18]
        cube = box("target_cube", [*xy, .025], [.025] * 3, mass=.075, color=COLORS[spec["object_color"]], instance_id=20, class_id=4, role="target")
        p.changeDynamics(cube, -1, lateralFriction=1.2, physicsClientId=cid)
        joints = JOINT_INDICES
        infos = [p.getJointInfo(robot, j, physicsClientId=cid) for j in joints]
        joint_names = [j[1].decode() for j in infos]
        lower, upper = [j[8] for j in infos], [j[9] for j in infos]
        orient = p.getQuaternionFromEuler([math.pi, 0, 0])
        q0 = [0, -.4, 0, -2.3, 0, 2., .8, .04, .04]
        for j, q in zip(joints, q0):
            p.resetJointState(robot, j, q, physicsClientId=cid)
        p.setJointMotorControlArray(robot, joints, p.POSITION_CONTROL, targetPositions=q0, forces=[87] * 7 + [20, 20], physicsClientId=cid)
        for _ in range(120):
            p.stepSimulation(physicsClientId=cid)
        initial = p.calculateInverseKinematics(robot, 11, [*xy, .205], orient, maxNumIterations=200, residualThreshold=1e-6, physicsClientId=cid)
        for j, q in zip(joints, list(initial[:7]) + [.04, .04]):
            p.resetJointState(robot, j, q, physicsClientId=cid)
        p.setJointMotorControlArray(robot, joints, p.POSITION_CONTROL, targetPositions=list(initial[:7]) + [.04, .04], forces=[87] * 7 + [25, 25], physicsClientId=cid)
        for _ in range(120):
            p.stepSimulation(physicsClientId=cid)
        # Initialization ends here. No reset, attachment or object teleport is
        # permitted below: observations represent true simulated motion.
        cam = {"eye_world_m": [1.15, -1.25, .95], "target_world_m": [.43, .05, .15], "up_world": [0, 0, 1],
               "near_m": .02, "far_m": 5., "vertical_fov_deg": 45.}
        view = p.computeViewMatrix(cam["eye_world_m"], cam["target_world_m"], cam["up_world"])
        projection = p.computeProjectionMatrixFOV(cam["vertical_fov_deg"], WIDTH / HEIGHT, cam["near_m"], cam["far_m"])
        cv_from_gl = np.diag([1., -1., -1., 1.])
        world_to_cv = cv_from_gl @ np.asarray(view).reshape(4, 4, order="F")
        focal = HEIGHT / (2 * math.tan(math.radians(cam["vertical_fov_deg"]) / 2))
        cam.update({"width": WIDTH, "height": HEIGHT, "K": [[focal, 0., WIDTH / 2], [0., focal, HEIGHT / 2], [0., 0., 1.]],
                    "world_to_camera_opencv": world_to_cv.tolist(), "camera_to_world_opencv": np.linalg.inv(world_to_cv).tolist(),
                    "view_opengl_column_major": list(view), "projection_opengl_column_major": list(projection),
                    "pixel_convention": "Top-left image; x right, y down; K maps to pixel-edge coordinates (pixel i,j center is i+0.5,j+0.5)",
                    "coordinate_convention": "World z-up metres; camera OpenCV x-right y-down z-forward; quaternions xyzw", "depth_convention": "Positive camera axial z depth in metres; far plane denotes no hit"})
        object_map = {o["body_id"]: o for o in objects}
        instance_classes = {o["instance_id"]: o["class_id"] for o in objects}
        # Explicit links allow downstream users to reconstruct raw renderer IDs.
        mappings = []
        for body, obj in object_map.items():
            for link in range(-1, p.getNumJoints(body, physicsClientId=cid)):
                mappings.append({"body_id": body, "link_index": link, "instance_id": obj["instance_id"], "class_id": obj["class_id"]})
        static_bodies = {o["body_id"] for o in objects if o.get("static")}
        def observe_state(index):
            state = p.getJointStates(robot, joints, physicsClientId=cid)
            ee = p.getLinkState(robot, 11, computeForwardKinematics=True, computeLinkVelocity=True, physicsClientId=cid)
            pos, quat = p.getBasePositionAndOrientation(cube, physicsClientId=cid)
            linear, angular = p.getBaseVelocity(cube, physicsClientId=cid)
            contacts = p.getContactPoints(bodyA=robot, bodyB=cube, physicsClientId=cid)
            unexpected = []
            for body in static_bodies:
                for c in p.getContactPoints(bodyA=robot, bodyB=body, physicsClientId=cid):
                    # The fixed base/first link may touch its table mounting plane.
                    if body == table and c[3] in (-1, 0):
                        continue
                    if c[8] < -1e-4:
                        unexpected.append({"other_body_id": body, "robot_link": c[3], "distance_m": c[8], "normal_force_N": c[9]})
                if body != table:
                    for c in p.getContactPoints(bodyA=cube, bodyB=body, physicsClientId=cid):
                        if c[8] < -1e-4:
                            unexpected.append({"other_body_id": body, "object_link": c[3], "distance_m": c[8], "normal_force_N": c[9]})
            return {"state_index": index, "time_s": index / FPS, "physics_step": index * 10,
                    "joint_position": [s[0] for s in state], "joint_velocity": [s[1] for s in state], "applied_motor_effort": [s[3] for s in state],
                    "ee_position_m": list(ee[4]), "ee_quaternion_xyzw": list(ee[5]), "ee_linear_velocity_m_s": list(ee[6]), "ee_angular_velocity_rad_s": list(ee[7]),
                    "object_position_m": list(pos), "object_quaternion_xyzw": list(quat), "object_linear_velocity_m_s": list(linear), "object_angular_velocity_rad_s": list(angular),
                    "contact_count": len(contacts), "finger_contact_links": sorted({c[3] for c in contacts if c[3] in (9, 10)}),
                    "contacts": [{"robot_link": c[3], "object_link": c[4], "position_robot_world_m": list(c[5]), "position_object_world_m": list(c[6]),
                                  "normal_on_object_world": list(c[7]), "distance_m": c[8], "normal_force_N": c[9]} for c in contacts],
                    "unexpected_collision_count": len(unexpected), "unexpected_collisions": unexpected}
        rows, actions, observations = [observe_state(0)], [], []
        encoder = av.open(str(output / "media/white_model.mp4"), "w")
        stream = encoder.add_stream("libx264", rate=FPS)
        stream.width, stream.height, stream.pix_fmt = WIDTH, HEIGHT, "yuv420p"
        stream.options = {"crf": "18", "preset": "veryfast"}
        for i in range(COUNT):
            pulse({"stage": "simulating", "frames_done": i, "frames_total": COUNT})
            _, _, rgba, db, seg = p.getCameraImage(WIDTH, HEIGHT, viewMatrix=view, projectionMatrix=projection,
                    renderer=p.ER_TINY_RENDERER, flags=p.ER_SEGMENTATION_MASK_OBJECT_AND_LINKINDEX, shadow=1, lightDirection=[-1, -1, 3], physicsClientId=cid)
            rgb = np.asarray(rgba, dtype=np.uint8).reshape(HEIGHT, WIDTH, 4)[..., :3]
            depthbuffer = np.asarray(db, dtype=np.float32).reshape(HEIGHT, WIDTH)
            depth = cam["far_m"] * cam["near_m"] / (cam["far_m"] - (cam["far_m"] - cam["near_m"]) * depthbuffer)
            raw = np.asarray(seg, dtype=np.int32).reshape(HEIGHT, WIDTH)
            bodies = raw & ((1 << 24) - 1)
            instances = np.full(raw.shape, -1, dtype=np.int16)  # unknown stays -1
            instances[raw == -1] = 0  # only renderer background maps to background
            semantic = instances.copy()
            for body, obj in object_map.items():
                mask = (raw >= 0) & (bodies == body)
                instances[mask] = obj["instance_id"]
                semantic[mask] = obj["class_id"]
            np.savez_compressed(output / "depth" / f"{i:06d}.npz", depth_m=depth.astype(np.float32))
            np.savez_compressed(output / "instance_seg" / f"{i:06d}.npz", instance_id=instances, renderer_body_link_id=raw)
            np.savez_compressed(output / "semantic_seg" / f"{i:06d}.npz", class_id=semantic)
            if i in (0, 48, 104):
                Image.fromarray(rgb).save(output / ("preview.jpg" if i == 0 else f"keyframe_{i:03d}.jpg"), quality=90)
            vf = av.VideoFrame.from_ndarray(rgb, format="rgb24")
            vf.pts, vf.time_base = i, Fraction(1, FPS)
            for packet in stream.encode(vf):
                encoder.mux(packet)
            observations.append({"frame_index": i, "pts": i, "time_base": [1, FPS], "time_s": i / FPS, "state_index": i,
                                 "camera_id": "fixed", "video": "media/white_model.mp4", "depth": f"depth/{i:06d}.npz",
                                 "instance_seg": f"instance_seg/{i:06d}.npz", "semantic_seg": f"semantic_seg/{i:06d}.npz"})
            desired, aperture, phase = command_at(i / FPS, xy)
            ik = p.calculateInverseKinematics(robot, 11, desired, orient, maxNumIterations=100, residualThreshold=1e-5, physicsClientId=cid)
            command = list(ik[:7]) + [aperture, aperture]
            require(all(math.isfinite(v) for v in command), "SIMULATION_COMMAND", "Nonfinite IK output")
            # Retain the raw target; never silently clip it to mask an invalid controller.
            require(all(lo - .001 <= q <= hi + .001 for q, lo, hi in zip(command, lower, upper)), "SIMULATION_COMMAND", "IK target exceeds joint limits")
            actions.append({"action_index": i, "issued_at_s": i / FPS, "effective_interval_s": [i / FPS, (i + 1) / FPS],
                            "observation_index": i, "state_before": i, "state_after": i + 1, "mode": "joint_position_target",
                            "joint_names": joint_names, "units": ["rad"] * 7 + ["m"] * 2, "target": command, "forces": [87.] * 7 + [25., 25.],
                            "force_units": ["N*m"] * 7 + ["N"] * 2, "planned_phase": phase, "planned_ee_position_m": desired})
            p.setJointMotorControlArray(robot, joints, p.POSITION_CONTROL, targetPositions=command, forces=[87] * 7 + [25, 25], positionGains=[.12] * 7 + [.3, .3], physicsClientId=cid)
            for _ in range(10):
                p.stepSimulation(physicsClientId=cid)
            rows.append(observe_state(i + 1))
        for packet in stream.encode():
            encoder.mux(packet)
        encoder.close()
        encoder = None
        validation = _source_validation(rows, destination, lower, upper)
        events = []
        for name, predicate, criterion in (
            ("first_contact", lambda s: s["contact_count"] > 0, "robot-cube contact_count > 0"),
            ("first_lift_over_5cm", lambda s: s["object_position_m"][2] - rows[0]["object_position_m"][2] > .05, "cube z exceeds initial z by 0.05m"),
            ("release_after_transport", lambda s: s["time_s"] > 3.3 and s["contact_count"] == 0, "t > 3.3s and robot-cube contact_count == 0")):
            found = next((s for s in rows if predicate(s)), None)
            events.append({"name": name, "observed": found is not None, "state_index": found["state_index"] if found else None,
                           "time_s": found["time_s"] if found else None, "criterion": criterion, "evidence": "states.jsonl"})
        channels = {
            "rgb": {"available": True, "provenance": "simulated", "path": "media/white_model.mp4", "count": COUNT, "shape": [HEIGHT, WIDTH, 3], "dtype": "uint8"},
            "depth": {"available": True, "provenance": "simulated", "glob": "depth/*.npz", "key": "depth_m", "unit": "m", "dtype": "float32", "shape": [HEIGHT, WIDTH], "count": COUNT},
            "instance_seg": {"available": True, "provenance": "simulated", "glob": "instance_seg/*.npz", "key": "instance_id", "dtype": "int16", "count": COUNT},
            "semantic_seg": {"available": True, "provenance": "mapped_from_renderer_ids", "glob": "semantic_seg/*.npz", "key": "class_id", "dtype": "int16", "count": COUNT},
            "state": {"available": True, "provenance": "simulated", "path": "states.jsonl", "count": COUNT + 1},
            "action": {"available": True, "provenance": "issued_simulator_command", "path": "actions.jsonl", "count": COUNT},
            "contact": {"available": True, "provenance": "simulated", "path": "states.jsonl", "sampling": "24Hz boundary contacts; substep-only contacts may be absent"},
            "normals": {"available": False, "reason": "not_exported_in_v1"}, "optical_flow": {"available": False, "reason": "not_exported_in_v1"}}
        metadata = {"width": WIDTH, "height": HEIGHT, "fps": FPS, "frame_count": COUNT, "duration_seconds": 5, "state_count": COUNT + 1,
                    "action_count": COUNT, "physics_hz": PHYSICS_HZ, "backend": "PyBullet DIRECT + CPU TinyRenderer", "pybullet_version": version("pybullet"), "template_version": TEMPLATE_VERSION}
        task = {**spec, "template_version": TEMPLATE_VERSION, "initial_target_xy_m": xy, "destination_xy_m": destination,
                "destination_definition": "right template is a +0.18m world Y translation; fixed scene convention", "success_criteria": list(validation["criteria"]),
                "controller": "scripted IK + joint position motors; no attachment or teleport after t=0"}
        manifest = {"schema": "episode_bundle_v1", "schema_version": 1, "template_version": TEMPLATE_VERSION, "metadata": metadata,
                    "channels": channels, "source_validation": "source_validation.json", "allowed_usage": ["preview", "research"],
                    "training_admission": "not_certified", "ecp_compatibility": "unverified", "provenance": "new_physical_simulation",
                    "time_contract": {"observation_count": COUNT, "state_count": COUNT + 1, "action_count": COUNT, "fps": FPS,
                                      "formula": "o[i],s[i] -> a[i] on [i/24,(i+1)/24) -> s[i+1]", "terminal_state_time_s": 5}}
        _json(output / "task.json", task)
        _json(output / "scene.json", {"schema_version": 1, "objects": objects, "world": "z_up_metres", "gravity_m_s2": [0, 0, -9.81], "seed": 7, "simulation_hz": PHYSICS_HZ, "solver_iterations": 100})
        _json(output / "embodiment.json", {"schema_version": 1, "id": "franka_panda_pybullet", "ecp_ref": None, "ecp_compatibility": "unverified", "joint_names": joint_names,
              "joint_indices": joints, "joint_position_units": ["rad"] * 7 + ["m"] * 2, "joint_velocity_units": ["rad/s"] * 7 + ["m/s"] * 2,
              "effort_units": ["N*m"] * 7 + ["N"] * 2, "joint_lower_limits": lower, "joint_upper_limits": upper, "ee_link_index": 11,
              "pose_quaternion_order": "xyzw", "world_frame": "z_up_metres", "urdf_sha256": _hash(data_root / "franka_panda/panda.urdf")})
        _json(output / "ontology.json", {"unknown_id": -1, "background_id": 0, "classes": {"1": "floor", "2": "table", "3": "robot", "4": "cube"},
              "instance_to_class": instance_classes, "renderer_mappings": mappings, "raw_segmentation_formula": "body_id + ((link_index + 1) << 24); -1 is renderer background"})
        _json(output / "cameras.json", {"fixed": cam})
        _json(output / "manifest.json", manifest)
        _json(output / "source_validation.json", validation)
        _jsonl(output / "observations.jsonl", observations)
        _jsonl(output / "actions.jsonl", actions)
        _jsonl(output / "states.jsonl", rows)
        _jsonl(output / "events.jsonl", events)
        files = sorted(pth.relative_to(output).as_posix() for pth in output.rglob("*") if pth.is_file())
        _json(output / "checksums.json", {"algorithm": "sha256", "files": [{"path": name, "size_bytes": (output / name).stat().st_size, "sha256": _hash(output / name)} for name in files]})
        files.append("checksums.json")
        pulse({"stage": "simulation_complete", "frames_done": COUNT, "frames_total": COUNT, "source_validation": validation["status"]})
        return {"files": sorted(files), "validation": validation, "channels": channels, "metadata": metadata, "manifest": manifest}
    finally:
        try:
            if encoder is not None:
                encoder.close()
        finally:
            p.disconnect(physicsClientId=cid)
