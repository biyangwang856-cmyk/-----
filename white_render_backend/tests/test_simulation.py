"""Physical/time/semantic acceptance at the generator's difficult boundary."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from white_render_backend.domain import EngineError
from white_render_backend.synthetic.simulation import command_at, generate_scene, validate_spec


@pytest.mark.parametrize("bad", [{"duration_seconds": 10}, {"seed": True}, {"fps": 30}, {"robot": "ur5"}, {"destination": "left"}, {"object_color": "purple"}])
def test_unsupported_physical_specs_fail_before_loading_simulator(bad):
    with pytest.raises(EngineError, match="SIMULATION_SPEC"):
        validate_spec(bad)


def test_commands_are_bounded_and_planned():
    for i in range(120):
        xyz, aperture, phase = command_at(i / 24, [.45, .01])
        assert np.isfinite(xyz).all() and 0 <= aperture <= .04
        assert .045 <= xyz[2] <= .205
        assert phase in {"approach", "close", "lift", "translate", "lower", "release", "retreat", "hold"}


def _read_lines(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines()]


@pytest.mark.integration
def test_real_episode_has_success_and_aligned_observations_actions_gt(tmp_path):
    pytest.importorskip("pybullet")
    av = pytest.importorskip("av")
    bundle = tmp_path / "episode"
    events = []
    result = generate_scene(bundle, {}, progress=events.append)
    states = _read_lines(bundle / "states.jsonl")
    actions = _read_lines(bundle / "actions.jsonl")
    observations = _read_lines(bundle / "observations.jsonl")
    assert (len(states), len(actions), len(observations)) == (121, 120, 120)
    assert states[-1]["time_s"] == 5
    assert result["validation"]["status"] == "pass", result["validation"]
    assert events[-1]["stage"] == "simulation_complete"
    for i, (obs, act) in enumerate(zip(observations, actions)):
        assert obs["state_index"] == act["state_before"] == i
        assert act["state_after"] == i + 1
        assert act["effective_interval_s"] == [states[i]["time_s"], states[i + 1]["time_s"]]
        assert act["issued_at_s"] == obs["time_s"]
        assert len(act["target"]) == len(states[i]["joint_position"]) == 9
        assert act["units"] == ["rad"] * 7 + ["m"] * 2
        assert states[i + 1]["physics_step"] - states[i]["physics_step"] == 10
    with av.open(str(bundle / "media/white_model.mp4")) as clip:
        decoded = list(clip.decode(video=0))
    assert len(decoded) == 120
    assert (decoded[0].width, decoded[0].height) == (864, 480)
    assert np.allclose([float(f.pts * f.time_base) for f in decoded], np.arange(120) / 24)
    camera = json.loads((bundle / "cameras.json").read_text())["fixed"]
    k = np.asarray(camera["K"])
    cv = np.asarray(camera["world_to_camera_opencv"])
    # Project the initial cube centre through each exported camera convention.
    point = np.array(states[0]["object_position_m"] + [1.])
    xyz = cv @ point
    uv = (k @ xyz[:3])[:2] / xyz[2]
    gl = np.asarray(camera["projection_opengl_column_major"]).reshape(4, 4, order="F") @ np.asarray(camera["view_opengl_column_major"]).reshape(4, 4, order="F") @ point
    ndc = gl[:3] / gl[3]
    np.testing.assert_allclose(uv, [(ndc[0] + 1) * 864 / 2, (1 - ndc[1]) * 480 / 2], atol=1e-4)
    assert 0 < uv[0] < 864 and 0 < uv[1] < 480 and xyz[2] > 0
    with np.load(bundle / "instance_seg/000000.npz") as z:
        instance = z["instance_id"]
        assert instance.dtype == np.int16 and 20 in instance
    with np.load(bundle / "semantic_seg/000000.npz") as z:
        assert np.all(z["class_id"][instance == 20] == 4)
    with np.load(bundle / "depth/000000.npz") as z:
        assert np.isfinite(z["depth_m"]).all() and z["depth_m"].min() > 0
        assert abs(float(np.median(z["depth_m"][instance == 20])) - xyz[2]) < .06
    records = json.loads((bundle / "checksums.json").read_text())["files"]
    for record in records:
        raw = (bundle / record["path"]).read_bytes()
        assert hashlib.sha256(raw).hexdigest() == record["sha256"]
        assert len(raw) == record["size_bytes"]
    assert set(result["files"]) == {x.relative_to(bundle).as_posix() for x in bundle.rglob("*") if x.is_file()}
    assert result["manifest"]["training_admission"] == "not_certified"
    assert result["channels"]["normals"]["available"] is False


def test_cancellation_disconnects_its_client(tmp_path):
    bullet = pytest.importorskip("pybullet")
    pytest.importorskip("av")
    class Cancelled(Exception):
        pass
    def callback(event):
        if event["stage"] == "simulating":
            raise Cancelled()
    with pytest.raises(Cancelled):
        generate_scene(tmp_path / "cancelled", {}, callback)
    # A new DIRECT connection should occupy slot 0; no old client was leaked.
    client = bullet.connect(bullet.DIRECT)
    try:
        assert client == 0
    finally:
        bullet.disconnect(physicsClientId=client)
