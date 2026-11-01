import pytest
from pydantic import ValidationError

from white_render_backend.synthetic.contracts import (
    AssetBody, ExportBody, JobBody, PreflightBody, QuoteBody, RenderOptions, TaskSpec,
)
from white_render_backend.synthetic.specs import appearance_issues, parse_spec


@pytest.mark.parametrize("text,color", [
    ("Franka 机械臂把红色方块抓起并放到桌面右侧，固定相机，5 秒", "red"),
    ("把蓝色方块搬到桌面右侧，五秒视频", "blue"),
    ("Panda 将绿色立方体拿起，然后放置到右边，24 FPS，864×480", "green"),
    ("搬运方块到右侧", "red"),
    ("Franka pick and place red cube", "red"),
])
def test_template_selection_has_explicit_defaults(text, color):
    value = parse_spec(text, "橡木桌面、金属材质、柔和侧光")
    assert value["status"] == "ready"
    assert value["issues"] == []
    spec = value["parsed_spec"]
    assert spec["object_color"] == color
    assert spec["template"] == "pick_place"
    assert spec["destination"] == "right"
    assert value["appearance_description"] == "橡木桌面、金属材质、柔和侧光"
    assert "appearance_description" not in spec
    assert spec["fps"] * spec["duration_seconds"] == 120
    assert value["defaults"]["camera"] == "fixed"


@pytest.mark.parametrize("text", ["生成机器人视频", "Franka 抓取方块", "机械臂把东西搬到右侧"])
def test_missing_task_does_not_fabricate_ready_spec(text):
    result = parse_spec(text)
    assert result["status"] == "needs_input"
    assert result["parsed_spec"] is None


@pytest.mark.parametrize("extra,code", [
    ("使用 UR5", "UNSUPPORTED_ROBOT"),
    ("双臂协作", "UNSUPPORTED_ROBOT"),
    ("然后堆叠", "UNSUPPORTED_TASK"),
    ("还要放入盒子", "UNSUPPORTED_CONTAINER"),
    ("抓两个方块", "UNSUPPORTED_OBJECT"),
    ("放到左侧", "UNSUPPORTED_DIRECTION"),
    ("相机绕拍", "UNSUPPORTED_CAMERA"),
    ("物体宽10厘米", "UNSUPPORTED_DIMENSION"),
    ("持续10秒", "UNSUPPORTED_DURATION"),
    ("时长0.5秒", "UNSUPPORTED_DURATION"),
    ("时长十五秒", "UNSUPPORTED_DURATION"),
    ("1分钟", "UNSUPPORTED_DURATION"),
    ("60 FPS", "UNSUPPORTED_FPS"),
    ("每秒30帧", "UNSUPPORTED_FPS"),
    ("1280×720", "UNSUPPORTED_RESOLUTION"),
    ("不要抓取", "TASK_CONFLICT"),
    ("不要放下", "TASK_CONFLICT"),
    ("只需要抬升", "TASK_CONFLICT"),
    ("不要固定相机", "TASK_CONFLICT"),
])
def test_conflicting_constraints_are_not_silently_discarded(extra, code):
    result = parse_spec("Franka 把红色方块抓起并放到桌面右侧。" + extra)
    assert result["status"] == "unsupported"
    assert result["parsed_spec"] is None
    assert code in {x["code"] for x in result["issues"]}


@pytest.mark.parametrize("text", [
    "抓起红色方块，然后放到右侧。目标是蓝色方块。",
    "把黄色方块搬运到桌面右侧",
    "不要红色方块，把方块搬到右侧",
])
def test_ambiguous_or_unavailable_object_color(text):
    assert parse_spec(text)["status"] == "unsupported"


def test_preservation_instruction_and_appearance_separation():
    assert parse_spec("搬运方块到右侧，不要移动相机")["status"] == "ready"
    assert appearance_issues("真实木纹，保持机械臂的轨迹，不要改变相机") == []
    result = parse_spec("搬运方块到右侧", "橡木桌面，相机绕拍一圈")
    assert result["status"] == "unsupported"
    assert result["issues"][0]["code"] == "APPEARANCE_CAMERA"
    assert appearance_issues("添加一个物体，改变动作")


def paid_render():
    return {"appearance_description": "橡木桌面，柔和灯光", "quote_id": "quote_1", "max_cost_minor_units": 500}


def test_modes_enforce_disjoint_references_and_billing():
    assert JobBody(mode="scene_only", spec_id="spec_1", spec_revision=1).render is None
    assert JobBody(mode="scene_and_render", spec_id="spec_1", spec_revision=1, render=paid_render()).render.variants == 1
    assert JobBody(mode="uploaded_video_render", source_asset_id="asset_1", render=paid_render()).spec_id is None
    invalid = [
        {"mode": "scene_only", "spec_id": "spec_1"},
        {"mode": "scene_only", "spec_id": "spec_1", "spec_revision": 1, "render": paid_render()},
        {"mode": "scene_and_render", "spec_id": "spec_1", "spec_revision": 1},
        {"mode": "uploaded_video_render", "source_asset_id": "asset_1", "spec_id": "spec_1", "render": paid_render()},
        {"mode": "uploaded_video_render", "source_asset_id": "asset_1", "spec_revision": 1, "render": paid_render()},
        {"mode": "scene_only", "spec_id": "spec_1", "spec_revision": 1, "source_asset_id": "asset_1"},
        {"mode": "scene_only", "spec_id": "spec_1", "spec_revision": 1, "python_code": "print(1)"},
    ]
    for payload in invalid:
        with pytest.raises(ValidationError):
            JobBody.model_validate(payload)


@pytest.mark.parametrize("change", [
    {"automatic_paid_retries": True}, {"automatic_paid_retries": 0},
    {"variants": 2}, {"variants": True}, {"max_cost_minor_units": "500"},
    {"max_cost_minor_units": True}, {"max_cost_minor_units": -1},
    {"currency": "USD"}, {"resolution": "2K"}, {"duration_seconds": 16},
    {"appearance_description": " "}, {"model": "other"}, {"quote_id": "../secret"},
])
def test_paid_settings_cannot_silently_coerce_or_expand(change):
    with pytest.raises(ValidationError):
        RenderOptions.model_validate({**paid_render(), **change})


def test_reference_validation_is_shared_by_quotes_and_preflight():
    for contract in (PreflightBody, QuoteBody):
        assert contract(source_asset_id="asset_1").source_asset_id == "asset_1"
        for value in ({}, {"spec_id": "s"}, {"spec_id": "s", "spec_revision": 1, "source_asset_id": "a"}, {"source_asset_id": "a", "spec_revision": 1}):
            with pytest.raises(ValidationError):
                contract.model_validate(value)


@pytest.mark.parametrize("change", [
    {"seed": 8}, {"destination": "left"}, {"fps": 30},
    {"schema_version": True}, {"duration_seconds": 5.0}, {"width": "864"},
    {"camera": "wrist"}, {"object_color": "purple"}, {"object_mass": 5},
])
def test_task_spec_only_exposes_validated_template_parameters(change):
    with pytest.raises(ValidationError):
        TaskSpec.model_validate(change)


def test_asset_and_export_require_managed_ids_not_paths():
    with pytest.raises(ValidationError):
        AssetBody(upload_id="C:/private/video.mp4")
    with pytest.raises(ValidationError):
        ExportBody(asset_id="../asset")
    assert ExportBody(asset_id="asset_1").kind == "available"
