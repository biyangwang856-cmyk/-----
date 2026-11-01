"""Deliberately bounded Chinese task parser, never executable model output.

This parser selects a tested template. It does not invent robot geometry,
positions, controllers or successful trajectories from free-form text.
"""
from __future__ import annotations

import re

from .contracts import SpecBody, TaskSpec


PARSER_VERSION = "zh-template-v1"
SUPPORTED_EXAMPLE = "Franka 机械臂把红色方块抓起并放到桌面右侧，固定相机，5 秒"


def appearance_issues(description: str) -> list[dict]:
    """Reject obvious task/camera overrides in a nominally visual description."""
    text = description.lower()
    # These are preservation instructions, not a request to move the camera.
    text = re.sub(r"(?:不要|不能|不得|禁止|不允许|不)(?:改变|移动|调整|更换)(?:相机|镜头|视角|机器人|机械臂|动作|轨迹|物体|几何|数量)", "", text)
    patterns = [
        ("APPEARANCE_CAMERA", r"绕拍|环绕(?:拍摄|镜头|相机)|运镜|推拉镜头|切换(?:镜头|视角)|移动(?:相机|镜头)|相机移动|camera\s+(?:orbit|move)|orbiting\s+camera", "外观要求不能改变固定相机；请仅描述材质、颜色和光照。"),
        ("APPEARANCE_GEOMETRY", r"(?:增加|添加|移除|删除|换成|变成)(?:一个|两个|一只|两只|新的)?(?:物体|方块|机械臂|机器人|夹爪)|(?:改变|更改)(?:轨迹|动作|物体数量)|双臂|多物体", "外观要求不能改变源场景的物体、机器人或动作。"),
    ]
    return [{"code": code, "message": message} for code, pattern, message in patterns if re.search(pattern, text)]


def parse_spec(description: str, appearance_description: str = "") -> dict:
    body = SpecBody(description=description, appearance_description=appearance_description)
    text = body.description.lower()
    issues = appearance_issues(body.appearance_description)
    needs_input = []

    def problem(code, message):
        if not any(item["code"] == code for item in issues):
            issues.append({"code": code, "message": message})

    unsupported = [
        ("UNSUPPORTED_ROBOT", r"ur\s*[-_]?\s*[0-9]+|aloha|kinova|xarm|unitree|leap|so[-_]?100|so[-_]?101|双臂|机器狗|四足|人形|双足|移动机器人|机械手|灵巧手", "首版仅支持固定 Franka Panda 单臂，不支持替换本体。"),
        ("UNSUPPORTED_TASK", r"堆叠|叠放|插入|插孔|开门|关门|拧|倒水|擦|扫|行走|走路|奔跑|跳跃|抛|扔|投掷|接住|装配|碰撞|障碍物|干扰物|移动平台|自由生成|任意场景|执行代码", "首版仅支持桌面单方块抓起、搬运、放置，不支持该任务或场景。"),
        ("UNSUPPORTED_CONTAINER", r"放[入进]|(?:杯|碗|盒|容器|篮|箱)(?:子|里|内|中)?", "首版目标是桌面区域，不是容器放置或其它目标物。"),
        ("UNSUPPORTED_OBJECT", r"球|圆柱|瓶|水果|苹果|香蕉|工具|多个(?:目标|物体|方块)|(?:两|二|三|四|五|[2-9])\s*(?:个|块)\s*(?:物体|目标|方块)|同时抓", "首版仅支持一个固定尺寸的方块。"),
        ("UNSUPPORTED_DIRECTION", r"左(?:侧|边|方|面)?|向前|向后|前方|后方|中间|中央|任意位置|随机位置|随机相机", "首版目标位置固定为模板右侧（世界 +Y 方向），不能指定其它位置。"),
        ("UNSUPPORTED_CAMERA", r"绕拍|运镜|环绕|切换(?:镜头|视角)|俯视|第一人称|腕部相机|手眼相机|多相机|多视角|移动(?:相机|镜头)|相机移动|camera\s+(?:orbit|move)", "首版仅支持已校准的固定单相机视角。"),
        ("UNSUPPORTED_DIMENSION", r"\d\s*(?:厘米|毫米|米|cm\b|mm\b|meters?\b)|(?:坐标|角度|尺寸|大小|距离)(?:为|是|设为|改为|设置)?\s*[（(\d]|缩小|放大|缩放|(?:改变|调整)(?:尺寸|距离|重力)|无重力", "首版不开放自定义尺寸、位置、距离、重力等物理参数。"),
    ]
    # Remove only well-defined preservation clauses before conflict detection.
    normalized = re.sub(r"(?:不要|不能|不得|禁止|不允许)(?:移动|改变)(?:相机|镜头|视角)", "固定相机", text)
    for code, pattern, message in unsupported:
        if re.search(pattern, normalized):
            problem(code, message)

    if re.search(r"(?:不要|不能|禁止|不得|不许|不需要|无需|不)(?:抓|夹|拾|搬|放|释放|移动)|(?:仅|只)(?:需要|需|要)?(?:抓取|抓起|抬升|拿起)|不放下|不释放", normalized):
        problem("TASK_CONFLICT", "抓放模板必须完成抓取、抬升、搬运、放置和释放，不能省略或否定其中阶段。")
    if re.search(r"(?:不要|不能|禁止|不得|不用|不需要)(?:固定|franka|panda|白模|白膜)", normalized):
        problem("TASK_CONFLICT", "请求与固定 Franka 白模模板冲突。")

    durations = re.findall(r"(?<![\d.])(\d+(?:\.\d+)?)\s*(?:秒|seconds?\b|sec\b|s\b)", normalized)
    chinese_durations = re.findall(r"([零一二两三四五六七八九十百]+)\s*秒", normalized)
    if any(float(value) != 5 for value in durations) or any(value != "五" for value in chinese_durations) or re.search(r"分钟|minutes?\b", normalized):
        problem("UNSUPPORTED_DURATION", "首版视频时长固定为 5 秒。")
    fps_values = re.findall(r"(\d+(?:\.\d+)?)\s*(?:fps|帧/秒|帧每秒)", normalized)
    if any(float(value) != 24 for value in fps_values) or re.search(r"(?:每秒|每一秒)\s*(?!24\b)\d+\s*帧", normalized):
        problem("UNSUPPORTED_FPS", "首版白模固定 24 FPS。")
    resolutions = re.findall(r"(\d+)\s*[x×*]\s*(\d+)", normalized)
    if any((int(w), int(h)) != (864, 480) for w, h in resolutions) or re.search(r"\b(?:1080p|720p|4k|2k)\b", normalized):
        problem("UNSUPPORTED_RESOLUTION", "白模输出固定 864×480，H3 渲染输出另用 768P。")

    colors = set()
    for value, token in (("red", r"红(?:色)?(?:的)?(?:方块|物块|立方体)|red\s+cube"), ("blue", r"蓝(?:色)?(?:的)?(?:方块|物块|立方体)|blue\s+cube"), ("green", r"绿(?:色)?(?:的)?(?:方块|物块|立方体)|green\s+cube")):
        if re.search(token, normalized):
            colors.add(value)
    if re.search(r"(?:黄|黑|白|紫|橙|粉)(?:色)?(?:的)?(?:方块|物块|立方体)", normalized):
        problem("UNSUPPORTED_OBJECT_COLOR", "目标方块只支持红、蓝、绿三种颜色。")
    if len(colors) > 1:
        problem("CONFLICTING_COLORS", "请为唯一目标方块选择一种颜色。")
    if re.search(r"(?:不|不要)(?:用|使用|选择)?(?:红|蓝|绿)(?:色)?(?:的)?方块", normalized):
        problem("CONFLICTING_COLORS", "否定颜色没有给出可执行的唯一目标颜色，请明确选择。")

    object_present = bool(re.search(r"方块|物块|立方体|cube", normalized))
    complete_task = bool(re.search(r"抓放|抓取.{0,12}放置|(?:抓|夹|拾|拿).{0,30}(?:放到|放下|放置|放在)|搬(?:运|到)?|移到|移动到|pick[_ -]?place|pick\s+and\s+place", normalized))
    if not object_present:
        needs_input.append({"code": "OBJECT_REQUIRED", "message": "请明确目标是单个方块；不能从其它目标物推断方块。"})
    if not complete_task:
        needs_input.append({"code": "TASK_REQUIRED", "message": "请描述完整抓放任务，例如把方块抓起并放到桌面右侧。"})

    defaults = {
        "robot": "franka_panda", "destination": "right", "scene": "table", "camera": "fixed",
        "width": 864, "height": 480, "fps": 24, "duration_seconds": 5, "seed": 7,
    }
    if not colors:
        defaults["object_color"] = "red"
    result = {
        "status": "unsupported" if issues else "needs_input" if needs_input else "ready",
        "parser_version": PARSER_VERSION,
        "description": body.description,
        "appearance_description": body.appearance_description,
        "parsed_spec": None,
        "defaults": defaults,
        "issues": issues + needs_input,
        "supported_example": SUPPORTED_EXAMPLE,
        "scope": "受约束中文模板解析；不执行用户代码，不提供任意任务理解。",
    }
    if result["status"] == "ready":
        result["parsed_spec"] = TaskSpec(object_color=next(iter(colors), "red")).model_dump()
    return result
