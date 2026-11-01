"""Bounded media inspection, deliberately separate from robotics fidelity certification."""
from __future__ import annotations

from hashlib import sha256
from pathlib import Path
import math
import statistics

from ..domain import EngineError, require

MAX_SOURCE_BYTES = 45_000_000
MAX_SOURCE_PIXELS = 1920 * 1080


def inspect_video(path: str | Path, *, max_bytes: int = 512_000_000,
                  max_frames: int = 1800, max_pixels: int = 3840 * 2160) -> dict:
    """Decode every frame within bounded limits; keep PTS, not guessed action labels.

    Duration is the decoded PTS span plus the last frame's duration. Container
    duration is reported separately, since provider files can have extra frames.
    No geometry/contact or real-world camera calibration is inferred here.
    """
    import av
    import numpy as np

    path = Path(path)
    require(path.is_file(), "MEDIA_MISSING", "Media file is unavailable", status=404)
    size = path.stat().st_size
    require(0 < size <= max_bytes, "MEDIA_SIZE", "Video exceeds the size limit", status=413)
    times, durations, black_frames = [], [], 0
    repeated, previous, dimensions = 0, None, None
    try:
        with av.open(str(path)) as container:
            videos = list(container.streams.video)
            require(len(videos) == 1, "MEDIA_STREAMS", "Exactly one video stream is required")
            stream = videos[0]
            codec = stream.codec_context.name
            width, height = stream.codec_context.width, stream.codec_context.height
            require(width > 0 and height > 0 and width * height <= max_pixels,
                    "MEDIA_DIMENSIONS", "Video dimensions exceed this service's limit")
            container_format = container.format.name
            fps = float(stream.average_rate) if stream.average_rate else None
            container_duration = float(container.duration / av.time_base) if container.duration is not None else None
            for frame in container.decode(stream):
                require(len(times) < max_frames, "MEDIA_FRAMES", "Video has too many frames")
                require(frame.pts is not None and frame.time_base is not None,
                        "MEDIA_PTS", "Every decoded frame must have a timestamp")
                t = float(frame.pts * frame.time_base)
                require(math.isfinite(t) and (not times or t > times[-1]),
                        "MEDIA_PTS", "Timestamps must be finite and strictly increasing")
                shape = (frame.width, frame.height)
                require(shape == (width, height) and shape[0] * shape[1] <= max_pixels,
                        "MEDIA_DIMENSIONS", "Changing or oversized frame dimensions are unsupported")
                dimensions = shape
                frame_duration = getattr(frame, "duration", None)
                durations.append(float(frame_duration * frame.time_base) if frame_duration else None)
                times.append(t)
                # Thumbnail statistics are diagnostics, never a physical quality score.
                thumb = frame.reformat(width=64, height=36, format="rgb24").to_ndarray()
                black_frames += int(float(np.max(thumb)) < 8)
                fingerprint = sha256(thumb.tobytes()).digest()
                repeated += int(previous == fingerprint)
                previous = fingerprint
    except EngineError:
        raise
    except Exception:
        raise EngineError("MEDIA_DECODE", "Video could not be fully decoded") from None
    require(len(times) >= 2 and dimensions is not None, "MEDIA_EMPTY", "At least two frames are required")
    intervals = [b - a for a, b in zip(times, times[1:])]
    median_interval = statistics.median(intervals)
    last_duration = durations[-1] or median_interval
    require(math.isfinite(last_duration) and last_duration > 0,
            "MEDIA_PTS", "Invalid final-frame duration")
    duration = times[-1] - times[0] + last_duration
    with path.open("rb") as handle:
        digest = sha256()
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return {
        "width": dimensions[0], "height": dimensions[1], "frame_count": len(times),
        "fps": fps, "measured_fps": 1.0 / median_interval,
        "duration_seconds": duration, "container_duration_seconds": container_duration,
        "pts_seconds": times, "time_origin_seconds": times[0],
        "constant_frame_rate": all(abs(x - median_interval) < 0.0001 for x in intervals),
        "codec": codec, "container_format": container_format, "bytes": size,
        "sha256": digest.hexdigest(), "black_frame_count": black_frames,
        "identical_thumbnail_pairs": repeated, "decoded_all_frames": True,
    }


def validate_source_video(path: str | Path, *, allow_low_fps=False) -> dict:
    """Validate 2–15 second media; low FPS is allowed only before conversion."""
    require(Path(path).suffix.lower() == ".mp4", "SOURCE_FORMAT", "Upload an MP4 video")
    media = inspect_video(path, max_bytes=MAX_SOURCE_BYTES, max_frames=901,
                          max_pixels=MAX_SOURCE_PIXELS)
    require("mp4" in media["container_format"].split(","), "SOURCE_FORMAT", "Expected an MP4 container")
    require(media["codec"] in {"h264", "hevc"}, "SOURCE_CODEC", "H.264 or HEVC video is required")
    require(256 <= media["width"] <= 1920 and 256 <= media["height"] <= 1920,
            "SOURCE_DIMENSIONS", "Supported dimensions are 256–1920 pixels, at most 2,073,600 pixels")
    require(0.4 <= media["width"] / media["height"] <= 2.5,
            "SOURCE_ASPECT", "Supported aspect ratio is 0.4–2.5")
    minimum_fps = 1 if allow_low_fps else 23.97
    require(media["fps"] is not None and minimum_fps <= media["fps"] <= 60.01,
            "SOURCE_FPS", "Upload supports 1–60 FPS; H3 reference requires 23.976–60 FPS")
    require(2-1e-6 <= media["duration_seconds"] <= 15+1e-6,
            "SOURCE_DURATION", "Reference video must be between 2 and 15 seconds")
    require(media["constant_frame_rate"], "SOURCE_VFR", "Variable frame rate is unsupported in this release")
    return media


def normalize_reference(source, destination, media):
    """Previous-frame hold, preserving wall-clock speed; never trim a long input."""
    import av
    from fractions import Fraction
    mapping = []
    with av.open(str(source)) as reader, av.open(str(destination), "w") as out:
        stream = out.add_stream("libx264", rate=24)
        stream.width, stream.height, stream.pix_fmt = media["width"], media["height"], "yuv420p"
        stream.options = {"crf": "18", "preset": "veryfast"}
        frames = iter(reader.decode(video=0))
        current, following = next(frames), next(frames, None)
        original_index = 0
        for i in range(math.ceil(media["duration_seconds"]*24-1e-6)):
            t = i/24
            while following is not None and float(following.pts*following.time_base)-media["time_origin_seconds"] <= t+1e-7:
                current, following = following, next(frames, None)
                original_index += 1
            frame = av.VideoFrame.from_ndarray(current.to_ndarray(format="rgb24"), format="rgb24")
            frame.pts, frame.time_base = i, Fraction(1,24)
            for packet in stream.encode(frame): out.mux(packet)
            mapping.append({"output_frame":i,"output_time_s":t,"original_frame":original_index,
                            "original_time_s":float(current.pts*current.time_base)-media["time_origin_seconds"]})
        for packet in stream.encode(): out.mux(packet)
    prepared = validate_source_video(destination)
    return prepared, {"method":"previous_frame_hold", "speed_changed":False,"trimmed":False,
                      "original_media":media,"prepared_media":prepared,"mapping":mapping,
                      "duration_quantization_seconds":prepared["duration_seconds"]-media["duration_seconds"]}


def evaluate_render(source_path: str | Path, output_path: str | Path) -> dict:
    """Basic checks do not turn a plausible rendering into labelled training data."""
    source, output = inspect_video(source_path), inspect_video(output_path)
    st = [x - source["time_origin_seconds"] for x in source["pts_seconds"]]
    ot = [x - output["time_origin_seconds"] for x in output["pts_seconds"]]
    matched = len(st) == len(ot) and all(abs(a - b) < 0.001 for a, b in zip(st, ot))
    duration_match = abs(source["duration_seconds"] - output["duration_seconds"]) < 0.001
    timing = "pass" if matched and duration_match else "fail"
    technical = "fail" if output["black_frame_count"] == output["frame_count"] else "pass"
    return {
        "schema": "render_quality_v1", "source_media": source, "output_media": output,
        "technical": {"status": technical, "basis": "decode, timestamps and all-black thumbnail check only"},
        "timing": {"status": timing, "basis": "decoded timestamp grid and frame count only; action synchronization unverified",
                   "unmatched_output_frames": max(0, output["frame_count"] - source["frame_count"]),
                   "duration_delta_seconds": output["duration_seconds"] - source["duration_seconds"]},
        "appearance": {"status": "unknown", "reason": "Requires human visual review"},
        "geometry": {"status": "unknown", "reason": "No calibrated geometric observation available"},
        "contact": {"status": "unknown", "reason": "No rendered physical contact ground truth"},
        "action_alignment": {"status": "unknown", "reason": "Matching PTS does not establish matching actions"},
        "training_eligible": False, "usage_scope": "evaluation_only",
        "source_gt_attached_to_render": False,
        "decision": "reject_direct_action_training" if timing == "fail" or technical == "fail" else "unknown",
    }


def comparison_sheet(source_path, output_path, destination):
    """Display matched wall-clock samples; do not warp the generated timeline."""
    import av
    from PIL import Image, ImageDraw
    media = inspect_video(source_path)
    final_time = media["pts_seconds"][-1] - media["time_origin_seconds"]
    requested = [final_time*fraction for fraction in (0., .25, .5, .75, 1.)]
    def samples(path):
        images, origin = [], None
        with av.open(str(path)) as clip:
            for frame in clip.decode(video=0):
                t = float(frame.pts * frame.time_base)
                if origin is None:origin = t
                while len(images) < len(requested) and t-origin >= requested[len(images)]-1e-5:
                    image = frame.to_image()
                    image.thumbnail((432,260))
                    images.append((image,t-origin))
        return images
    left, right = samples(source_path), samples(output_path)
    sheet = Image.new("RGB", (884, 290*len(requested)+40), "white")
    draw = ImageDraw.Draw(sheet)
    draw.text((12,10),"Source white model",fill="black")
    draw.text((452,10),"H3 output / visual inspection only",fill="black")
    for i in range(len(requested)):
        for x, frames in ((10,left),(452,right)):
            if i < len(frames):
                image, at = frames[i]
                sheet.paste(image,(x,40+i*290))
                draw.text((x,304+i*290),f"PTS {at:.4f}s",fill="black")
            else:
                draw.text((x,40+i*290),"No output frame at this time",fill="red")
    sheet.save(destination,quality=87)
