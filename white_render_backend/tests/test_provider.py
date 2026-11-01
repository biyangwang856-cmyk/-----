"""Paid-submission and fidelity boundaries; all HTTP is a local test double."""
import base64
from fractions import Fraction
import json
from pathlib import Path

import av
import httpx
import numpy as np
import pytest

from white_render_backend.domain import EngineError
from white_render_backend.synthetic.providers.minimax import (
    MiniMaxProvider, SubmissionUnknown, build_request, request_summary,
)
from white_render_backend.synthetic.quality import evaluate_render, inspect_video, validate_source_video


def encode(path, frames=120, *, fps=24, black=False):
    with av.open(str(path), "w") as container:
        stream = container.add_stream("libx264", rate=fps)
        stream.width, stream.height, stream.pix_fmt = 256, 256, "yuv420p"
        for i in range(frames):
            rgb = np.full((256, 256, 3), 0 if black else 100 + i % 100, dtype=np.uint8)
            frame = av.VideoFrame.from_ndarray(rgb, format="rgb24")
            frame.pts, frame.time_base = i, Fraction(1, fps)
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    return path


@pytest.fixture(scope="module")
def source(tmp_path_factory):
    return encode(tmp_path_factory.mktemp("h3_source") / "white.mp4")


def test_request_contract_and_safe_summary(source):
    payload = build_request(source, "橡木桌面、柔和侧光")
    assert payload["model"] == "MiniMax-H3" and payload["duration"] == 5
    assert payload["resolution"] == "768P" and payload["ratio"] == "adaptive"
    reference = payload["content"][1]
    assert reference["role"] == "reference_video"
    data = reference["video_url"]["url"].split(",", 1)[1]
    assert base64.b64decode(data) == source.read_bytes()
    frozen = request_summary(payload)
    assert frozen["source_bytes"] == source.stat().st_size
    assert "base64" not in json.dumps(frozen)
    assert not frozen["automatic_paid_retries"]


@pytest.mark.parametrize("status", [500, 502, 408, 409, 302])
def test_ambiguous_paid_status_never_retries(source, status):
    calls = []
    def handler(req):
        calls.append(req)
        return httpx.Response(status, headers={"location": "https://evil.example"})
    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        provider = MiniMaxProvider("test-secret", client=client)
        with pytest.raises(SubmissionUnknown):
            provider.submit(source, "wood tabletop")
    assert len(calls) == 1


def test_timeout_is_unknown_without_leaking_transport_details(source):
    calls = []
    def handler(req):
        calls.append(req)
        raise httpx.ReadTimeout("transport includes test-secret and base64", request=req)
    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(SubmissionUnknown) as exc:
            MiniMaxProvider("test-secret", client=client).submit(source, "wood tabletop")
    assert "test-secret" not in str(exc.value) and "base64" not in str(exc.value)
    assert len(calls) == 1


@pytest.mark.parametrize("body", [{}, {"task_id": None}, {"task_id": True}, {"task_id": "../x"}, []])
def test_missing_or_invalid_task_id_is_unknown(source, body):
    with httpx.Client(transport=httpx.MockTransport(lambda _: httpx.Response(200, json=body))) as client:
        with pytest.raises(SubmissionUnknown):
            MiniMaxProvider("test-secret", client=client).submit(source, "wood tabletop")


def test_explicit_rejection_redacts_provider_body(source):
    def handler(req):
        return httpx.Response(401, json={"error": {"message": req.headers["authorization"]}})
    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(EngineError) as exc:
            MiniMaxProvider("test-secret", client=client).submit(source, "wood tabletop")
    assert exc.value.code == "H3_REQUEST_REJECTED"
    assert "test-secret" not in str(exc.value)


def test_submit_and_readonly_preflight(source):
    calls = []
    def handler(req):
        calls.append(req)
        assert req.headers["authorization"] == "Bearer test-secret"
        if req.method == "POST":
            return httpx.Response(200, json={"task_id": "123"})
        return httpx.Response(200, json={"task": {"id": "123", "status": "running"}})
    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        provider = MiniMaxProvider("test-secret", client=client)
        assert provider.submit(source, "wood tabletop")["task_id"] == "123"
        calls.clear()
        preflight = provider.preflight(source, "wood tabletop", query_task_id="123")
        assert preflight["paid_requests"] == 0
        assert preflight["connectivity"] == "authenticated_read_only_query_passed"
        assert [req.method for req in calls] == ["GET"]


def test_query_normalizes_usage_without_asserting_actual_debit():
    body = {"task": {"id": "123", "status": "succeeded",
                     "content": {"url": "https://cdn.hailuoai.com/result.mp4"},
                     "usage": {"input_seconds": 5, "output_seconds": 5, "secret": "do not copy"}}}
    with httpx.Client(transport=httpx.MockTransport(lambda _: httpx.Response(200, json=body))) as client:
        value = MiniMaxProvider("test-secret", client=client).query("123")
    assert value["status"] == "succeeded" and value["actual_cost_minor_units"] is None
    assert value["usage"] == {"input_seconds": 5, "output_seconds": 5}
    assert "secret" not in json.dumps(value)


def test_download_does_not_leak_auth_and_cannot_overwrite(tmp_path):
    calls = []
    def handler(req):
        calls.append(req)
        assert "authorization" not in req.headers and "cookie" not in req.headers
        return httpx.Response(200, content=b"test video bytes")
    with httpx.Client(transport=httpx.MockTransport(handler),
                      headers={"Authorization": "Bearer client-default-secret"},
                      auth=("name", "password"), cookies={"session": "sensitive"}) as client:
        provider = MiniMaxProvider("api-secret", download_client=client)
        destination = tmp_path / "result.mp4"
        assert provider.download("https://cdn.hailuoai.com/result.mp4", destination)["bytes"] == 16
        assert destination.read_bytes() == b"test video bytes"
        with pytest.raises(EngineError) as exc:
            provider.download("https://cdn.hailuoai.com/result.mp4", destination)
        assert exc.value.code == "OUTPUT_EXISTS"
        assert len(calls) == 1


@pytest.mark.parametrize("url", ["http://cdn.hailuoai.com/a", "https://127.0.0.1/a",
    "https://cdn.hailuoai.com.evil.example/a", "https://user:pass@cdn.hailuoai.com/a",
    "https://cdn.hailuoai.com:8443/a", "file:///secret"])
def test_download_rejects_untrusted_urls(tmp_path, url):
    calls = []
    with httpx.Client(transport=httpx.MockTransport(lambda req: calls.append(req))) as client:
        with pytest.raises(EngineError) as exc:
            MiniMaxProvider("key", download_client=client).download(url, tmp_path / "a.mp4")
    assert exc.value.code == "H3_DOWNLOAD_HOST" and not calls


def test_download_redirect_to_private_host_rejected(tmp_path):
    calls = []
    def handler(req):
        calls.append(req)
        return httpx.Response(302, headers={"location": "https://127.0.0.1/internal"})
    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(EngineError) as exc:
            MiniMaxProvider("key", download_client=client).download("https://cdn.hailuoai.com/a", tmp_path / "a.mp4")
    assert exc.value.code == "H3_DOWNLOAD_HOST" and len(calls) == 1
    assert not list(tmp_path.glob("*.partial"))


def test_source_timestamp_contract(source):
    media = validate_source_video(source)
    assert media["frame_count"] == 120 and media["duration_seconds"] == pytest.approx(5)
    assert media["pts_seconds"][0] == 0 and media["pts_seconds"][-1] == pytest.approx(119 / 24)


def test_extra_output_frames_fail_timing_and_do_not_invent_gt(source, tmp_path):
    output = encode(tmp_path / "extra.mp4", frames=124)
    quality = evaluate_render(source, output)
    assert quality["timing"]["status"] == "fail"
    assert quality["timing"]["unmatched_output_frames"] == 4
    assert quality["geometry"]["status"] == quality["contact"]["status"] == "unknown"
    assert quality["decision"] == "reject_direct_action_training"
    assert quality["training_eligible"] is False


def test_equal_pts_is_not_robot_fidelity(source):
    quality = evaluate_render(source, source)
    assert quality["timing"]["status"] == "pass"
    assert quality["action_alignment"]["status"] == "unknown"
    assert quality["training_eligible"] is False


def test_all_black_output_fails_technical(source, tmp_path):
    output = encode(tmp_path / "black.mp4", black=True)
    assert evaluate_render(source, output)["technical"]["status"] == "fail"


def test_corrupt_and_overlong_media_rejected(tmp_path):
    invalid = tmp_path / "bad.mp4"
    invalid.write_bytes(b"not a video")
    with pytest.raises(EngineError, match="MEDIA_DECODE"):
        inspect_video(invalid)
    long_video = encode(tmp_path / "long.mp4", frames=361)
    with pytest.raises(EngineError, match="SOURCE_DURATION"):
        validate_source_video(long_video)


def test_recorded_official_output_host_is_allowed_but_other_buckets_are_not():
    provider = MiniMaxProvider()
    provider._validate_download_url("https://algeng-video-infer.oss-cn-shanghai.aliyuncs.com/test.mp4")
    with pytest.raises(EngineError, match="H3_DOWNLOAD_HOST"):
        provider._validate_download_url("https://another-bucket.oss-cn-shanghai.aliyuncs.com/test.mp4")


def test_missing_credentials_do_not_break_offline_preflight(source, monkeypatch):
    monkeypatch.delenv("MINIMAX_API_KEY", raising=False)
    monkeypatch.setattr("white_render_backend.synthetic.providers.minimax.load_api_key", lambda: "")
    provider = MiniMaxProvider()
    result = provider.preflight(source, "wood tabletop")
    assert result["configured"] is False and result["request_valid"] is True
    assert result["paid_requests"] == 0 and result["connectivity"] == "not_checked"
