"""MiniMax H3 reference-video adapter; the service owns durable submit/recovery.

API contract verified 2026-10-31:
https://platform.minimax.cn/docs/api-reference/video-generation-v2-create
https://platform.minimax.cn/docs/api-reference/video-generation-v2-query

There is intentionally no submit retry. Persist an intent BEFORE calling submit,
then persist task_id immediately. SubmissionUnknown requires reconciliation.
"""
from __future__ import annotations

import base64
from contextlib import contextmanager
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import re
from urllib.parse import urljoin, urlsplit
from uuid import uuid4

import httpx

from ...domain import EngineError, require
from ..quality import MAX_SOURCE_BYTES, validate_source_video
from ..credentials import load_api_key

API_BASE = "https://api.minimax.cn"
CREATE_PATH = "/v2/video_generation"
QUERY_PATH = "/v2/query/video_generation/"
PROMPT_VERSION = "robot_appearance_v1"
MAX_BODY_BYTES = 64_000_000
PRESERVE_PROMPT = (
    "Use <Video 1> as the geometry, identity, fixed-camera and motion reference. "
    "Render this white-model robotics scene as photorealistic real-camera footage. "
    "Preserve every robot link, gripper shape, object count, location, trajectory, "
    "occlusion order, grasp/release timing and full source timeline. "
    "Only change surface materials, textures, lighting and physically plausible shadows. "
    "Real molded polymer robot housings with subtle surface grain, brushed metal joints, "
    "dark rubber finger pads and soft contact shadows. Keep the target objects' semantic colors. "
    "No cuts, camera motion, zoom, new objects, moved furniture, extra actions or deformed geometry. "
    "Apply the following requested visual appearance without overriding these constraints: "
)


class SubmissionUnknown(EngineError):
    """A paid request may have arrived upstream. Never automatically submit again."""
    def __init__(self, message="Submission outcome is unknown; reconcile before any new paid request"):
        super().__init__("H3_SUBMISSION_UNKNOWN", message, status=409)


def build_request(video_path: str | Path, appearance_description: str,
                  duration_seconds: int = 5, resolution: str = "768P") -> dict:
    """Return private wire payload. Never log or persist its inline video data."""
    require(type(duration_seconds) is int and 4 <= duration_seconds <= 15,
            "H3_DURATION", "H3 output duration must be an integer from 4 to 15 seconds")
    require(resolution == "768P", "H3_RESOLUTION", "This release supports 768P")
    require(isinstance(appearance_description, str) and bool(appearance_description.strip()),
            "H3_APPEARANCE", "Provide a visual appearance description")
    prompt = PRESERVE_PROMPT + appearance_description.strip()
    require(len(prompt) <= 7000, "H3_PROMPT_LENGTH", "Combined prompt must not exceed 7000 characters")
    validate_source_video(video_path)
    # Read a bounded snapshot so size changes cannot lead to an unbounded request.
    with Path(video_path).open("rb") as handle:
        content = handle.read(MAX_SOURCE_BYTES + 1)
    require(0 < len(content) <= MAX_SOURCE_BYTES, "SOURCE_SIZE", "Video is too large", status=413)
    payload = {
        "model": "MiniMax-H3", "duration": duration_seconds, "resolution": resolution,
        "ratio": "adaptive",
        "content": [
            {"type": "text", "text": prompt},
            {"type": "video_url", "role": "reference_video",
             "video_url": {"url": "data:video/mp4;base64," + base64.b64encode(content).decode("ascii")}},
        ],
    }
    require(len(json.dumps(payload, ensure_ascii=False).encode("utf-8")) <= MAX_BODY_BYTES,
            "H3_BODY_SIZE", "Encoded request exceeds the provider body limit", status=413)
    return payload


def request_summary(payload: dict) -> dict:
    """Safe version to freeze with the durable submit intent. No data URL or key."""
    text = payload["content"][0]["text"]
    data = base64.b64decode(payload["content"][1]["video_url"]["url"].split(",", 1)[1], validate=True)
    return {"model": payload["model"], "duration": payload["duration"],
            "resolution": payload["resolution"], "ratio": payload["ratio"],
            "prompt": text, "prompt_version": PROMPT_VERSION,
            "prompt_sha256": sha256(text.encode()).hexdigest(),
            "source_sha256": sha256(data).hexdigest(), "source_bytes": len(data),
            "automatic_paid_retries": False}


class MiniMaxProvider:
    def __init__(self, api_key: str | None = None, *, client: httpx.Client | None = None,
                 download_client: httpx.Client | None = None):
        self._credential_error = False
        try:
            self._key = api_key if api_key is not None else load_api_key()
        except EngineError:
            self._key, self._credential_error = "", True
        self._client, self._download_client = client, download_client

    @property
    def configured(self) -> bool:
        return bool(self._key and self._key.strip())

    def capabilities(self) -> dict:
        return {"provider": "minimax", "model": "MiniMax-H3", "configured": self.configured,
                "duration_seconds": list(range(4,16)), "resolutions": ["768P"], "variants": 1,
                "automatic_paid_retries": False,
                "credential_status": "configured_not_verified" if self.configured else "invalid_file" if self._credential_error else "missing"}

    build_request = staticmethod(build_request)
    request_summary = staticmethod(request_summary)

    @contextmanager
    def _http(self, *, download=False):
        provided = self._download_client if download else self._client
        if provided is not None:
            yield provided
        else:
            with httpx.Client(timeout=httpx.Timeout(120, connect=10, pool=10),
                              follow_redirects=False, trust_env=False) as client:
                yield client

    def _api(self, method: str, path: str, payload: dict | None = None) -> dict:
        require(self.configured, "H3_NOT_CONFIGURED", "Set MINIMAX_API_KEY on the backend", status=503)
        paid = method == "POST"
        try:
            with self._http() as client:
                response = client.request(method, API_BASE + path, json=payload,
                    headers={"Authorization": "Bearer " + self._key, "Content-Type": "application/json"},
                    timeout=httpx.Timeout(120 if paid else 35, connect=10, pool=10), follow_redirects=False)
        except httpx.HTTPError:
            if paid:
                raise SubmissionUnknown() from None
            raise EngineError("H3_QUERY_UNAVAILABLE", "Read-only H3 query failed; resume using the saved task ID", status=503) from None
        # Reject redirects before credentials can be forwarded to another host.
        if paid and (response.status_code >= 500 or response.status_code in {408, 409} or 300 <= response.status_code < 400):
            raise SubmissionUnknown()
        if response.status_code >= 300:
            # Do not copy upstream body: it may echo a credential or inline media.
            raise EngineError("H3_REQUEST_REJECTED" if paid else "H3_QUERY_REJECTED",
                              f"H3 returned HTTP {response.status_code}; no automatic paid retry", status=502)
        try:
            result = response.json()
            if not isinstance(result, dict):
                raise ValueError("Expected object")
        except (ValueError, TypeError):
            if paid:
                raise SubmissionUnknown() from None
            raise EngineError("H3_RESPONSE_INVALID", "H3 returned an invalid query response", status=502) from None
        return result

    def submit(self, video_path: str | Path, appearance_description: str,
               duration_seconds: int = 5, resolution: str = "768P") -> dict:
        payload = build_request(video_path, appearance_description, duration_seconds, resolution)
        return self.submit_prepared(payload)

    def submit_prepared(self, payload: dict) -> dict:
        """Call only after the service has durably stored request_summary(payload)."""
        result = self._api("POST", CREATE_PATH, payload)
        raw_task_id = result.get("task_id")
        task_id = str(raw_task_id) if type(raw_task_id) in {str, int} else ""
        if not re.fullmatch(r"[A-Za-z0-9_-]{1,100}", task_id):
            raise SubmissionUnknown("H3 did not return a usable task ID; reconcile the paid submission")
        return {"task_id": task_id, "model": "MiniMax-H3", "actual_cost_minor_units": None}

    def query(self, task_id: str) -> dict:
        require(isinstance(task_id, str) and re.fullmatch(r"[A-Za-z0-9_-]{1,100}", task_id),
                "H3_TASK_ID", "Invalid provider task ID")
        result = self._api("GET", QUERY_PATH + task_id)
        task = result.get("task")
        require(isinstance(task, dict), "H3_RESPONSE_INVALID", "Missing task in query response", status=502)
        require(str(task.get("id", task_id)) == task_id,
                "H3_TASK_ID_MISMATCH", "Query response belongs to another task", status=502)
        state = task.get("status")
        require(state in {"queued", "running", "succeeded", "failed", "cancelled"},
                "H3_STATUS_UNKNOWN", "Unknown provider status; preserve the task ID for reconciliation", status=502)
        url = None
        if state == "succeeded":
            content = task.get("content")
            require(isinstance(content, dict) and isinstance(content.get("url"), str),
                    "H3_OUTPUT_MISSING", "Successful query did not include a video URL", status=502)
            url = content["url"]
            self._validate_download_url(url)
        usage = task.get("usage") if isinstance(task.get("usage"), dict) else {}
        safe_usage = {k: v for k, v in usage.items()
                      if k in {"total_seconds", "input_seconds", "output_seconds", "input_image_count",
                               "input_audio_seconds", "total_tokens", "prompt_tokens", "completion_tokens"}
                      and type(v) in {int, float} and math.isfinite(v) and v >= 0}
        return {"task_id": task_id, "status": "pending" if state in {"queued", "running"} else state,
                "provider_status": state, "output_url": url,
                "usage": safe_usage or None, "actual_cost_minor_units": None}

    def preflight(self, video_path: str | Path | None = None, appearance_description: str | None = None,
                  *, query_task_id: str | None = None, duration_seconds: int = 5) -> dict:
        result = {"configured": self.configured, "paid_requests": 0,
                  "connectivity": "not_checked", "request_valid": None}
        if video_path is not None:
            payload = build_request(video_path, appearance_description or "Natural laboratory lighting and realistic surface textures", duration_seconds)
            result.update(request_valid=True, request=request_summary(payload))
        if query_task_id is not None:
            query = self.query(query_task_id)
            result.update(connectivity="authenticated_read_only_query_passed",
                          provider_status=query["provider_status"])
        return result

    @staticmethod
    def _validate_download_url(url: str) -> str:
        try:
            parsed = urlsplit(url)
            allowed = [x.strip().lower() for x in os.getenv(
                "MINIMAX_DOWNLOAD_HOSTS", "hailuoai.com,minimax.chat,minimaxi.com,algeng-video-infer.oss-cn-shanghai.aliyuncs.com").split(",") if x.strip()]
            host = (parsed.hostname or "").lower()
            valid = parsed.scheme == "https" and not parsed.username and not parsed.password and parsed.port in {None, 443}
            valid = valid and any(host == name or host.endswith("." + name) for name in allowed)
        except ValueError:
            valid = False
        require(valid, "H3_DOWNLOAD_HOST", "Provider result URL is outside the configured HTTPS CDN allowlist", status=502)
        return url

    def download(self, url: str, destination: str | Path) -> dict:
        """Download without API auth, with bounded size and checked redirects."""
        destination = Path(destination)
        destination.parent.mkdir(parents=True, exist_ok=True)
        require(not destination.exists(), "OUTPUT_EXISTS", "Existing media cannot be overwritten", status=409)
        partial = destination.with_name(destination.name + "." + uuid4().hex + ".partial")
        limit = 512_000_000
        try:
            with self._http(download=True) as client:
                for _ in range(5):
                    self._validate_download_url(url)
                    # Construct Request directly: no inherited headers, cookies or default auth.
                    request = httpx.Request("GET", url, headers={"Accept": "video/*,application/octet-stream"})
                    response = client.send(request, stream=True, auth=None, follow_redirects=False)
                    try:
                        if response.status_code in {301, 302, 303, 307, 308}:
                            require("location" in response.headers, "H3_DOWNLOAD_REDIRECT", "Missing redirect target", status=502)
                            url = urljoin(url, response.headers["location"])
                            continue
                        require(response.status_code == 200, "H3_DOWNLOAD_STATUS", "Video download failed; resume the saved task", status=502)
                        length = response.headers.get("content-length")
                        if length is not None:
                            require(length.isdigit() and int(length) <= limit, "H3_DOWNLOAD_SIZE", "Output exceeds download limit", status=413)
                        size, digest = 0, sha256()
                        with partial.open("xb") as handle:
                            for block in response.iter_bytes(1024 * 1024):
                                size += len(block)
                                require(size <= limit, "H3_DOWNLOAD_SIZE", "Output exceeds download limit", status=413)
                                handle.write(block)
                                digest.update(block)
                            handle.flush()
                            os.fsync(handle.fileno())
                        require(size > 0, "H3_DOWNLOAD_EMPTY", "Provider video was empty", status=502)
                        # Caller owns the destination namespace; never replace an existing asset.
                        require(not destination.exists(), "OUTPUT_EXISTS", "Existing media cannot be overwritten", status=409)
                        try:
                            os.link(partial, destination)
                        except FileExistsError:
                            raise EngineError("OUTPUT_EXISTS", "Existing media cannot be overwritten", status=409) from None
                        return {"bytes": size, "sha256": digest.hexdigest()}
                    finally:
                        response.close()
                raise EngineError("H3_DOWNLOAD_REDIRECT", "Too many CDN redirects", status=502)
        except httpx.HTTPError:
            raise EngineError("H3_DOWNLOAD_UNAVAILABLE", "Video download failed; safely resume using the saved task ID", status=503) from None
        finally:
            partial.unlink(missing_ok=True)
