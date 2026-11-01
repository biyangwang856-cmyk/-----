"""Application service for independently resumable source and rendered assets."""
from __future__ import annotations

from copy import deepcopy
import json
import math
import os
from pathlib import Path
import shutil
import threading
import time
from uuid import uuid4
import zipfile

from ..domain import EngineError, require
from ..util import contained, digest, files_manifest, read_json, safe_id, sha256, write_json
from .contracts import JobBody, TaskSpec
from .providers.minimax import MiniMaxProvider, SubmissionUnknown
from .quality import validate_source_video, normalize_reference, evaluate_render, comparison_sheet
from .specs import parse_spec, appearance_issues
from .store import Store


class SyntheticService:
    def __init__(self, platform, provider=None, simulator=None):
        self.platform = platform
        self.root = platform.root / "synthetic"
        self.root.mkdir(parents=True, exist_ok=True)
        self.store = Store(self.root / "tasks.sqlite")
        self.provider = provider or MiniMaxProvider()
        self.simulator = simulator

    def project_root(self, project):
        return self.root / "projects" / safe_id(project)

    def capabilities(self):
        from .simulation import capabilities
        sim = capabilities()
        return {"schema_version": 1, "simulation": sim,
                "templates": [TaskSpec().model_dump()] if sim["available"] else [],
                "object_colors": ["red", "blue", "green"],
                "parser": "constrained_template_rules", "provider": self.provider.capabilities(),
                "upload": {"format": "mp4", "duration_seconds_range": [2,15],
                           "max_bytes": 45_000_000, "fps_range": [1, 60], "gt_upload": False,
                           "low_fps_conversion": "previous-frame hold to 24 FPS; original retained"},
                "modes": ["scene_only", "scene_and_render", "uploaded_video_render"],
                "training_certification": False, "automatic_paid_retries": False}

    def register_upload(self, project, upload):
        identity = "upload_" + uuid4().hex
        self.store.put(project, "upload", identity, upload)
        return {**upload, "upload_id": identity}

    def create_spec(self, project, description, appearance_description=""):
        result = parse_spec(description, appearance_description)
        identity = "spec_" + uuid4().hex
        value = {**result, "spec_revision": 1}
        self.store.put(project, "spec", identity, value)
        return {"spec_id": identity, **value}

    def _asset(self, project, identity):
        return self.store.get(project, "asset", identity)["payload"]

    def asset(self, project, identity):
        value = self._asset(project, identity)
        return {"asset_id": identity, **{k: v for k, v in value.items() if k != "root"}}

    def asset_file(self, project, identity, filename):
        asset = self._asset(project, identity)
        entry = next((f for f in asset["files"] if f["path"] == filename), None)
        require(entry is not None, "NOT_FOUND", "File is not in the asset manifest", status=404)
        path = contained(asset["root"], filename, existing=True)
        require(path.stat().st_size == entry["size"], "ASSET_CHANGED", "Stored asset size changed", status=409)
        return path

    def video_path(self, project, identity):
        asset = self._asset(project, identity)
        require(asset.get("video_file"), "ASSET_NOT_VIDEO", "Select a source video")
        path = self.asset_file(project, identity, asset["video_file"])
        entry = next(f for f in asset["files"] if f["path"] == asset["video_file"])
        require(sha256(path) == entry["sha256"], "ASSET_CHANGED", "Source fingerprint differs", status=409)
        return path

    @staticmethod
    def _preview(video, output):
        import av
        with av.open(str(video)) as container:
            first = next(container.decode(video=0))
            first.to_image().save(output, quality=85)

    def _payload(self, root, kind, *, video_file=None, parents=None, channels=None, quality=None, metadata=None):
        return {"root": str(root), "kind": kind, "video_file": video_file,
                "files": files_manifest(root), "parents": parents or [], "channels": channels or {},
                "quality": quality or {}, "metadata": metadata or {},
                "allowed_usage": ["preview", "research"], "training_eligible": False}

    def register_asset(self, project, upload_id, description=""):
        upload = self.store.get(project, "upload", upload_id)["payload"]
        source = contained(self.platform.project_root(project) / "uploads", upload["path"], existing=True)
        require(source.suffix.lower() == ".mp4", "SOURCE_FORMAT", "Upload an MP4")
        require(sha256(source) == upload["sha256"], "UPLOAD_CHANGED", "Uploaded file changed", status=409)
        original_media = validate_source_video(source, allow_low_fps=True)
        identity = "asset_" + uuid4().hex
        root = self.project_root(project) / "assets" / identity
        (root / "media").mkdir(parents=True)
        original = root / "media" / "original.mp4"
        shutil.copyfile(source, original)
        require(sha256(original) == original_media["sha256"], "UPLOAD_CHANGED", "Uploaded file changed during registration", status=409)
        video = root / "media" / "white_model.mp4"
        if original_media["fps"] < 23.97:
            media, preprocessing = normalize_reference(original, video, original_media)
        else:
            shutil.copyfile(original, video)
            media = original_media
            preprocessing = {"method":"identity", "trimmed":False, "speed_changed":False,
                             "original_media":original_media, "prepared_media":media}
        write_json(root / "preprocessing.json", preprocessing)
        self._preview(video, root / "preview.jpg")
        channels = {"rgb": {"available": True, "provenance": "user_uploaded"},
                    **{key: {"available": False, "reason": "not_provided_in_mp4"}
                       for key in ("action", "state", "depth", "contact", "instance_seg", "semantic_seg")}}
        quality = {"technical": {"status": "pass"}, "source_task": {"status": "unknown"},
                   "geometry": {"status": "unknown"}, "training_eligible": False}
        write_json(root / "manifest.json", {"schema": "uploaded_media_v1", "description": description,
                   "metadata": media, "original_metadata": original_media, "preprocessing": "preprocessing.json",
                   "channels": channels, "provenance": "user_uploaded", "gt_available": False})
        write_json(root / "quality.json", quality)
        payload = self._payload(root, "uploaded_white_model", video_file="media/white_model.mp4",
                                channels=channels, quality=quality, metadata=media)
        self.store.put(project, "asset", identity, payload)
        return self.asset(project, identity)

    def _reference(self, project, body):
        if body.get("spec_id"):
            spec = self.store.get(project, "spec", body["spec_id"])["payload"]
            require(body.get("spec_revision") == spec["spec_revision"], "SPEC_REVISION", "Specification revision differs", status=409)
            require(spec["status"] == "ready", "SPEC_NOT_READY", "Resolve specification issues first", status=409)
            return {"spec_id": body["spec_id"], "spec_revision": spec["spec_revision"],
                    "spec_hash": digest(spec), "duration_seconds": 5}, spec
        identity = body.get("source_asset_id")
        asset = self._asset(project, identity)
        require(asset["kind"] in {"simulated_white_model", "uploaded_white_model"}, "SOURCE_KIND", "Use a white-model source asset")
        media = validate_source_video(self.video_path(project, identity))
        return {"source_asset_id": identity, "source_hash": media["sha256"], "duration_seconds": media["duration_seconds"]}, asset

    @staticmethod
    def _check_appearance(text):
        issues = appearance_issues(text)
        require(not issues, "APPEARANCE_CONFLICT", "; ".join(i["message"] for i in issues))

    def preflight(self, project, body):
        reference, _ = self._reference(project, body)
        appearance = body.get("appearance_description", "")
        self._check_appearance(appearance)
        path = self.video_path(project, body["source_asset_id"]) if body.get("source_asset_id") else None
        task_id = body.get("known_task_id")
        if task_id:
            # A shared provider credential must not turn an arbitrary task ID into
            # a cross-project metadata oracle. Reconciliation uses a dedicated role.
            known = [self.store.attempt(project, j["id"]) for j in self.store.jobs(project)]
            require(any(a and a["task_id"] == task_id for a in known), "TASK_SCOPE", "Task is not bound to this project", status=403)
        result = self.provider.preflight(path, appearance or None,
                    query_task_id=task_id if body.get("check_provider") else None,
                    duration_seconds=max(4,min(15,math.ceil(reference["duration_seconds"]-1e-6))))
        simulation_ready = self.capabilities()["simulation"]["available"] or self.simulator is not None
        return {"reference": reference, "provider": result, "paid_requests": 0,
                "simulation_available": simulation_ready,
                "ready_to_render": bool(self.provider.configured and appearance and (body.get("source_asset_id") or simulation_ready)),
                "note": "Credential presence is not authenticated connectivity; query requires a known project task"}

    def quote(self, project, body):
        reference, _ = self._reference(project, body)
        appearance = body.get("appearance_description", "").strip()
        require(appearance, "H3_APPEARANCE", "Provide visual appearance requirements")
        self._check_appearance(appearance)
        duration = body.get("duration_seconds") or max(4,min(15,math.ceil(reference["duration_seconds"]-1e-6)))
        require(type(duration) is int and 4 <= duration <= 15,"H3_DURATION","Output must be an integer from 4 to 15 seconds")
        amount = math.ceil(reference["duration_seconds"] * 50) + duration*50
        identity = "quote_" + uuid4().hex
        payload = {"reference": reference, "appearance_description": appearance,
                   "model": "MiniMax-H3", "duration_seconds": duration, "resolution": "768P", "variants": 1,
                   "duration_note": "Output uses whole seconds; input timeline is retained without trimming",
                   "estimated_cost_minor_units": amount, "currency": "CNY", "actual_cost_minor_units": None,
                   "price_rule": "h3_768p_20261031_input50_output50_fen_per_second",
                   "expires_at_epoch": time.time() + 1800, "automatic_paid_retries": False,
                   "billing_note": "Estimate only; upstream final debit is not returned by this service"}
        self.store.put(project, "quote", identity, payload)
        return {"quote_id": identity, **payload}

    def submit(self, project, body, key):
        body = JobBody.model_validate(body).model_dump(exclude_none=True)
        reference, _ = self._reference(project, body)
        render = body.get("render")
        quote_id = None
        if render:
            require(self.provider.configured, "H3_NOT_CONFIGURED", "Configure MINIMAX_API_KEY on the backend", status=503)
            self._check_appearance(render["appearance_description"])
            quote_id = render["quote_id"]
            quote = self.store.get(project, "quote", quote_id)["payload"]
            requested_duration = render.get("duration_seconds",quote["duration_seconds"])
            require(requested_duration == quote["duration_seconds"], "QUOTE_MISMATCH", "Output duration differs from quote", status=409)
            render["duration_seconds"] = requested_duration
            require(quote["reference"] == reference and quote["appearance_description"] == render["appearance_description"],
                    "QUOTE_MISMATCH", "Quote must match the source and appearance text", status=409)
            require(render["max_cost_minor_units"] >= quote["estimated_cost_minor_units"], "BUDGET_EXCEEDED", "Budget is below the quoted estimate", status=409)
        if body["mode"] != "uploaded_video_render":
            from .simulation import capabilities
            require(self.simulator is not None or capabilities()["available"], "SIMULATION_UNAVAILABLE", "Install simulation dependencies on the worker", status=503)
        return self.public_job(self.store.submit(project, body, key, quote_id=quote_id))

    def export(self, project, body, key):
        asset = self._asset(project, body["asset_id"])
        require(body.get("purpose", "research") != "action_supervised_training", "TRAINING_NOT_CERTIFIED",
                "No asset in this release is certified for action-supervised training", status=409)
        if body.get("kind") == "semantic_gt":
            require(asset["kind"] == "simulated_white_model", "GT_UNAVAILABLE", "Only simulated sources contain semantic/GT channels", status=409)
        return self.public_job(self.store.submit(project, {"mode": "export", **body}, key))

    def public_job(self, job):
        attempt = self.store.attempt(job["project"], job["id"])
        result = {"id": job["id"], "job_id": job["id"], "project": job["project"], "kind": "synthetic",
                  "mode": job["payload"]["mode"], "state": job["state"], "phase": job["phase"],
                  "result": job["result"], "error": job["error"], "cancel_requested": bool(job["cancel"]),
                  "created_at": job["created_at"], "updated_at": job["updated_at"],
                  "actual_cost_minor_units": None, "automatic_paid_retries": False}
        if attempt:
            result["provider"] = {"task_id": attempt["task_id"], "submission_state": attempt["phase"],
                                  "usage": attempt["payload"].get("usage"),
                                  "cancellation_note": "Local cancellation does not cancel or refund an upstream task"}
        return result

    def get_job(self, project, job_id):
        return self.public_job(self.store.job(project, job_id))

    def action(self, project, job_id, action):
        require(action in {"cancel", "retry"}, "NOT_FOUND", "Unknown job action", status=404)
        return self.public_job(getattr(self.store, action)(project, job_id))

    def reconcile(self, project, job_id, task_id, note):
        self.provider.query(task_id)  # read-only and authenticated; input match still operator-attested
        return self.public_job(self.store.reconcile(project, job_id, task_id, note))

    def _publish(self, job, stage, kind, key, **kwargs):
        identity = "asset_" + uuid4().hex
        dest = self.project_root(job["project"]) / "assets" / identity
        dest.parent.mkdir(parents=True, exist_ok=True)
        self.store.pulse(job, "publishing_" + kind)
        os.rename(stage, dest)
        payload = self._payload(dest, kind, **kwargs)
        self.store.publish(job, identity, payload, key)
        return identity

    def _scene(self, job, stage):
        from .simulation import generate_scene
        spec = self.store.get(job["project"], "spec", job["payload"]["spec_id"])["payload"]
        simulation = self.simulator or generate_scene
        result = simulation(stage, spec["parsed_spec"], lambda e: self.store.pulse(job, e["stage"]))
        video = stage / "media" / "white_model.mp4"
        media = validate_source_video(video)
        require(media["frame_count"] == 120, "SIMULATION_TIMING", "Expected 120 source frames")
        quality = {"source_task": result["validation"], "technical": {"status": "pass"},
                   "training_eligible": False, "ecp_compatibility": "unverified"}
        write_json(stage / "quality.json", quality)
        # Refresh checksums after app-layer quality is added.
        write_json(stage / "checksums.json", {"files": files_manifest(stage, exclude=("checksums.json",))})
        return self._publish(job, stage, "simulated_white_model", "source_asset_id", video_file="media/white_model.mp4",
                             channels=result["channels"], quality=quality, metadata=media)

    def _render(self, job, stage, source_id):
        project = job["project"]
        source = self._asset(project, source_id)
        if source["kind"] == "simulated_white_model":
            require(source["quality"]["source_task"]["status"] == "pass", "SOURCE_TASK_FAILED",
                    "Source asset is preserved; physical task failed and paid rendering was skipped", status=409)
        attempt = self.store.attempt(project, job["id"])
        if not attempt:
            render = job["payload"]["render"]
            prepared = self.provider.build_request(self.video_path(project, source_id), render["appearance_description"],
                                                   render.get("duration_seconds",5), render.get("resolution","768P"))
            audit = self.provider.request_summary(prepared)
            audit.update(source_asset_id=source_id, quote_id=render["quote_id"], actual_cost_minor_units=None)
            intent = self.store.begin_submission(job, audit)
            try:
                receipt = self.provider.submit_prepared(prepared)
            except SubmissionUnknown:
                self.store.receipt(project, job["id"], intent, "unknown")
                raise
            except EngineError:
                self.store.receipt(project, job["id"], intent, "rejected")
                raise
            # Persist remote receipt before further work, even if cancelled while POST was running.
            self.store.receipt(project, job["id"], intent, "submitted", receipt["task_id"])
            self.store.defer(job)
            return False
        if not attempt["task_id"]:
            raise SubmissionUnknown("A saved submit intent has no task ID. Reconcile it; do not generate again")
        self.store.pulse(job, "querying_provider")
        query = self.provider.query(attempt["task_id"])
        self.store.receipt(project, job["id"], attempt["intent_id"], query["status"],
                           details={"usage": query.get("usage"), "actual_cost_minor_units": None})
        if query["status"] == "pending":
            self.store.defer(job, seconds=10)
            return False
        require(query["status"] == "succeeded", "H3_TASK_FAILED", "Upstream generation failed; no automatic paid retry", status=502)
        self.store.pulse(job, "downloading_render")
        stage.mkdir(parents=True, exist_ok=True)
        output = stage / "rendered.mp4"
        self.provider.download(query["output_url"], output)
        quality = evaluate_render(self.video_path(project, source_id), output)
        comparison_sheet(self.video_path(project, source_id), output, stage / "comparison.jpg")
        self._preview(output, stage / "preview.jpg")
        write_json(stage / "quality.json", quality)
        write_json(stage / "request.json", attempt["payload"])
        write_json(stage / "manifest.json", {"schema": "render_asset_v1", "source_asset_id": source_id,
            "provider_task_id": attempt["task_id"], "usage": query.get("usage"), "actual_cost_minor_units": None,
            "source_gt_attached_to_render": False, "training_eligible": False})
        self._publish(job, stage, "h3_render", "render_asset_id", video_file="rendered.mp4", parents=[source_id],
                      channels={"rgb": {"available": True, "provenance": "model_generated"}},
                      quality=quality, metadata=quality["output_media"])
        return True

    def _export(self, job, stage):
        body = job["payload"]
        asset = self._asset(job["project"], body["asset_id"])
        stage.mkdir(parents=True, exist_ok=True)
        archive = stage / "bundle.zip"
        selected = asset["files"]
        if body.get("kind") == "media":
            selected = [f for f in selected if f["path"].endswith((".mp4", ".jpg")) or f["path"] == "quality.json"]
        export_manifest = {"schema": "synthetic_export_v1", "source_asset_id": body["asset_id"],
            "source_kind": asset["kind"], "export_kind": body.get("kind", "available"),
            "files": selected, "channels": asset["channels"] if body.get("kind") != "media" else
                {"rgb": asset["channels"].get("rgb", {"available": False}),
                 "gt": {"available": False, "reason": "media_only_export"}},
            "purpose": body.get("purpose", "research"), "training_eligible": False}
        with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=4) as bundle:
            for entry in selected:
                self.store.pulse(job, "exporting_bundle")
                path = self.asset_file(job["project"], body["asset_id"], entry["path"])
                require(sha256(path) == entry["sha256"], "ASSET_CHANGED", "An exported file fingerprint differs", status=409)
                bundle.write(path, "episode_bundle_v1/" + entry["path"])
            bundle.writestr("export_manifest.json", json.dumps(export_manifest, ensure_ascii=False, allow_nan=False))
        with zipfile.ZipFile(archive) as bundle:
            require(bundle.testzip() is None and len(bundle.namelist()) == len(selected)+1, "EXPORT_READBACK", "Archive readback failed")
        write_json(stage / "export.json", {"source_asset_id": body["asset_id"], "kind": body.get("kind"),
            "file_count": len(selected)+1, "archive_sha256": sha256(archive), "gt_available": asset["kind"] == "simulated_white_model" and body.get("kind") != "media",
            "purpose": body.get("purpose", "research"), "training_eligible": False})
        self._publish(job, stage, "export", "export_asset_id", parents=[body["asset_id"]], metadata={"file_count": len(selected)+1})

    def run_once(self):
        job = self.store.claim()
        if job is None:
            return None
        stop = threading.Event()
        def heartbeat():
            while not stop.wait(15):
                try:
                    self.store.pulse(job)
                except EngineError:
                    return
        thread = threading.Thread(target=heartbeat, daemon=True)
        thread.start()
        stage = self.project_root(job["project"]) / "staging" / (job["id"] + "_" + job["lease"])
        try:
            self.store.pulse(job, "starting")
            mode = job["payload"]["mode"]
            if mode == "export":
                if not job["result"].get("export_asset_id"):
                    self._export(job, stage)
            else:
                source_id = job["result"].get("source_asset_id") or job["payload"].get("source_asset_id")
                if source_id is None:
                    source_id = self._scene(job, stage / "source")
                elif not job["result"].get("source_asset_id"):
                    self.store.add_result(job, source_asset_id=source_id)
                source = self._asset(job["project"], source_id)
                if source["kind"] == "simulated_white_model":
                    require(source["quality"]["source_task"]["status"] == "pass", "SOURCE_TASK_FAILED",
                            "White-model output is preserved; the physical task did not pass", status=409)
                if mode != "scene_only" and not job["result"].get("render_asset_id"):
                    if not self._render(job, stage / "render", source_id):
                        return self.get_job(job["project"], job["id"])
            self.store.finish(job)
        except Exception as exc:
            # Do not expose traceback/local paths, upstream response text or credentials.
            error = exc.as_dict() if isinstance(exc, EngineError) else {"code": "INTERNAL_FAILURE", "message": "Worker stage failed; inspect local diagnostics"}
            # Diagnostic type and source locations are sufficient to locate an
            # unexpected failure without persisting response bodies or secrets.
            import traceback
            write_json(self.root / "diagnostics" / (job["id"] + ".json"),
                       {"job_id": job["id"], "exception_type": type(exc).__name__,
                        "code": error["code"], "frames": [{"file": Path(f.filename).name,
                        "function": f.name, "line": f.lineno} for f in traceback.extract_tb(exc.__traceback__)[-8:]]})
            state = "submission_unknown" if isinstance(exc, SubmissionUnknown) else "failed"
            attempt = self.store.attempt(job["project"], job["id"])
            if attempt and attempt["phase"] in {"intent", "unknown"} and not attempt["task_id"]:
                state = "submission_unknown"
                error = {"code": "H3_SUBMISSION_UNKNOWN", "message": "Submission may have reached H3; reconcile before any new paid request"}
            try:
                self.store.finish(job, state=state, error=error)
            except EngineError:
                pass  # A newer lease owns the state; never overwrite it.
        finally:
            stop.set()
            thread.join(timeout=2)
        return self.get_job(job["project"], job["id"])
