"""Business hard-point tests. The small simulator here is explicitly a test double."""
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
import json
from pathlib import Path
import shutil
import zipfile

import av
from fastapi.testclient import TestClient
import httpx
import numpy as np
import pytest

from white_render_backend.api import create_app
from white_render_backend.domain import EngineError
from white_render_backend.synthetic.providers.minimax import MiniMaxProvider
from white_render_backend.synthetic.store import Store
from white_render_backend.util import write_json


@pytest.fixture(scope="module")
def video(tmp_path_factory):
    path = tmp_path_factory.mktemp("synthetic_api_video") / "white.mp4"
    with av.open(str(path), "w") as output:
        stream = output.add_stream("libx264", rate=24)
        stream.width, stream.height, stream.pix_fmt = 864, 480, "yuv420p"
        for i in range(120):
            frame = av.VideoFrame.from_ndarray(np.full((480,864,3), 40+i, np.uint8), format="rgb24")
            for packet in stream.encode(frame):output.mux(packet)
        for packet in stream.encode():output.mux(packet)
    return path


class Upstream:
    def __init__(self, video):
        self.video, self.posts, self.gets = video, 0, 0
        self.behavior = "success"
        self.pending = False
    def handle(self, request):
        if request.method == "POST":
            self.posts += 1
            if self.behavior == "timeout":raise httpx.ReadTimeout("secret must not leak", request=request)
            return httpx.Response(200, json={"task_id":"12345"})
        self.gets += 1
        status = "running" if self.pending else "failed" if self.behavior == "failed" else "succeeded"
        return httpx.Response(200, json={"task":{"id":"12345","status":status,
                             "content":{"url":"https://cdn.hailuoai.com/result.mp4"}}})
    def download(self, request):
        assert "authorization" not in request.headers
        return httpx.Response(200, content=self.video.read_bytes())


@pytest.fixture
def system(tmp_path, video):
    upstream = Upstream(video)
    provider = MiniMaxProvider("test-only-secret", client=httpx.Client(transport=httpx.MockTransport(upstream.handle)),
                              download_client=httpx.Client(transport=httpx.MockTransport(upstream.download)))
    def simulator(output, spec, progress):
        (output / "media").mkdir(parents=True)
        shutil.copyfile(video, output / "media/white_model.mp4")
        write_json(output / "manifest.json", {"schema":"episode_bundle_v1", "test_double":True})
        write_json(output / "states.json", {"test_double":True})
        return {"validation":{"status":"pass"},"channels":{"state":{"available":True}},"metadata":{}}
    app = create_app(tmp_path / "service", auth={"editor":{"project":"p","roles":["editor"]},
        "viewer":{"project":"p","roles":["viewer"]},"other":{"project":"other","roles":["editor"]},
        "operator":{"project":"p","roles":["render_operator"]}}, synthetic_provider=provider, simulator=simulator)
    return TestClient(app), app.state.synthetic, upstream


def call(client, method, path, body=None, key=None, token="editor"):
    return client.request(method, "/api/projects/p/"+path, json=body,
        headers={"Authorization":"Bearer "+token, **({"Idempotency-Key":key} if key else {})})


def new_spec(client):
    response = call(client,"POST","synthetic/specs",{"description":"Franka 把红色方块抓起并放到桌面右侧，固定相机，5秒"})
    assert response.status_code == 200, response.text
    value = response.json()
    assert value["status"] == "ready"
    return {"spec_id":value["spec_id"],"spec_revision":1}


def paid_request(client, reference, mode="scene_and_render"):
    quote = call(client,"POST","synthetic/quotes",{**reference,"appearance_description":"真实木纹桌面和柔和侧光"}).json()
    return {"mode":mode,**reference,"render":{"quote_id":quote["quote_id"],"max_cost_minor_units":600,
        "appearance_description":"真实木纹桌面和柔和侧光"}}


def run_ready(service):
    with service.store.connect(True) as db:db.execute("UPDATE jobs SET next_run=0 WHERE state='queued'")
    return service.run_once()


def test_white_only_export_range_auth_and_no_paid_call(system):
    client, service, upstream = system
    service.provider._key = ""  # White-only must work without credentials.
    body = {"mode":"scene_only",**new_spec(client)}
    response = call(client,"POST","synthetic/jobs",body,"white")
    assert response.status_code == 202, response.text
    job = run_ready(service)
    assert job["state"] == "succeeded", job
    source_id = job["result"]["source_asset_id"]
    assert upstream.posts == 0
    asset = call(client,"GET",f"synthetic/assets/{source_id}",token="viewer").json()
    assert "root" not in asset
    path = f"/api/projects/p/synthetic/assets/{source_id}/files/media/white_model.mp4"
    assert client.get(path).status_code == 401
    assert client.get(path,headers={"Authorization":"Bearer other"}).status_code == 403
    partial = client.get(path, headers={"Authorization":"Bearer viewer","Range":"bytes=0-31"})
    assert partial.status_code == 206 and len(partial.content) == 32
    assert "Content-Range" in partial.headers
    link = call(client,"POST",f"synthetic/assets/{source_id}/media-link",{"filename":"media/white_model.mp4"},token="viewer").json()
    assert client.get(link["url"],headers={"Range":"bytes=0-7"}).status_code == 206
    assert client.get(link["url"].replace("/p/","/other/")).status_code == 404
    assert call(client,"POST",f"synthetic/assets/{source_id}/media-link",{"filename":"../tasks.sqlite"}).status_code == 404
    export = call(client,"POST","synthetic/exports",{"asset_id":source_id,"kind":"semantic_gt"},"export")
    assert export.status_code == 202
    finished = run_ready(service)
    assert finished["state"] == "succeeded"
    export_id = finished["result"]["export_asset_id"]
    data = call(client,"GET",f"synthetic/assets/{export_id}/files/bundle.zip")
    with zipfile.ZipFile(BytesIO(data.content)) as z:
        assert z.testzip() is None and "episode_bundle_v1/media/white_model.mp4" in z.namelist()
        assert json.loads(z.read("export_manifest.json"))["source_kind"] == "simulated_white_model"
    media_job = call(client,"POST","synthetic/exports",{"asset_id":source_id,"kind":"media"},"export-media")
    assert media_job.status_code == 202
    media_result = run_ready(service)
    media_asset = media_result["result"]["export_asset_id"]
    archive = call(client,"GET",f"synthetic/assets/{media_asset}/files/bundle.zip")
    with zipfile.ZipFile(BytesIO(archive.content)) as z:
        manifest = json.loads(z.read("export_manifest.json"))
        assert manifest["channels"]["gt"]["available"] is False
        assert "episode_bundle_v1/manifest.json" not in z.namelist()
        assert "episode_bundle_v1/states.json" not in z.namelist()
    assert call(client,"POST","synthetic/exports",{"asset_id":source_id,"purpose":"action_supervised_training"},"bad").status_code == 409


def test_a_full_h3_lifecycle_with_transport_double(system):
    client, service, upstream = system
    body = paid_request(client,new_spec(client))
    created = call(client,"POST","synthetic/jobs",body,"paid")
    assert created.status_code == 202, created.text
    same = call(client,"POST","synthetic/jobs",body,"paid")
    assert same.json()["id"] == created.json()["id"]
    first = run_ready(service)
    assert first["state"] == "queued" and upstream.posts == 1, first
    second = run_ready(service)
    assert second["state"] == "succeeded" and upstream.posts == 1, second
    source_id, render_id = second["result"]["source_asset_id"], second["result"]["render_asset_id"]
    asset = call(client,"GET",f"synthetic/assets/{render_id}").json()
    assert asset["parents"] == [source_id]
    assert asset["quality"]["training_eligible"] is False
    assert asset["quality"]["geometry"]["status"] == "unknown"
    assert "test-only-secret" not in json.dumps(asset)
    assert call(client,"POST","synthetic/jobs",body,"paid_again").status_code == 409


def test_uploaded_b_preflight_without_generation_or_fake_gt(system, video):
    client, service, upstream = system
    uploaded = client.post("/api/projects/p/uploads",files={"file":("input.mp4",video.read_bytes(),"video/mp4")},headers={"Authorization":"Bearer editor"})
    assert uploaded.status_code == 200
    reg = call(client,"POST","synthetic/assets",{"upload_id":uploaded.json()["upload_id"]})
    assert reg.status_code == 201, reg.text
    asset_id = reg.json()["asset_id"]
    preflight = call(client,"POST","synthetic/preflight",{"source_asset_id":asset_id,"appearance_description":"木纹桌面"})
    assert preflight.status_code == 200 and preflight.json()["provider"]["request_valid"] is True
    assert upstream.posts == 0 and upstream.gets == 0
    assert call(client,"POST","synthetic/preflight",{"source_asset_id":asset_id,"check_provider":True,"known_task_id":"99999"}).status_code == 403
    assert call(client,"POST","synthetic/exports",{"asset_id":asset_id,"kind":"semantic_gt"},"false-gt").status_code == 409
    # Exercise B's worker using a transport double only, never the real provider.
    body = paid_request(client,{"source_asset_id":asset_id},"uploaded_video_render")
    assert call(client,"POST","synthetic/jobs",body,"upload-render").status_code == 202
    run_ready(service)
    assert run_ready(service)["state"] == "succeeded" and upstream.posts == 1


def test_unknown_post_is_not_resubmitted_and_needs_operator_reconciliation(system):
    client, service, upstream = system
    upstream.behavior = "timeout"
    body = paid_request(client,new_spec(client))
    call(client,"POST","synthetic/jobs",body,"unknown")
    failed = run_ready(service)
    assert failed["state"] == "submission_unknown" and upstream.posts == 1
    assert "source_asset_id" in failed["result"]
    assert "secret" not in json.dumps(failed)
    assert call(client,"POST",f"jobs/{failed['id']}/retry").status_code == 409
    assert service.run_once() is None
    assert call(client,"POST",f"synthetic/jobs/{failed['id']}/reconcile",{"task_id":"12345","note":"verified"}).status_code == 403
    upstream.behavior = "success"
    response = call(client,"POST",f"synthetic/jobs/{failed['id']}/reconcile",{"task_id":"12345","note":"Matched source and time in provider console"},token="operator")
    assert response.status_code == 200
    assert run_ready(service)["state"] == "succeeded" and upstream.posts == 1


def test_provider_failure_keeps_source_and_forbids_paid_retry(system):
    client, service, upstream = system
    upstream.behavior = "failed"
    call(client,"POST","synthetic/jobs",paid_request(client,new_spec(client)),"failure")
    run_ready(service)
    job = run_ready(service)
    assert job["state"] == "failed" and job["error"]["code"] == "H3_TASK_FAILED"
    assert call(client,"GET",f"synthetic/assets/{job['result']['source_asset_id']}").status_code == 200
    assert call(client,"POST",f"jobs/{job['id']}/retry").status_code == 409
    assert upstream.posts == 1


def test_expired_paid_intent_never_reposts(system):
    client, service, upstream = system
    call(client,"POST","synthetic/jobs",paid_request(client,new_spec(client)),"crash")
    job = service.store.claim()
    source = service._scene(job,service.project_root("p")/"staging"/"crash_source")
    service.store.begin_submission(job,{"source_asset_id":source})
    with service.store.connect(True) as db:db.execute("UPDATE jobs SET lease_until=0 WHERE id=?",(job["id"],))
    resumed = run_ready(service)
    assert resumed["state"] == "submission_unknown" and upstream.posts == 0


def test_store_claim_fencing_and_idempotency(tmp_path):
    store = Store(tmp_path/"queue.sqlite")
    original = store.submit("p",{"mode":"scene_only"},"same")
    with ThreadPoolExecutor(max_workers=4) as pool:claims=list(pool.map(lambda _:store.claim(),range(4)))
    job = next(j for j in claims if j)
    assert sum(j is not None for j in claims) == 1
    with store.connect(True) as db:db.execute("UPDATE jobs SET lease_until=0 WHERE id=?",(job["id"],))
    fresh = store.claim()
    with pytest.raises(EngineError,match="LEASE_LOST"):store.begin_submission(job,{})
    store.begin_submission(fresh,{})
    with pytest.raises(EngineError,match="SUBMISSION_EXISTS"):store.begin_submission(fresh,{})
    with pytest.raises(EngineError,match="IDEMPOTENCY_CONFLICT"):store.submit("p",{"mode":"different"},"same")
    with pytest.raises(EngineError,match="NOT_FOUND"):store.job("other",original["id"])


def test_resume_after_render_publication_does_not_download_again(system):
    client, service, upstream = system
    call(client,"POST","synthetic/jobs",paid_request(client,new_spec(client)),"publication")
    run_ready(service)
    done = run_ready(service)
    old_result = done["result"]
    reads = upstream.gets
    # Crash immediately after publishing render, before the terminal job write.
    with service.store.connect(True) as db:
        db.execute("UPDATE jobs SET state='running',lease_until=0 WHERE id=?",(done["id"],))
    resumed = run_ready(service)
    assert resumed["state"] == "succeeded" and resumed["result"] == old_result
    assert upstream.posts == 1 and upstream.gets == reads


def test_invalid_budget_quote_and_schema_do_not_submit(system):
    client, service, upstream = system
    ref = new_spec(client)
    assert call(client,"POST","synthetic/jobs",{"mode":"scene_only",**ref}).status_code == 422
    body = paid_request(client,ref)
    body["render"]["max_cost_minor_units"] = 1
    assert call(client,"POST","synthetic/jobs",body,"low").status_code == 409
    body["render"]["max_cost_minor_units"] = 600
    body["render"]["appearance_description"] = "另外的外观"
    assert call(client,"POST","synthetic/jobs",body,"changed").status_code == 409
    body["render"]["automatic_paid_retries"] = True
    assert call(client,"POST","synthetic/jobs",body,"retry").status_code == 422
    assert upstream.posts == 0
