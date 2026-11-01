"""REST surface. Media links are short-lived project/file-scoped capabilities."""
import hashlib
import secrets
import time
from urllib.parse import quote

from fastapi import Depends, Header, Query
from fastapi.responses import FileResponse

from ..domain import require
from .contracts import (AssetBody, ExportBody, JobBody, MediaLinkBody, PreflightBody,
                        QuoteBody, ReconcileBody, SpecBody)


def register_routes(app, service, identity, access):
    prefix = "/api/projects/{project}/synthetic"
    read_roles = ["viewer", "editor", "publisher", "scorer"]

    @app.get(prefix + "/capabilities", tags=["synthetic"])
    def capabilities(project: str, user=Depends(identity)):
        access(project, user, read_roles)
        return service.capabilities()

    @app.post(prefix + "/specs", tags=["synthetic"])
    def specs(project: str, body: SpecBody, user=Depends(identity)):
        access(project, user, ["editor"])
        return service.create_spec(project, **body.model_dump())

    @app.post(prefix + "/assets", status_code=201, tags=["synthetic"])
    def create_asset(project: str, body: AssetBody, user=Depends(identity)):
        access(project, user, ["editor"])
        return service.register_asset(project, **body.model_dump())

    @app.get(prefix + "/assets", tags=["synthetic"])
    def assets(project: str, offset: int = Query(0, ge=0), limit: int = Query(50, ge=1, le=200), user=Depends(identity)):
        access(project, user, read_roles)
        rows = service.store.list(project, "asset")
        return {"items": [service.asset(project, r["id"]) for r in rows[offset:offset+limit]], "total": len(rows)}

    @app.get(prefix + "/assets/{asset_id}", tags=["synthetic"])
    def asset(project: str, asset_id: str, user=Depends(identity)):
        access(project, user, read_roles)
        return service.asset(project, asset_id)

    @app.get(prefix + "/assets/{asset_id}/quality", tags=["synthetic"])
    def quality(project: str, asset_id: str, user=Depends(identity)):
        access(project, user, read_roles)
        return service.asset(project, asset_id)["quality"]

    @app.get(prefix + "/assets/{asset_id}/files/{filename:path}", tags=["synthetic"])
    def file(project: str, asset_id: str, filename: str, download: bool = False, user=Depends(identity)):
        access(project, user, read_roles)
        path = service.asset_file(project, asset_id, filename)
        return FileResponse(path, filename=path.name if download else None,
                            headers={"Cache-Control": "private, no-store", "X-Content-Type-Options": "nosniff"})

    @app.post(prefix + "/assets/{asset_id}/media-link", tags=["synthetic"])
    def media_link(project: str, asset_id: str, body: MediaLinkBody, user=Depends(identity)):
        access(project, user, read_roles)
        service.asset_file(project, asset_id, body.filename)
        token = secrets.token_urlsafe(32)
        expires = time.time() + body.expires_seconds
        identity_hash = "link_" + hashlib.sha256(token.encode()).hexdigest()
        service.store.put(project, "media_link", identity_hash, {"asset_id": asset_id, "filename": body.filename, "expires_at": expires})
        return {"url": f"/api/projects/{quote(project, safe='')}/synthetic/media/{token}",
                "expires_at_epoch": expires, "note": "Anyone holding this URL can access this file until expiry"}

    @app.get(prefix + "/media/{token}", tags=["synthetic"])
    def signed_media(project: str, token: str):
        require(len(token) <= 100, "MEDIA_LINK", "Invalid media link", status=403)
        identity_hash = "link_" + hashlib.sha256(token.encode()).hexdigest()
        link = service.store.get(project, "media_link", identity_hash)["payload"]
        require(link["expires_at"] > time.time(), "MEDIA_LINK_EXPIRED", "Refresh the media link", status=403)
        path = service.asset_file(project, link["asset_id"], link["filename"])
        return FileResponse(path, headers={"Cache-Control": "private, no-store", "Referrer-Policy": "no-referrer",
                                         "X-Content-Type-Options": "nosniff"})

    @app.post(prefix + "/preflight", tags=["synthetic"])
    def preflight(project: str, body: PreflightBody, user=Depends(identity)):
        access(project, user, ["editor"])
        return service.preflight(project, body.model_dump(exclude_none=True))

    @app.post(prefix + "/quotes", tags=["synthetic"])
    def quotes(project: str, body: QuoteBody, user=Depends(identity)):
        access(project, user, ["editor"])
        return service.quote(project, body.model_dump(exclude_none=True))

    @app.post(prefix + "/jobs", status_code=202, tags=["synthetic"])
    def jobs(project: str, body: JobBody, user=Depends(identity), idempotency_key: str | None = Header(default=None)):
        access(project, user, ["editor"])
        return service.submit(project, body.model_dump(exclude_none=True), idempotency_key)

    @app.get(prefix + "/jobs", tags=["synthetic"])
    def list_jobs(project: str, offset: int = Query(0, ge=0), limit: int = Query(50, ge=1, le=200), user=Depends(identity)):
        access(project, user, read_roles)
        jobs = service.store.jobs(project)
        return {"items": [service.public_job(j) for j in jobs[offset:offset+limit]], "total": len(jobs)}

    @app.post(prefix + "/jobs/{job_id}/reconcile", tags=["synthetic"])
    def reconcile(project: str, job_id: str, body: ReconcileBody, user=Depends(identity)):
        # Operator attests the task's account/history matches the unknown input.
        # Normal editors cannot claim another tenant's task by guessing an ID.
        access(project, user, ["render_operator"])
        return service.reconcile(project, job_id, body.task_id, body.note)

    @app.post(prefix + "/exports", status_code=202, tags=["synthetic"])
    def exports(project: str, body: ExportBody, user=Depends(identity), idempotency_key: str | None = Header(default=None)):
        access(project, user, ["editor"])
        return service.export(project, body.model_dump(), idempotency_key)
