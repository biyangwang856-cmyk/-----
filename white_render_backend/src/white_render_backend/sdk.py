import httpx
from pathlib import Path
from urllib.parse import quote

class Client:
    def __init__(self,url,token,project,timeout=60):
            self.project=project
            self.http=httpx.Client(base_url=url.rstrip("/"),headers={"Authorization":f"Bearer {token}"},timeout=timeout)
    def request(self,method,path,**kwargs):
            result=self.http.request(method,f"/api/projects/{self.project}/{path.lstrip('/')}",**kwargs)
            result.raise_for_status()
            return result.json()
    def job(self,identity):return self.request("GET",f"jobs/{identity}")
    def upload(self,path):
            path=Path(path)
            with path.open("rb") as stream:return self.request("POST","uploads",files={"file":(path.name,stream,"application/octet-stream")})
    def cancel(self,job_id):return self.request("POST",f"jobs/{job_id}/cancel")
    def retry(self,job_id):return self.request("POST",f"jobs/{job_id}/retry")
    def close(self):self.http.close()
    def synthetic_capabilities(self):return self.request("GET","synthetic/capabilities")
    def create_scene_spec(self,description,appearance_description=""):
            return self.request("POST","synthetic/specs",json={"description":description,"appearance_description":appearance_description})
    def register_video(self,upload_id,description=""):
            return self.request("POST","synthetic/assets",json={"upload_id":upload_id,"description":description})
    def synthetic_preflight(self,**reference):return self.request("POST","synthetic/preflight",json=reference)
    def render_quote(self,**reference):return self.request("POST","synthetic/quotes",json=reference)
    def synthetic_job(self,idempotency_key,**spec):
            return self.request("POST","synthetic/jobs",json=spec,headers={"Idempotency-Key":idempotency_key})
    def synthetic_asset(self,asset_id):return self.request("GET",f"synthetic/assets/{asset_id}")
    def synthetic_quality(self,asset_id):return self.request("GET",f"synthetic/assets/{asset_id}/quality")
    def export_asset(self,asset_id,idempotency_key,kind="available",purpose="research"):
            return self.request("POST","synthetic/exports",json={"asset_id":asset_id,"kind":kind,"purpose":purpose},
                                headers={"Idempotency-Key":idempotency_key})
    def media_link(self,asset_id,filename,expires_seconds=300):
            return self.request("POST",f"synthetic/assets/{asset_id}/media-link",
                                json={"filename":filename,"expires_seconds":expires_seconds})
    def download_asset(self,asset_id,filename,destination):
            destination=Path(destination)
            destination.parent.mkdir(parents=True,exist_ok=True)
            url=f"/api/projects/{self.project}/synthetic/assets/{asset_id}/files/{quote(filename,safe='/')}"
            with self.http.stream("GET",url) as response:
                response.raise_for_status()
                with destination.open("xb") as output:
                    for chunk in response.iter_bytes():output.write(chunk)
            return destination
    def __enter__(self):return self
    def __exit__(self,*args):self.close()
