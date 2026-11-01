"""Only the local file space for white-model/render uploads; no dataset engine."""
import os
from pathlib import Path
from uuid import uuid4
from .domain import require
from .util import contained, safe_id, sha256

class Workspace:
    def __init__(self,root):
        self.root=Path(root).resolve()
        self.root.mkdir(parents=True,exist_ok=True)
    def project_root(self,project):return self.root/'projects'/safe_id(project)
    def upload(self,project,filename,stream):
        require(filename and Path(filename).name==filename and '/' not in filename and '\\' not in filename,'UPLOAD_NAME','Invalid filename')
        root=self.project_root(project)/'uploads';root.mkdir(parents=True,exist_ok=True)
        dest=contained(root,uuid4().hex+'_'+filename)
        temporary=dest.with_suffix('.partial');size=0
        try:
            with temporary.open('xb') as out:
                for block in iter(lambda:stream.read(1024*1024),b''):
                    size+=len(block)
                    require(size<=45_000_000,'UPLOAD_LIMIT','Upload exceeds 45 MB',status=413)
                    out.write(block)
            os.rename(temporary,dest)
        finally:temporary.unlink(missing_ok=True)
        return {'path':str(dest),'filename':filename,'size':size,'sha256':sha256(dest)}
