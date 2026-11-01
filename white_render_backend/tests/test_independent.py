import subprocess
import sys
from pathlib import Path

def test_app_starts_without_original_engine_or_dataset_dependencies(tmp_path):
    source=Path(__file__).resolve().parents[1]/'src'
    code='''
import sys,importlib.abc
sys.path.insert(0,sys.argv[1])
class Deny(importlib.abc.MetaPathFinder):
    def find_spec(self,fullname,path=None,target=None):
        if fullname.split('.')[0] in {'naso_data_engine','lerobot','datasets','pyarrow','sqlalchemy','huggingface_hub'}:
            raise ImportError('Excluded dependency requested: '+fullname)
sys.meta_path.insert(0,Deny())
from white_render_backend.api import create_app
from fastapi.testclient import TestClient
c=TestClient(create_app(sys.argv[2],auth={'a':{'project':'p','roles':['editor']}}))
assert c.get('/healthz').status_code==200
r=c.get('/api/projects/p/synthetic/capabilities',headers={'Authorization':'Bearer a'})
assert r.status_code==200
paths=c.get('/openapi.json').json()['paths']
assert not any('/imports' in p or '/builds' in p or '/releases' in p for p in paths)
print('standalone dependency isolation passed')
'''
    result=subprocess.run([sys.executable,'-c',code,str(source),str(tmp_path/'app')],capture_output=True,text=True)
    assert result.returncode==0,result.stderr
