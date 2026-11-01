from fractions import Fraction
import json
from pathlib import Path
import xml.etree.ElementTree as ET
import zipfile

import av
import numpy as np
import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from white_render_backend.api import create_app
from white_render_backend.domain import EngineError
from white_render_backend.synthetic.contracts import RenderOptions
from white_render_backend.synthetic.credentials import load_api_key
from white_render_backend.synthetic.providers.minimax import MiniMaxProvider
from white_render_backend.synthetic.quality import validate_source_video


def encode(path, duration, fps=24):
    with av.open(str(path),'w') as out:
        stream=out.add_stream('libx264',rate=fps)
        stream.width,stream.height,stream.pix_fmt=320,256,'yuv420p'
        for i in range(round(duration*fps)):
            data=np.full((256,320,3),100+i%100,np.uint8)
            f=av.VideoFrame.from_ndarray(data,format='rgb24')
            f.pts,f.time_base=i,Fraction(1,fps)
            for packet in stream.encode(f):out.mux(packet)
        for packet in stream.encode():out.mux(packet)
    return path


@pytest.mark.parametrize('seconds',[2,5,13.8,15])
def test_source_duration_supported(tmp_path,seconds):
    fps=20 if seconds==13.8 else 24
    video=encode(tmp_path/'input.mp4',seconds,fps)
    assert validate_source_video(video,allow_low_fps=True)['duration_seconds']==pytest.approx(seconds)


@pytest.mark.parametrize('seconds',[1.5,15+1/24,16])
def test_duration_outside_range_is_rejected(tmp_path,seconds):
    video=encode(tmp_path/'bad.mp4',seconds)
    with pytest.raises(EngineError,match='SOURCE_DURATION'):
        validate_source_video(video)


def test_upload_full_low_fps_and_quote_budget(tmp_path):
    video=encode(tmp_path/'original.mp4',13.8,20)
    app=create_app(tmp_path/'work',auth={'test':{'project':'p','roles':['editor']}},synthetic_provider=MiniMaxProvider('test-key'))
    client=TestClient(app,headers={'Authorization':'Bearer test'})
    prefix='/api/projects/p/'
    u=client.post(prefix+'uploads',files={'file':('input.mp4',video.read_bytes(),'video/mp4')}).json()
    response=client.post(prefix+'synthetic/assets',json={'upload_id':u['upload_id']})
    assert response.status_code==201,response.text
    asset=response.json(); identity=asset['asset_id']
    assert asset['metadata']['fps']==24
    assert asset['metadata']['frame_count']==332
    assert asset['metadata']['duration_seconds']>=13.8
    service=app.state.synthetic
    original=service.asset_file('p',identity,'media/original.mp4')
    assert original.read_bytes()==video.read_bytes()
    prep=json.loads(service.asset_file('p',identity,'preprocessing.json').read_text())
    assert prep['trimmed'] is False and prep['speed_changed'] is False
    assert prep['mapping'][-1]['original_frame']==275
    ref={'source_asset_id':identity,'appearance_description':'Natural oak tabletop and satin polymer robot'}
    preflight=client.post(prefix+'synthetic/preflight',json=ref)
    assert preflight.status_code==200,preflight.text
    assert preflight.json()['provider']['request']['duration']==14
    quote=client.post(prefix+'synthetic/quotes',json=ref).json()
    assert quote['duration_seconds']==14
    assert quote['estimated_cost_minor_units']==1392
    body={'mode':'uploaded_video_render','source_asset_id':identity,'render':{
        'quote_id':quote['quote_id'],'max_cost_minor_units':500,'appearance_description':ref['appearance_description']}}
    assert client.post(prefix+'synthetic/jobs',json=body,headers={'Idempotency-Key':'low-budget'}).status_code==409
    body['render']['max_cost_minor_units']=1500
    body['render']['duration_seconds']=5
    assert client.post(prefix+'synthetic/jobs',json=body,headers={'Idempotency-Key':'duration-mismatch'}).status_code==409
    body['render'].pop('duration_seconds')
    response=client.post(prefix+'synthetic/jobs',json=body,headers={'Idempotency-Key':'full-length'})
    assert response.status_code==202,response.text
    job=service.store.job('p',response.json()['id'])
    assert job['payload']['render']['duration_seconds']==14
    # Stop before worker: no real generation is requested by these tests.
    service.action('p',job['id'],'cancel')


@pytest.mark.parametrize('duration',[4,10,15])
def test_output_schema_accepts_supported_integer(duration):
    assert RenderOptions(appearance_description='oak',quote_id='q',max_cost_minor_units=1500,duration_seconds=duration).duration_seconds==duration


@pytest.mark.parametrize('duration',[3,16,True,5.0,'14'])
def test_output_schema_rejects_invalid(duration):
    with pytest.raises(ValidationError):
        RenderOptions(appearance_description='oak',quote_id='q',max_cost_minor_units=1500,duration_seconds=duration)


def test_file_credential_precedes_old_environment_and_explicit_key_precedes_file(tmp_path,monkeypatch):
    doc=tmp_path/'credential.docx'
    expected='sk-'+'testvalue'*5
    with zipfile.ZipFile(doc,'w') as z:
        z.writestr('word/document.xml','<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"><w:t>'+expected+'</w:t></w:document>')
    monkeypatch.setenv('MINIMAX_API_KEY_FILE',str(doc))
    monkeypatch.setenv('MINIMAX_API_KEY','old-key')
    assert load_api_key()==expected
    assert MiniMaxProvider()._key==expected
    assert MiniMaxProvider('explicit-test-key')._key=='explicit-test-key'
    doc.write_bytes(b'bad document')
    with pytest.raises(EngineError,match='H3_CREDENTIAL_FILE') as error:
        load_api_key()
    assert expected not in str(error.value)
    unavailable=MiniMaxProvider()
    assert not unavailable.configured
    assert unavailable.capabilities()['credential_status']=='invalid_file'
