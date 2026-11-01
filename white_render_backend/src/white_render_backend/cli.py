import argparse
import os
import time
from pathlib import Path
from .api import create_app
from .runtime import Workspace
from .synthetic.service import SyntheticService
from .util import write_json

def main():
    parser=argparse.ArgumentParser(description='白模生成与H3渲染独立后端')
    parser.add_argument('--root',default=os.getenv('WHITE_RENDER_ROOT','work'))
    sub=parser.add_subparsers(dest='command',required=True)
    p=sub.add_parser('serve');p.add_argument('--host',default='127.0.0.1');p.add_argument('--port',type=int,default=8018)
    p=sub.add_parser('worker');p.add_argument('--once',action='store_true')
    p=sub.add_parser('openapi');p.add_argument('output')
    args=parser.parse_args()
    if args.command=='serve':
        import uvicorn
        uvicorn.run(create_app(args.root),host=args.host,port=args.port,access_log=False)
    elif args.command=='openapi':write_json(args.output,create_app(args.root,auth={}).openapi())
    else:
        service=SyntheticService(Workspace(args.root))
        while True:
            result=service.run_once()
            if result:print(result['id'],result['state'],result['phase'],flush=True)
            if args.once:return
            if result is None:time.sleep(2)
if __name__=='__main__':main()
