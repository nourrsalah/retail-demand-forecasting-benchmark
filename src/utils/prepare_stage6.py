"""Pin public checkpoint revisions, download only safetensors/config, hash files."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import time
ROOT=Path(__file__).resolve().parents[2]
os.environ['HF_HOME']=str(ROOT/'.cache/huggingface')
os.environ['HF_HUB_DISABLE_XET']='1'
os.environ['HF_HUB_DISABLE_IMPLICIT_TOKEN']='1'
os.environ.setdefault('HF_HUB_DOWNLOAD_TIMEOUT','60')
from huggingface_hub import HfApi, snapshot_download


def main():
    settings=json.loads((ROOT/'configs/stage6.json').read_text())
    lock_path=ROOT/settings['checkpoint_lock']
    lock=json.loads(lock_path.read_text()) if lock_path.exists() else {'models':{}}
    api=HfApi(token=False)
    for spec in settings['models']:
        key=spec['id']
        if key not in lock['models']:
            info=api.model_info(spec['repo_id'],files_metadata=True)
            files=[f for f in info.siblings if f.rfilename in ['config.json','model.safetensors','README.md']]
            if not any(f.rfilename=='model.safetensors' for f in files):raise ValueError('Expected safe checkpoint absent')
            lock['models'][key]={'repo_id':spec['repo_id'],'revision':info.sha,
                'files':{f.rfilename:{'size':f.size,'expected_sha256':getattr(f.lfs,'sha256',None) if f.lfs else None} for f in files}}
            lock_path.write_text(json.dumps(lock,indent=2)+'\n')
        item=lock['models'][key]
        if item.get('snapshot_path'):
            cached=ROOT/item['snapshot_path']
            valid=True
            for name,meta in item['files'].items():
                if not (cached/name).is_file():
                    valid=False;break
                with (cached/name).open('rb') as handle:
                    valid=hashlib.file_digest(handle,'sha256').hexdigest()==meta.get('sha256')
                if not valid:break
            if valid:
                print(f'Reusing verified {key}',flush=True)
                continue
        required=sum(f.get('size') or 0 for f in item['files'].values())
        minimum=settings['minimum_free_disk_gib']*1024**3
        if shutil.disk_usage(ROOT).free<required+minimum:raise RuntimeError('Insufficient disk headroom; no files removed')
        print(f'Downloading {key} at {item["revision"]}: {required/1024**2:.1f} MiB',flush=True)
        start=time.perf_counter()
        path=Path(snapshot_download(item['repo_id'],revision=item['revision'],
             allow_patterns=list(item['files']),cache_dir=ROOT/'.cache/huggingface/hub',token=False,max_workers=1))
        item['snapshot_path']=str(path.relative_to(ROOT));item['download_seconds']=time.perf_counter()-start
        for name,meta in item['files'].items():
            with (path/name).open('rb') as f: digest=hashlib.file_digest(f,'sha256').hexdigest()
            if meta['expected_sha256'] and meta['expected_sha256']!=digest:raise ValueError('Checkpoint hash mismatch')
            meta['sha256']=digest
        lock_path.write_text(json.dumps(lock,indent=2)+'\n')
        print(f'Verified {key}',flush=True)


if __name__=='__main__':main()
