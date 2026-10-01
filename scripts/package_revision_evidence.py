"""Create a data/code-only local release candidate, with no publication action."""
import argparse
import hashlib
import json
from pathlib import Path
import zipfile


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--checkpoint-root',required=True)
    ap.add_argument('--out',required=True)
    args=ap.parse_args()
    out=Path(args.out)
    if out.exists():
        raise FileExistsError(out)
    out.parent.mkdir(parents=True,exist_ok=True)
    entries=[]
    for folder in ['configs','models','physics','scripts','tests','data_revision',
                   'results_revision/stage2','results_revision/stage3','results_revision/stage4']:
        for path in sorted(Path(folder).rglob('*')):
            if path.is_file() and '__pycache__' not in path.parts and path.suffix not in ['.pyc','.pdf','.png']:
                entries.append((path,path.as_posix()))
    for name in ['README.md','LICENSE','pyproject.toml','Makefile','train.py','eval.py',
                 'revision/stage2_design.json','revision/statistical_protocol.json',
                 'revision/protocol_addendum_after_pilot.json','revision/REPRODUCE_STAGE2_STAGE3.txt',
                 'revision/stage4_design.json','revision/REPRODUCE_FOLLOWUP.txt']:
        entries.append((Path(name),name))
    for seed in range(5):
        path=Path(args.checkpoint_root)/f'bidir_seed{seed}.pt'
        entries.append((path,f'checkpoints/collision/{path.name}'))
    hashes={}
    with zipfile.ZipFile(out,'w') as archive:
        for path,name in entries:
            content=path.read_bytes()
            if path.suffix=='.sh' or path.name=='Makefile':
                content=content.replace(b'\r\n',b'\n')
            hashes[name]=hashlib.sha256(content).hexdigest()
            archive.writestr(name,content,compress_type=zipfile.ZIP_STORED if path.suffix in ['.npz','.pt'] else zipfile.ZIP_DEFLATED,compresslevel=3)
        archive.writestr('FILE_SHA256.json',json.dumps(hashes,indent=2),compress_type=zipfile.ZIP_DEFLATED)
    print(json.dumps(dict(archive=str(out),files=len(entries),bytes=out.stat().st_size,
                         sha256=hashlib.sha256(out.read_bytes()).hexdigest()),indent=2))


if __name__=='__main__':
    main()
