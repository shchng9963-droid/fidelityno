"""Additive public code/data archive; no private manuscript material."""
import argparse,hashlib,json,zipfile
from pathlib import Path

ap=argparse.ArgumentParser();ap.add_argument('--out',required=True);a=ap.parse_args()
out=Path(a.out)
if out.exists():raise FileExistsError(out)
out.parent.mkdir(parents=True,exist_ok=True)
entries=[]
for folder in ['configs','models','physics','scripts','tests','data_revision/stage5','results_revision/stage5']:
    for p in sorted(Path(folder).rglob('*')):
        if p.is_file() and '__pycache__' not in p.parts and p.suffix not in ['.pyc','.pdf','.png','.log']:
            entries.append(p)
for p in ['README.md','LICENSE','pyproject.toml','Makefile','train.py','eval.py',
          'revision/stage5_design.json','revision/REPRODUCE_INDEPENDENT_WEIGHTS.txt']:
    entries.append(Path(p))
hashes={}
with zipfile.ZipFile(out,'w') as archive:
    for p in entries:
        assert not any(x in p.parts for x in ['manuscript','submitted_sources'])
        content=p.read_bytes()
        if p.suffix=='.sh' or p.name=='Makefile':content=content.replace(b'\r\n',b'\n')
        hashes[p.as_posix()]=hashlib.sha256(content).hexdigest()
        archive.writestr(p.as_posix(),content,compress_type=zipfile.ZIP_STORED if p.suffix=='.npz' else zipfile.ZIP_DEFLATED,compresslevel=6)
    archive.writestr('FILE_SHA256_STAGE5.json',json.dumps(hashes,indent=2),compress_type=zipfile.ZIP_DEFLATED)
print(json.dumps(dict(path=str(out),files=len(entries),bytes=out.stat().st_size,
    sha256=hashlib.sha256(out.read_bytes()).hexdigest()),indent=2))
