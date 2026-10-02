"""Server pipeline: validation locks first, then fresh generation/test/inference."""
import argparse,json,os,subprocess,sys,time
from concurrent.futures import ThreadPoolExecutor,as_completed
from pathlib import Path

ap=argparse.ArgumentParser();ap.add_argument('--phase',choices=['validation','test'],required=True)
ap.add_argument('--ckpts',nargs=5,required=True);a=ap.parse_args()
logs=Path('results_revision/stage5/logs');logs.mkdir(parents=True,exist_ok=True)
def run(family):
    name=family+'_'+a.phase;status=logs/(name+'_status.json')
    if status.exists():raise FileExistsError(status)
    env=dict(os.environ,OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2',CUDA_VISIBLE_DEVICES='1' if family=='exchange' else '0')
    ckpts=[] if family=='two_bath' else ['--ckpts',*a.ckpts]
    modes=['validation'] if a.phase=='validation' else ['generate','test']
    commands=[[sys.executable,'scripts/eval_revision_independent_weights.py','--family',family,'--mode',mode,*ckpts] for mode in modes]
    if a.phase=='test':commands.append([sys.executable,'scripts/analyze_revision_independent_weights.py','--family',family])
    start=time.time();code=0
    with (logs/(name+'.log')).open('w') as log:
        for command in commands:
            result=subprocess.run(command,env=env,stdout=log,stderr=log)
            if result.returncode:code=result.returncode;break
    rec=dict(family=family,phase=a.phase,seconds=time.time()-start,returncode=code)
    status.write_text(json.dumps(rec,indent=2));return rec
with ThreadPoolExecutor(max_workers=3) as pool:
    records=[]
    for future in as_completed([pool.submit(run,f) for f in ['zz','exchange','two_bath']]):
        rec=future.result();records.append(rec);print(json.dumps(rec),flush=True)
if any(r['returncode'] for r in records):raise SystemExit('one or more jobs failed')
