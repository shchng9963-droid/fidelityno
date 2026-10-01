"""Run independent server jobs and retain complete logs and return codes."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--phase",choices=["prepare","validation","test"],required=True)
    a=ap.parse_args(); logs=Path("results_revision/stage2/logs"); logs.mkdir(parents=True,exist_ok=True)
    jobs=[]
    checkpoints=["/home/wangshuchang/fidelityno_prxq/checkpoints/collision/bidir_seed"+str(i)+".pt" for i in range(5)]
    for family in ["zz","exchange"]:
        if a.phase=="prepare":
            jobs.append((family+"_data",[sys.executable,"scripts/run_revision_generation.py","--family",family]))
            jobs.append((family+"_grid",[sys.executable,"scripts/eval_revision_grid.py","--family",family,
                "--n-base","2048" if family=="zz" else "1024","--seed","20260807" if family=="zz" else "20260810",
                "--out","results_revision/stage2/"+family+"/grid"]))
        else:
            jobs.append((family+"_"+a.phase,[sys.executable,"scripts/eval_revision_confirmatory.py",
                "--family",family,"--split",a.phase,"--ckpts",*checkpoints]))
    def run(item):
        name,cmd=item; start=time.time()
        env=dict(os.environ,OMP_NUM_THREADS="2",OPENBLAS_NUM_THREADS="2",MKL_NUM_THREADS="2")
        if name.startswith("exchange"): env["CUDA_VISIBLE_DEVICES"]="1"
        else: env["CUDA_VISIBLE_DEVICES"]="0"
        with (logs/(name+".log")).open("w") as f:
            result=subprocess.run(cmd,env=env,stdout=f,stderr=f)
        record=dict(name=name,command=cmd,returncode=result.returncode,seconds=time.time()-start)
        (logs/(name+"_status.json")).write_text(json.dumps(record,indent=2))
        return record
    records=[]
    with ThreadPoolExecutor(max_workers=4) as pool:
        for future in as_completed([pool.submit(run,j) for j in jobs]):
            record=future.result(); records.append(record); print(json.dumps(record),flush=True)
    if any(r["returncode"] for r in records): raise SystemExit("a job failed; inspect logs")


if __name__=="__main__": main()
