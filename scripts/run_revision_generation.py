"""Generate prespecified independent data splits without examining outcomes."""
import argparse
import json
from pathlib import Path
import subprocess
import sys


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--family",required=True,choices=["zz","exchange"])
    a=ap.parse_args(); design=json.loads(Path("revision/stage2_design.json").read_text())
    root=Path("data_revision")/a.family; root.mkdir(parents=True,exist_ok=True)
    for split,(parameter,eta,n) in design[a.family+"_seeds"].items():
        path=root/(split+".npz")
        if path.exists(): raise FileExistsError(path)
        cmd=[sys.executable,"scripts/gen_collision_independent_eta_split.py","--family",a.family,
             "--out",str(path),"--n",str(n),"--parameter-seed",str(parameter),"--eta-seed",str(eta)]
        with (root/(split+"_generation.log")).open("w") as log:
            subprocess.run(cmd,check=True,stdout=log,stderr=log)
        print("generated",a.family,split,"n",n,flush=True)


if __name__=="__main__": main()
