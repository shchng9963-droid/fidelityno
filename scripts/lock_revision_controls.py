"""Lock strongest nonneural controls on validation, before test access.

This explicitly includes zero-query-shot priors and prevents attributing a
measurement saving to a neural architecture when a simpler prior suffices.
"""
import hashlib
import json
from pathlib import Path
import pandas as pd


for family in ["zz","exchange"]:
    root=Path("results_revision/stage2")/family
    if (root/"test/summary.csv").exists(): raise RuntimeError("test has already been evaluated")
    df=pd.read_csv(root/"validation/summary.csv")
    rows=[]
    for target in [.065,.053]:
        candidates=df[(~df.method.str.startswith("neural")) & (df.method!="dfe")
                      & (~df.method.str.endswith("_raw")) & (df.selection_score<=target)]
        choice=None
        if len(candidates):
            best=candidates.sort_values(["shots","offline_shots","selection_score","allocation"]).iloc[0]
            choice={k:(int(best[k]) if k in ["shots","offline_shots"] else str(best[k]))
                    for k in ["method","allocation","shots","offline_shots"]}
            choice["validation_mae"]=float(best.mae)
        rows.append(dict(target=target,control=choice))
    payload=dict(status="additional control locked after validation and before test",
                 motivation="strong simple and zero-query-shot alternatives required by reviewers",
                 validation_summary_sha256=hashlib.sha256((root/"validation/summary.csv").read_bytes()).hexdigest(),
                 controls=rows)
    (root/"validation/control_selection.json").write_text(json.dumps(payload,indent=2))
    print(family,json.dumps(payload),flush=True)
