"""Paired bootstrap and shot accounting for locked validation choices."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd


def tensor(rows,errors,row_map,method,allocation,budget):
    chosen=rows[(rows.method==method)&(rows.allocation==allocation)&(rows.shots==budget)]
    seeds=sorted(chosen.checkpoint.unique())
    lookup={(int(r["cal_draw"]),int(r["repeat"]),r["checkpoint"]):i for i,r in chosen.iterrows()}
    return np.array([[[errors[row_map[lookup[d,r,s]]] for s in seeds] for r in range(5)] for d in range(20)])


def paired_bootstrap(a,b,n_boot=2000):
    nd,nr,ns,nq=a.shape
    if b.shape[:2]!=(nd,nr) or b.shape[-1]!=nq: raise ValueError("unpaired data")
    if b.shape[2] not in [1,ns]: raise ValueError("incompatible checkpoints")
    rng=np.random.default_rng(202610031)
    aa=a.reshape(nd*nr*ns,nq)
    bb=b.reshape(nd*nr*b.shape[2],nq)
    boot_a=[]; boot_b=[]
    for start in range(0,n_boot,64):
        batch=min(64,n_boot-start)
        w=np.zeros((batch,nd,nr),dtype=np.float32)
        for k in range(batch):
            ds=rng.integers(nd,size=nd); rs=rng.integers(nr,size=(nd,nr))
            np.add.at(w[k],(np.repeat(ds,nr),rs.ravel()),1/(nd*nr))
        sw=rng.multinomial(ns,np.full(ns,1/ns),size=batch).astype(np.float32)/ns
        w_a=np.einsum("bdr,bs->bdrs",w,sw).reshape(batch,-1)
        query_w=rng.multinomial(nq,np.full(nq,1/nq),size=batch).astype(np.float32)/nq
        amean=w_a@aa
        bmean=(w.reshape(batch,-1)@bb if b.shape[2]==1 else w_a@bb)
        boot_a.extend((amean*query_w).sum(1).tolist())
        boot_b.extend((bmean*query_w).sum(1).tolist())
    boot_a=np.array(boot_a); boot_b=np.array(boot_b); delta=boot_a-boot_b
    return dict(hybrid_mae=float(a.mean()),baseline_mae=float(b.mean()),difference=float(a.mean()-b.mean()),
        hybrid_ci95_low=float(np.quantile(boot_a,.025)),hybrid_ci95_high=float(np.quantile(boot_a,.975)),
        baseline_ci95_low=float(np.quantile(boot_b,.025)),baseline_ci95_high=float(np.quantile(boot_b,.975)),
        hybrid_upper95=float(np.quantile(boot_a,.95)),baseline_upper95=float(np.quantile(boot_b,.95)),
        difference_ci95_low=float(np.quantile(delta,.025)),difference_ci95_high=float(np.quantile(delta,.975)),
        noninferior_bonferroni_two_targets=bool(np.quantile(delta,.975)<.002),
        equivalent_bonferroni_two_targets=bool(np.quantile(delta,.025)>-.002 and np.quantile(delta,.975)<.002))


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--family",required=True,choices=["zz","exchange"])
    a=ap.parse_args(); family=Path("results_revision/stage2")/a.family
    rows=pd.read_csv(family/"test/runs.csv")
    archive=np.load(family/"test/paired_errors.npz")
    errors=archive["absolute_errors"]
    row_map={int(r):i for i,r in enumerate(archive["row_ids"])}
    lock=json.loads((family/"validation/selection.json").read_text())
    records=[]
    for choice in lock["selections"]:
        h=choice["hybrid"]; b=choice["dfe"]; target=choice["target"]
        if h is None or b is None:
            records.append(dict(family=a.family,target=target,status="target not reached in validation")); continue
        ha=tensor(rows,errors,row_map,h["method"],h["allocation"],h["shots"])
        ba=tensor(rows,errors,row_map,b["method"],b["allocation"],b["shots"])
        result=paired_bootstrap(ha,ba)
        certified=result["hybrid_upper95"]<=target and result["baseline_upper95"]<=target
        saving=b["shots"]-h["shots"]
        valid=certified and result["noninferior_bonferroni_two_targets"] and saving>0
        record=dict(family=a.family,target=target,hybrid_method=h["method"],hybrid_allocation=h["allocation"],
                    baseline_allocation=b["allocation"],hybrid_shots=h["shots"],baseline_shots=b["shots"],
                    offline_hybrid=h["offline_shots"],offline_baseline=b["offline_shots"],
                    common_target_confirmed=certified,validated_shot_saving=valid,
                    descriptive_query_shot_reduction=saving/b["shots"],
                    break_even_queries=(h["offline_shots"]-b["offline_shots"])/saving if valid else None,
                    first_strict_saving_query=int(np.floor((h["offline_shots"]-b["offline_shots"])/saving))+1 if valid else None,**result)
        records.append(record); print(json.dumps(record),flush=True)
    pd.DataFrame(records).to_csv(family/"confirmed_operating_points.csv",index=False)
    controls=json.loads((family/"validation/control_selection.json").read_text())
    control_rows=[]
    for choice,control in zip(lock["selections"],controls["controls"]):
        c=control["control"]; h=choice["hybrid"]
        if c is None or h is None: continue
        ha=tensor(rows,errors,row_map,h["method"],h["allocation"],h["shots"])
        ca=tensor(rows,errors,row_map,c["method"],c["allocation"],c["shots"])
        result=paired_bootstrap(ha,ca)
        control_rows.append(dict(family=a.family,target=choice["target"],control_method=c["method"],
            control_allocation=c["allocation"],control_shots=c["shots"],control_offline_shots=c["offline_shots"],
            neural_method=h["method"],neural_shots=h["shots"],neural_offline_shots=h["offline_shots"],
            control_target_confirmed=result["baseline_upper95"]<=choice["target"],**result))
    pd.DataFrame(control_rows).to_csv(family/"strong_control_comparisons.csv",index=False)
    secondary=[]
    for allocation in ["fixed","pilot"]+(["zz_known"] if a.family=="zz" else []):
        h=tensor(rows,errors,row_map,"neural_fusion",allocation,32)
        for other in ["dfe","summary_ridge_fusion"]:
            b=tensor(rows,errors,row_map,other,allocation,32)
            result=paired_bootstrap(h,b)
            secondary.append(dict(family=a.family,allocation=allocation,contrast="neural_fusion minus "+other,
                                  shots=32,status="prespecified secondary; individual intervals, no familywise claim",**result))
    pd.DataFrame(secondary).to_csv(family/"secondary_paired_comparisons.csv",index=False)
    # Retain the primary cost/accuracy data, not a fitted interpolation.
    val=pd.read_csv(family/"validation/summary.csv")
    val[val.method.isin(["dfe","neural_fusion","neural_eb","summary_ridge_fusion","summary_ridge_eb"])].to_csv(
        family/"validation_cost_accuracy.csv",index=False)
    (family/"analysis_manifest.json").write_text(json.dumps(dict(bootstrap=2000,
        selection_sha256=hashlib.sha256((family/"validation/selection.json").read_bytes()).hexdigest(),
        design_sha256=hashlib.sha256(Path("revision/stage2_design.json").read_bytes()).hexdigest(),
        statistical_scope="selected budgets fixed before test; two target comparisons per family; exchange is external replication",
        break_even_scope="deterministic quotient at fixed budgets, conditional on accuracy confirmation"),indent=2))


if __name__=="__main__": main()
