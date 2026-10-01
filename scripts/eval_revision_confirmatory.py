"""Validation-selected, independent-test confirmation of measurement fusion."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
from physics.baselines.hybrid import batch_pauli_expectations_from_choi, fit_convex_fusion
from physics.baselines.revision_measurement import sample_measurement, residual_prior_variance, empirical_bayes_fusion
from scripts.eval_exact_composition import exact_predictions
from scripts.eval_label_budget_baselines import product_predictions,summary_features,fit_ridge,apply_ridge
from scripts.eval_recalibrated import predict_quantiles,fit_recalibrate


def digest(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def canonical(name): return re.sub(r"neural_bidir_seed[0-4]", "neural", name)


def prepare(path, ckpts, cache_path):
    expected={"data_sha256":digest(path),"checkpoint_sha256":{p:digest(p) for p in ckpts}}
    if cache_path.exists():
        cached=np.load(cache_path,allow_pickle=False)
        provenance=json.loads(str(cached["provenance"]))
        # Relocation of an otherwise identical released checkpoint must not
        # invalidate its cache. Compare contents, not machine-specific paths.
        if (provenance['data_sha256']!=expected['data_sha256'] or
            sorted(provenance['checkpoint_sha256'].values())!=sorted(expected['checkpoint_sha256'].values())):
            raise RuntimeError("stale feature cache")
        return {k:cached[k] for k in cached.files if k!="provenance"}
    raw=np.load(path,allow_pickle=True)
    exact=exact_predictions(raw); prod=product_predictions(raw)
    ex=batch_pauli_expectations_from_choi(raw["true_choi_real"]+1j*raw["true_choi_imag"])
    y=raw["y"].astype(float)
    if np.max(np.abs(ex.mean(1)-y))>5e-6: raise RuntimeError("target mismatch")
    payload=dict(y=y,ex=ex,features=summary_features(raw,prod,exact),length=raw["length"],
                 product_affine=prod,exact_marginal_affine=exact,x_fingerprints=np.array([
                     hashlib.sha256(x.tobytes()).hexdigest() for x in raw["x"]]))
    for p in ckpts:
        q,truth=predict_quantiles(p,str(path),device="cuda")
        np.testing.assert_allclose(truth,y,atol=1e-6)
        payload["neural_"+Path(p).stem]=q.mean(1)
    cache_path.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(cache_path,**payload,provenance=json.dumps(expected,sort_keys=True))
    return payload


def fit(name, calibration, evaluation, training, target, evaluate_idx):
    if name=="constant_mean": return np.full(len(evaluate_idx),np.mean(target))
    if name=="constant_median": return np.full(len(evaluate_idx),np.median(target))
    if name=="summary_ridge":
        return np.clip(apply_ridge(evaluation["features"][evaluate_idx],
                       fit_ridge(calibration["features"][training],target,1.)),0,1)
    slope,offset=fit_recalibrate(calibration[name][training],target)
    return np.clip(slope*evaluation[name][evaluate_idx]+offset,0,1)


def select(summary,root):
    selected=[]
    for target in [.065,.053]:
        options={}
        for role,methods in [("dfe",["dfe"]),("hybrid",["neural_fusion","neural_eb"])]:
            eligible=summary[(summary.method.isin(methods)) & (summary.selection_score<=target)]
            if eligible.empty:
                options[role]=None
            else:
                best=eligible.sort_values(["shots","offline_shots","selection_score","allocation"]).iloc[0]
                options[role]={k:(int(best[k]) if k in ["shots","offline_shots"] else str(best[k]))
                               for k in ["method","allocation","shots","offline_shots"]}
                options[role]["validation_mae"]=float(best.mae)
                options[role]["selection_score"]=float(best.selection_score)
        selected.append(dict(target=target,**options))
    payload=dict(status="locked before test",validation_summary_sha256=digest(root/"summary.csv"),
                 design_sha256=digest("revision/stage2_design.json"),selections=selected)
    (root/"selection.json").write_text(json.dumps(payload,indent=2))
    print(json.dumps(payload,indent=2),flush=True)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--family",choices=["zz","exchange"],required=True)
    ap.add_argument("--split",choices=["validation","test"],required=True)
    ap.add_argument("--ckpts",nargs="+",required=True)
    a=ap.parse_args(); design=json.loads(Path("revision/stage2_design.json").read_text())
    root=Path("results_revision/stage2")/a.family/a.split
    root.mkdir(parents=True,exist_ok=True)
    if (root/"summary.csv").exists(): raise FileExistsError("evaluation already completed")
    family_root=root.parent; data=Path("data_revision")/a.family
    lock=None; configurations=None
    allocations=["fixed","pilot"]+(["zz_known"] if a.family=="zz" else [])
    if a.split=="test":
        lock=json.loads((family_root/"validation/selection.json").read_text())
        if lock["design_sha256"]!=digest("revision/stage2_design.json"): raise RuntimeError("design changed after selection")
        if lock["validation_summary_sha256"]!=digest(family_root/"validation/summary.csv"): raise RuntimeError("validation changed")
        configurations=set()
        for s in lock["selections"]:
            for role in ["dfe","hybrid"]:
                if s[role]: configurations.add((s[role]["method"],s[role]["allocation"],s[role]["shots"]))
        control_lock=json.loads((family_root/"validation/control_selection.json").read_text())
        if control_lock["validation_summary_sha256"]!=lock["validation_summary_sha256"]:
            raise RuntimeError("control validation hash mismatch")
        for item in control_lock["controls"]:
            c=item["control"]
            if c: configurations.add((c["method"],c["allocation"],c["shots"]))
        # Prespecified secondary same-budget controls, no post-test selection.
        for alloc in allocations:
            for name in ["dfe","neural_fusion","summary_ridge_fusion","constant_mean_fusion",
                         "constant_median_fusion","product_affine_fusion","exact_marginal_affine_fusion",
                         "neural_eb","summary_ridge_eb","constant_mean_eb"]:
                configurations.add((name,alloc,32))
    calibration=prepare(data/"calibration.npz",a.ckpts,family_root/"cache_calibration.npz")
    evaluation=prepare(data/(a.split+".npz"),a.ckpts,family_root/("cache_"+a.split+".npz"))
    if set(calibration["x_fingerprints"]) & set(evaluation["x_fingerprints"]): raise RuntimeError("calibration/query input overlap")
    if a.split=="test":
        val=np.load(family_root/"cache_validation.npz",allow_pickle=False)
        if set(val["x_fingerprints"]) & set(evaluation["x_fingerprints"]): raise RuntimeError("validation/test input overlap")
    source_names=["product_affine","exact_marginal_affine"]+["neural_"+Path(p).stem for p in a.ckpts]
    names=["constant_mean","constant_median","summary_ridge",*source_names]
    n=len(evaluation["y"]); all_queries=np.arange(n)
    rows=[]; aggregates={}; stored=[]; stored_ids=[]
    def add(name,alloc,budget,draw,repeat,pred,cost):
        errors=np.abs(np.clip(pred,0,1)-evaluation["y"])
        method=canonical(name); key=(method,alloc,budget)
        rows.append(dict(method=method,checkpoint=(name if name.startswith("neural_") else "none"),
            allocation=alloc,shots=budget,cal_draw=draw,repeat=repeat,mae=float(errors.mean()),
            rmse=float(np.sqrt(np.mean(errors**2))),offline_shots=cost))
        if key not in aggregates: aggregates[key]=[np.zeros(n),0]
        aggregates[key][0]+=errors; aggregates[key][1]+=1
        if a.split=="test":
            stored.append(errors.astype(np.float32)); stored_ids.append(len(rows)-1)
    def wanted(method,alloc,b): return configurations is None or (method,alloc,b) in configurations
    for name in source_names: add(name+"_raw","none",0,-1,-1,evaluation[name],0)
    # Label streams are shared across validation and test, giving the same
    # fitted deployment rule; query measurement streams are independent.
    family_seed=202610101 if a.family=="zz" else 202610102
    for draw in range(20):
        cal=np.arange(64*draw,64*(draw+1))
        for repeat in range(5):
            labels,label_var,_=sample_measurement(calibration["ex"][cal],64,
                np.random.default_rng(np.random.SeedSequence([family_seed,draw,repeat,1])),
                "zz_known" if a.family=="zz" else "fixed")
            priors={}
            for name in names:
                oof=np.zeros(64)
                for fold in range(4):
                    held=np.arange(64)%4==fold
                    oof[held]=fit(name,calibration,calibration,cal[~held],labels[~held],cal[held])
                prior=fit(name,calibration,evaluation,cal,labels,all_queries)
                priors[name]=(prior,oof,residual_prior_variance(oof,labels,label_var))
                add(name+"_prior","none",0,draw,repeat,prior,4096)
            for ai,alloc in enumerate(allocations):
                for b in design["budgets"]:
                    needed=[m for m in ["dfe",*[canonical(name)+s for name in names for s in ["_fusion","_eb"]]] if wanted(m,alloc,b)]
                    if not needed: continue
                    crng=np.random.default_rng(np.random.SeedSequence([family_seed,draw,repeat,ai,b,2]))
                    trng=np.random.default_rng(np.random.SeedSequence([family_seed,draw,repeat,ai,b,3 if a.split=="validation" else 4]))
                    cal_obs,_,_=sample_measurement(calibration["ex"][cal],b,crng,alloc)
                    obs,var,counts=sample_measurement(evaluation["ex"],b,trng,alloc)
                    assert np.all(counts==b)
                    if wanted("dfe",alloc,b): add("dfe",alloc,b,draw,repeat,obs,0)
                    for name,(prior,oof,tau2) in priors.items():
                        if wanted(canonical(name)+"_fusion",alloc,b):
                            w=fit_convex_fusion(oof,cal_obs,labels)
                            add(name+"_fusion",alloc,b,draw,repeat,prior+w*(obs-prior),4096+64*b)
                        if wanted(canonical(name)+"_eb",alloc,b):
                            add(name+"_eb",alloc,b,draw,repeat,empirical_bayes_fusion(prior,obs,var,tau2),4096)
        print(a.family,a.split,"calibration group",draw+1,"/20",flush=True)
    df=pd.DataFrame(rows); df.to_csv(root/"runs.csv",index=False)
    summaries=[]
    for key,(error_sum,count) in aggregates.items():
        method,alloc,b=key
        sub=df[(df.method==method)&(df.allocation==alloc)&(df.shots==b)]
        qmean=error_sum/count
        draw_means=sub.groupby("cal_draw").mae.mean().values
        seed_means=sub.groupby("checkpoint").mae.mean().values
        variance=qmean.var(ddof=1)/n
        if len(draw_means)>1: variance+=draw_means.var(ddof=1)/len(draw_means)
        if len(seed_means)>1: variance+=seed_means.var(ddof=1)/len(seed_means)
        mean=float(qmean.mean())
        summaries.append(dict(method=method,allocation=alloc,shots=b,mae=mean,
            selection_score=mean+1.645*np.sqrt(variance),offline_shots=int(sub.offline_shots.max()),runs=count))
    summary=pd.DataFrame(summaries); summary.to_csv(root/"summary.csv",index=False)
    if a.split=="validation": select(summary,root)
    else:
        np.savez_compressed(root/"paired_errors.npz",absolute_errors=np.array(stored),row_ids=stored_ids,
                            lengths=evaluation["length"],truth=evaluation["y"])
    (root/"manifest.json").write_text(json.dumps(dict(family=a.family,split=a.split,
        data_sha256=digest(data/(a.split+".npz")),calibration_sha256=digest(data/"calibration.npz"),
        design_sha256=digest("revision/stage2_design.json"),selection=lock,
        settings="20 calibration groups x 5 measurement repeats x 5 fixed checkpoints",
        transfer="frozen ZZ-trained checkpoints; labelled target-domain recalibration",
        n=n,calibration_n=1280,query_calibration_input_overlap=0,
        control_selection=control_lock if a.split=="test" else None),indent=2))
    print("saved",root,flush=True)


if __name__=="__main__": main()
