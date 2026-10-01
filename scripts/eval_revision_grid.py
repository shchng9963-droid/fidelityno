"""Nested empirical spans and distribution-specific risks on fixed inputs."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
from physics.channels.batched_memory import memory_grid_fidelity
from physics.channels.collision_nonmarkov import collision_sequence, _collision_unitary, PLUS
from physics.channels.collision_exchange_memory import exchange_collision_sequence, _exchange_unitary, DEFAULT_BATH
from physics.composition import exact_sequence_fidelity


def weighted_risk(values, weights):
    order = np.argsort(values, axis=1)
    ordered = np.take_along_axis(values, order, axis=1)
    weights = weights/weights.sum()
    w = np.broadcast_to(weights, values.shape)
    ordered_w = np.take_along_axis(w, order, axis=1)
    median = ordered[np.arange(len(values)), (ordered_w.cumsum(axis=1)>=.5).argmax(axis=1)]
    return (np.abs(values-median[:,None])*weights).sum(axis=1)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--family", choices=["zz","exchange"], required=True)
    ap.add_argument("--n-base", type=int, required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out=Path(args.out); out.mkdir(parents=True,exist_ok=True)
    rng=np.random.default_rng(args.seed)
    grid=np.linspace(.85,.99,113)
    # A separate 128-point Gauss rule approximates the continuous-uniform
    # distribution. It is not labelled an exact continuous Bayes risk.
    nodes, quadrature=np.polynomial.legendre.leggauss(128)
    continuous=.92+.07*nodes
    values=np.empty((args.n_base,113)); continuous_values=np.empty((args.n_base,128))
    lengths=[]; all_params=[]; zero_delta=0.
    generator=collision_sequence if args.family=="zz" else exchange_collision_sequence
    unitary=_collision_unitary if args.family=="zz" else _exchange_unitary
    bath=PLUS if args.family=="zz" else DEFAULT_BATH
    for i in range(args.n_base):
        length=int(rng.choice([8,16,24])); lengths.append(length)
        base=generator(length,eta=0.,rng=rng)
        us=[unitary(*p) for p in base.params]
        combined=memory_grid_fidelity(us,np.r_[grid,continuous,0.],bath)
        values[i]=combined[:113]; continuous_values[i]=combined[113:-1]
        zero_delta=max(zero_delta,abs(combined[-1]-exact_sequence_fidelity(base.marginals)))
        padded=np.full((24,base.params.shape[1]),np.nan); padded[:length]=base.params; all_params.append(padded)
        if (i+1)%256==0: print(args.family,i+1,"/",args.n_base,flush=True)
    if zero_delta>1e-10: raise RuntimeError("eta=0 cross-check failed")
    rows=[]; prior_span=np.zeros(args.n_base)
    for n in [15,29,57,113]:
        v=values[:,::112//(n-1)]
        span=np.ptp(v,axis=1)
        if np.any(span+1e-12<prior_span): raise RuntimeError("non-nested span")
        increase=span-prior_span
        for i in range(args.n_base):
            rows.append(dict(family=args.family,base=i,length=lengths[i],grid_n=n,empirical_span=span[i],
                worst_case_lower_bound=span[i]/2,uniform_discrete_bayes_mae=np.abs(v[i]-np.median(v[i])).mean(),
                span_increase_from_previous=increase[i] if n>15 else 0.))
        prior_span=span
    df=pd.DataFrame(rows); df.to_csv(out/"per_base.csv",index=False)
    summary=df.groupby(["family","grid_n"]).agg(n_base=("base","size"),
        empirical_span=("empirical_span","mean"),worst_case_lower_bound=("worst_case_lower_bound","mean"),
        discrete_uniform_bayes_mae=("uniform_discrete_bayes_mae","mean"),
        max_span_increase=("span_increase_from_previous","max")).reset_index()
    summary.to_csv(out/"summary.csv",index=False)
    df.groupby(["family","grid_n","length"]).mean(numeric_only=True).to_csv(out/"by_length.csv")
    uniform=weighted_risk(continuous_values,quadrature)
    # Beta(2,1) density on the rescaled retention interval, approximated by
    # the same quadrature with a stated changed prior.
    high_weight=quadrature*(nodes+1)
    tilted=weighted_risk(continuous_values,high_weight)
    pd.DataFrame(dict(length=lengths,uniform_continuous_quadrature_risk=uniform,
                      beta_2_1_quadrature_risk=tilted)).to_csv(out/"distribution_sensitivity.csv",index=False)
    np.savez_compressed(out/"audit.npz",eta_grid=grid,fidelity_grid=values,lengths=lengths,parameters=all_params,
                        quadrature_eta=continuous,quadrature_weights=quadrature,quadrature_fidelity=continuous_values)
    (out/"manifest.json").write_text(json.dumps(dict(args=vars(args),max_eta_zero_delta=zero_delta,
        continuous_risk_is_quadrature_approximation=True,
        interpretation="nested-grid empirical spans are lower bounds, not certified continuous suprema"),indent=2))
    print(summary.to_string(index=False),flush=True)
    print("quadrature risks",uniform.mean(),tilted.mean(),flush=True)


if __name__=="__main__": main()
