
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from tqdm import tqdm
from physics.channels.single_qubit import sample_single_qubit
from physics.channels.two_qubit import sample_two_qubit, sample_order_sensitive_two_qubit_sequence
from physics.channels.lindblad import sample_lindblad
from physics.channels.device_regime import (
    sample_single_qubit_device,
    sample_two_qubit_device,
    regime_summary as device_regime_summary,
)
from physics.composition import exact_sequence_fidelity, sequence_features, composed_stats, channel_reference_fidelity
from physics.fidelity import FIDELITY_KIND, fidelity_formula

FAMILIES_1Q=["amplitude_damping","phase_damping","depolarizing","pauli","lindblad"]
FAMILIES_2Q=["correlated_dephasing","two_qubit_depolarizing","imperfect_cnot","imperfect_swap"]
ORDER_SENSITIVE_TAG="order_sensitive"

def _active_families(dim:int, exclude_family: str | None = None):
    fams = list(FAMILIES_1Q if dim == 2 else FAMILIES_2Q)
    if exclude_family:
        fams = [f for f in fams if f != exclude_family]
    if not fams:
        raise ValueError(f"no active families for dim={dim}, exclude={exclude_family}")
    return fams


def _order_sensitive_active_families(exclude_family: str | None = None):
    fams = list(FAMILIES_2Q)
    if exclude_family:
        fams = [f for f in fams if f != exclude_family]
    if not fams:
        raise ValueError(f"no active order-sensitive families remain after excluding {exclude_family}")
    return fams


def _sample_single_channel(rng, split_family="mixed", dim=2, exclude_family: str | None = None, regime: str = "broad"):
    if regime == "device":
        if dim == 2:
            if split_family == "mixed":
                fam = rng.choice(_active_families(dim, exclude_family))
            else:
                fam = split_family
            return sample_single_qubit_device(rng, fam)
        if split_family == "mixed":
            fam = rng.choice(_active_families(dim, exclude_family))
        else:
            fam = split_family
        return sample_two_qubit_device(rng, fam)

    if dim==2:
        if split_family=="mixed":
            fam=rng.choice(_active_families(dim, exclude_family))
            return sample_lindblad(rng) if fam=="lindblad" else sample_single_qubit(rng,fam)
        if split_family=="lindblad":
            return sample_lindblad(rng)
        return sample_single_qubit(rng, split_family)

    if split_family=="mixed":
        return sample_two_qubit(rng, rng.choice(_active_families(dim, exclude_family)))
    return sample_two_qubit(rng, split_family)


def sample_sequence(rng, split_family="mixed", dim=2, length: int | None = None, exclude_family: str | None = None, required_family: str | None = None, regime: str = "broad"):
    if split_family == ORDER_SENSITIVE_TAG:
        if dim != 4:
            raise ValueError("order_sensitive benchmark currently requires dim=4 (two-qubit channels)")
        if length is None:
            raise ValueError("order_sensitive sequence sampling requires explicit length")
        if length < 8:
            raise ValueError("order_sensitive benchmark requires sequence length >= 8")
        return sample_order_sensitive_two_qubit_sequence(
            rng,
            length,
            exclude_family=exclude_family,
            required_family=required_family,
        )

    if length is None:
        raise ValueError("sample_sequence requires explicit length")
    return [_sample_single_channel(rng, split_family, dim, exclude_family=exclude_family, regime=regime) for _ in range(length)], None

def _family_index(name: str, dim: int) -> int | None:
    fams = FAMILIES_1Q if dim == 2 else FAMILIES_2Q
    if dim == 2:
        if name.startswith("amplitude_damping"): key="amplitude_damping"
        elif name.startswith("phase_damping"): key="phase_damping"
        elif name.startswith("depolarizing"): key="depolarizing"
        elif name.startswith("pauli"): key="pauli"
        elif name.startswith("lindblad"): key="lindblad"
        else: return None
    else:
        if "cnot" in name: key="imperfect_cnot"
        elif "swap" in name: key="imperfect_swap"
        elif "correlated_dephasing" in name: key="correlated_dephasing"
        elif "two_qubit_depolarizing" in name: key="two_qubit_depolarizing"
        else: return None
    return fams.index(key)

def build_dataset(out:Path,n:int,seed:int,lengths:list[int],family:str,dim:int,max_len:int,exclude_family: str | None = None,representation: str = "choi_hermitian",required_family: str | None = None,regime: str = "broad"):
    from physics.representations import feature_dim_for_representation
    rng=np.random.default_rng(seed); feat_dim=feature_dim_for_representation(dim, representation); fams=FAMILIES_1Q if dim==2 else FAMILIES_2Q
    X=np.zeros((n,max_len,feat_dim),np.float32); mask=np.zeros((n,max_len),np.float32); y=np.zeros(n,np.float32); stats=np.zeros((n,2),np.float32); lens=np.zeros(n,np.int32); per_fid=np.ones((n,max_len),np.float32); fam_id=np.empty(n, dtype=object); family_counts=np.zeros((n,len(fams)),np.int32); family_idx_seq=np.full((n,max_len), -1, np.int16)
    perm_gap_random=np.zeros(n,np.float32); perm_gap_reverse=np.zeros(n,np.float32); fidelity_random_perm=np.zeros(n,np.float32); fidelity_reverse=np.zeros(n,np.float32)
    benchmark_tag = ORDER_SENSITIVE_TAG if family == ORDER_SENSITIVE_TAG else family
    for idx in tqdm(range(n),desc=f"gen {out.name}"):
        L=int(rng.choice(lengths)); seq, seq_meta = sample_sequence(rng,family,dim,length=L,exclude_family=exclude_family,required_family=required_family,regime=regime)
        X[idx],mask[idx]=sequence_features(seq,max_len,dim,representation); y[idx]=exact_sequence_fidelity(seq); st=composed_stats(seq); stats[idx]=[st['trace'],st['purity']]; lens[idx]=L
        per_fid[idx,:L]=[channel_reference_fidelity(ch) for ch in seq]; fam_id[idx]=','.join(ch.name for ch in seq[:min(3,L)])
        if seq_meta is not None:
            perm_gap_random[idx] = seq_meta['perm_gap_random']
            perm_gap_reverse[idx] = seq_meta['perm_gap_reverse']
            fidelity_random_perm[idx] = seq_meta['fidelity_random_perm']
            fidelity_reverse[idx] = seq_meta['fidelity_reverse']
        for pos, ch in enumerate(seq[:max_len]):
            j=_family_index(ch.name, dim)
            if j is not None:
                family_counts[idx,j]+=1
                family_idx_seq[idx,pos]=j
    if family == ORDER_SENSITIVE_TAG:
        active_families = _order_sensitive_active_families(exclude_family)
    elif family == "mixed":
        active_families = _active_families(dim, exclude_family)
    else:
        active_families = [family]
    np.savez_compressed(out,x=X,mask=mask,y=y,stats=stats,length=lens,per_fid=per_fid,family_prefix=fam_id,family_counts=family_counts,family_idx_seq=family_idx_seq,family_names=np.array(fams,dtype=object),perm_gap_random=perm_gap_random,perm_gap_reverse=perm_gap_reverse,fidelity_random_perm=fidelity_random_perm,fidelity_reverse=fidelity_reverse)
    return {"file":str(out),"n":n,"seed":seed,"lengths":lengths,"family":family,"exclude_family":exclude_family,"required_family":required_family,"dim":dim,"max_len":max_len,"representation":representation,"active_families":active_families,"y_mean":float(y.mean()),"y_std":float(y.std()),"benchmark_tag":benchmark_tag,"perm_gap_random_mean":float(perm_gap_random.mean()),"perm_gap_reverse_mean":float(perm_gap_reverse.mean()),"fidelity_kind":FIDELITY_KIND,"fidelity_formula":fidelity_formula(),"regime":regime,"y_description":"Per-sequence entanglement fidelity F_e against the natural reference target (identity for noise channels; ideal unitary for imperfect gates). Use ef_to_avg(F_e, dim) from physics.fidelity for average gate fidelity."}

def default_holdout(family: str, dim: int) -> str:
    if family != "mixed": return family
    return "pauli" if dim == 2 else "correlated_dephasing"

def _parse_lengths(text: str) -> list[int]:
    vals = [int(x) for x in text.split(',') if x.strip()]
    if not vals:
        raise ValueError("length list must be non-empty")
    return vals


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--outdir',default='data'); ap.add_argument('--n-train',type=int,default=1000); ap.add_argument('--n-test',type=int,default=300); ap.add_argument('--n-calib',type=int,default=0,help='Optional explicit calibration split size. If >0, writes calib.npz sampled from ID lengths but disjoint RNG seed.'); ap.add_argument('--seed',type=int,default=0); ap.add_argument('--dim',type=int,default=2); ap.add_argument('--family',default='mixed'); ap.add_argument('--holdout-family',default=None); ap.add_argument('--max-len',type=int,default=48); ap.add_argument('--representation',default='choi_hermitian',choices=['choi_hermitian','raw_choi_flat','compressed_hermitian','ptm']); ap.add_argument('--train-lengths',default='2,4,8,16'); ap.add_argument('--id-lengths',default='2,4,8,16'); ap.add_argument('--length-ood-lengths',default='24,32,48'); ap.add_argument('--family-ood-lengths',default='8,16,24'); ap.add_argument('--regime',default='broad',choices=['broad','device'],help='broad = v1 NISQ-tail sampling ranges; device = narrow ranges around real QPU calibration (PRXQ P0.1b).')
    args=ap.parse_args(); outdir=Path(args.outdir); outdir.mkdir(parents=True,exist_ok=True)
    holdout=args.holdout_family or default_holdout(args.family,args.dim)
    if args.family == ORDER_SENSITIVE_TAG:
        if args.regime == 'device':
            raise ValueError("device regime is not yet wired through the order_sensitive sampler.")
        exclude = holdout
        family_ood_family = ORDER_SENSITIVE_TAG
        family_ood_required = holdout
    else:
        exclude=holdout if args.family=="mixed" else None
        family_ood_family = holdout
        family_ood_required = None
    train_lengths = _parse_lengths(args.train_lengths)
    id_lengths = _parse_lengths(args.id_lengths)
    length_ood_lengths = _parse_lengths(args.length_ood_lengths)
    family_ood_lengths = _parse_lengths(args.family_ood_lengths)
    regime = args.regime
    manifest=[]
    manifest.append(build_dataset(outdir/'train.npz',args.n_train,args.seed,train_lengths,args.family,args.dim,args.max_len,exclude_family=exclude,representation=args.representation,regime=regime))
    if args.n_calib > 0:
        manifest.append(build_dataset(outdir/'calib.npz',args.n_calib,args.seed+4,id_lengths,args.family,args.dim,args.max_len,exclude_family=exclude,representation=args.representation,regime=regime))
    manifest.append(build_dataset(outdir/'id_test.npz',args.n_test,args.seed+1,id_lengths,args.family,args.dim,args.max_len,exclude_family=exclude,representation=args.representation,regime=regime))
    manifest.append(build_dataset(outdir/'length_ood.npz',args.n_test,args.seed+2,length_ood_lengths,args.family,args.dim,args.max_len,exclude_family=exclude,representation=args.representation,regime=regime))
    manifest.append(build_dataset(outdir/'family_ood.npz',args.n_test,args.seed+3,family_ood_lengths,family_ood_family,args.dim,args.max_len,exclude_family=None,representation=args.representation,required_family=family_ood_required,regime=regime))
    top = {"holdout_family":holdout,"representation":args.representation,"fidelity_kind":FIDELITY_KIND,"fidelity_formula":fidelity_formula(),"regime":regime,"splits":manifest}
    if regime == 'device':
        top["regime_summary"] = device_regime_summary()
    (outdir/'manifest.json').write_text(json.dumps(top,indent=2))
if __name__=='__main__': main()
