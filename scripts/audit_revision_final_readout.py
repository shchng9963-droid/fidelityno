"""Secondary known-rate readout sensitivity of the frozen stage-five rules.

No training, validation, selection, or new confirmatory claims. Run on the
experimental server, with the released feature caches and stage-five locks.
"""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
from scripts.eval_revision_confirmatory import fit, canonical, digest
from physics.baselines.revision_measurement import (
    sample_measurement, residual_prior_variance, empirical_bayes_fusion,
    calibration_outcome_variances, independent_measurement_variance,
)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--family', choices=['zz', 'exchange'], required=True)
    a = ap.parse_args()
    root = Path('results_revision/final_readout_20261015') / a.family
    if root.exists():
        raise FileExistsError(root)
    design_path = Path('revision/final_readout_design_20261015.json')
    design = json.loads(design_path.read_text())
    stage = Path('results_revision/stage5') / a.family
    lock_path = stage / 'selection.json'
    lock = json.loads(lock_path.read_text())
    configs = {(c['method'], c['allocation'], c['shots'])
               for item in lock['selections'] if item['target'] in design['targets']
               if (c := item['configuration']) is not None}
    calpath = Path('results_revision/stage2') / a.family / 'cache_calibration.npz'
    testpath = stage / 'cache_test.npz'
    with np.load(calpath, allow_pickle=False) as z:
        cal = dict(z)
    with np.load(testpath, allow_pickle=False) as z:
        test = dict(z)
    cp, tp = (json.loads(str(z['provenance'])) for z in [cal, test])
    assert sorted(cp['checkpoint_sha256'].values()) == sorted(tp['checkpoint_sha256'].values())
    assert cp['data_sha256'] == digest(Path('data_revision') / a.family / 'calibration.npz')
    assert tp['data_sha256'] == digest(Path('data_revision/stage5') / a.family / 'test.npz')
    assert not set(cal['x_fingerprints']) & set(test['x_fingerprints'])
    names = ['constant_mean', 'summary_ridge'] + sorted(k for k in cal if k.startswith('neural_'))
    assert len(names) == 7
    assert len(cal['y']) == 1280 and len(test['y']) == 4096
    # At p=0, reproduce the primary calibration and query random streams.
    stage_design = json.loads(Path('revision/stage5_design.json').read_text())
    seed = {'zz': 202610101, 'exchange': 202610102}[a.family]
    primary_query_seed = stage_design['test_measurement_seeds'][a.family]
    allocations = ['fixed', 'pilot'] + (['zz_known'] if a.family == 'zz' else [])
    rows, moments = [], []
    allq = np.arange(len(test['y']))

    def add(name, alloc, budget, p, draw, rep, pred):
        error = np.abs(np.clip(pred, 0, 1) - test['y'])
        rows.append(dict(method=canonical(name), allocation=alloc, shots=budget,
                         readout_error=p, cal_draw=draw, repeat=rep,
                         checkpoint=name if name.startswith('neural_') else 'none',
                         mae=float(error.mean())))

    for pi, p in enumerate(design['flip_rates']):
        attenuation = 1 - 2 * p
        for draw in range(20):
            idx = np.arange(draw * 64, (draw + 1) * 64)
            for rep in range(5):
                cal_stream = [seed, draw, rep, 1] if p == 0 else [design['seed'], pi, seed, draw, rep, 1]
                labels, lv, count, detail = sample_measurement(
                    cal['ex'][idx], 64, np.random.default_rng(np.random.SeedSequence(cal_stream)),
                    'zz_known' if a.family == 'zz' else 'fixed', p, return_observations=True)
                assert np.all(count == 64)
                vcal = calibration_outcome_variances(detail)
                moments.append(dict(readout_error=p, cal_draw=draw, repeat=rep, variance=vcal.tolist()))
                priors = {}
                for name in names:
                    oof = np.zeros(64)
                    for fold in range(4):
                        held = np.arange(64) % 4 == fold
                        oof[held] = fit(name, cal, cal, idx[~held], labels[~held], idx[held])
                    prior = fit(name, cal, test, idx, labels, allq)
                    priors[name] = (prior, residual_prior_variance(oof, labels, lv))
                    add(name + '_prior', 'none', 0, p, draw, rep, prior)
                for alloc, budget in sorted({(c[1], c[2]) for c in configs}):
                    ai = allocations.index(alloc)
                    stream = ([primary_query_seed, draw, rep, ai, budget, 4] if p == 0 else
                              [design['seed'], pi, seed, draw, rep, ai, budget, 4])
                    obs, _, count, detail = sample_measurement(
                        test['ex'], budget, np.random.default_rng(np.random.SeedSequence(stream)),
                        alloc, p, return_observations=True)
                    assert np.all(count == budget)
                    settings = vcal if alloc == 'zz_known' or a.family != 'zz' else np.r_[vcal, 0.]
                    # The calibration variance is for attenuated +/-1 signs.
                    # Divide by attenuation**2 exactly once, after pooling.
                    variance = independent_measurement_variance(settings, detail['setting_shots'], attenuation)
                    if ('dfe', alloc, budget) in configs:
                        add('dfe', alloc, budget, p, draw, rep, obs)
                    for name, (prior, tau) in priors.items():
                        if (canonical(name) + '_iv', alloc, budget) in configs:
                            pred = empirical_bayes_fusion(prior, obs, variance, tau)
                            add(name + '_iv', alloc, budget, p, draw, rep, pred)
        print(a.family, 'completed known flip rate', p, flush=True)
    df = pd.DataFrame(rows)
    summary = df.groupby(['method', 'allocation', 'shots', 'readout_error']).mae.agg(
        ['mean', 'std', 'count']).reset_index()
    # Exact p=0 regression check against the released primary risk summaries.
    original = pd.read_csv(stage / 'test/summary.csv')
    for row in summary[summary.readout_error == 0].itertuples():
        match = original[(original.method == row.method) & (original.allocation == row.allocation)
                         & (original.shots == row.shots)]
        assert len(match) == 1
        np.testing.assert_allclose(row.mean, match.iloc[0].mae, atol=1e-12, rtol=0)
    root.mkdir(parents=True)
    df.to_csv(root / 'runs.csv', index=False)
    summary.to_csv(root / 'summary.csv', index=False)
    (root / 'calibration_moments.json').write_text(json.dumps(moments, indent=2))
    provenance = dict(
        status='secondary fixed-configuration sensitivity, no new confirmation or budget selection',
        family=a.family, design_sha256=digest(design_path), selection_sha256=digest(lock_path),
        calibration_cache_sha256=digest(calpath), test_cache_sha256=digest(testpath),
        checkpoint_sha256=tp['checkpoint_sha256'], clean_regression_check='all MAEs within 1e-12',
        code_sha256={str(p): digest(p) for p in [Path(__file__),
            Path('physics/baselines/revision_measurement.py'), Path('scripts/eval_revision_confirmatory.py')]},
        assumptions='known independent symmetric flips; same rate for calibration and queries; independent bath preparation',
        costs='4096 calibration shots per prior; rate estimation and input characterisation excluded',
        output_sha256={p.name: digest(p) for p in root.iterdir() if p.is_file()})
    (root / 'manifest.json').write_text(json.dumps(provenance, indent=2))
    print(summary.to_string(index=False), flush=True)


if __name__ == '__main__':
    main()
