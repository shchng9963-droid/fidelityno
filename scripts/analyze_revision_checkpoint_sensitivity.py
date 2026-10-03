"""Exploratory checkpoint sensitivity of the existing, locked stage-five test.

No new observations, budget selection, model fitting or primary-result writes.
The four-checkpoint resampling intervals are diagnostics, not confirmation.
"""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.analyze_revision_confirmation import tensor

ROOT = Path('results_revision/stage5/exchange')
OUT = Path('results_revision/closeout_20261003')
BOOT_SEED = 202610301
NBOOT = 4000


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def resampling_weights(nq, seed=BOOT_SEED, nboot=NBOOT):
    rng = np.random.default_rng(seed)
    w = np.zeros((nboot, 20, 5), dtype=np.float32)
    for i in range(nboot):
        draws = rng.integers(20, size=20)
        repeats = rng.integers(5, size=(20, 5))
        np.add.at(w[i], (np.repeat(draws, 5), repeats.ravel()), 1 / 100)
    sw = rng.multinomial(4, np.full(4, .25), size=nboot).astype(np.float32) / 4
    qw = rng.multinomial(nq, np.full(nq, 1 / nq), size=nboot).astype(np.float32) / nq
    return w.reshape(nboot, 100), np.einsum('bdr,bs->bdrs', w, sw).reshape(nboot, 400), qw


def bootstrap(x, w, qw):
    flat = x.reshape(-1, x.shape[-1])
    assert flat.shape[0] == w.shape[1]
    return np.concatenate([((w[i:i + 64] @ flat) * qw[i:i + 64]).sum(1)
                           for i in range(0, len(w), 64)])


def main():
    sources = [ROOT / 'test/runs.csv', ROOT / 'test/paired_errors.npz',
               ROOT / 'selection.json', ROOT / 'risk_intervals.csv',
               ROOT / 'paired_contrasts.csv', Path('revision/stage5_design.json')]
    before = {str(p): digest(p) for p in sources}
    rows = pd.read_csv(ROOT / 'test/runs.csv')
    archive = np.load(ROOT / 'test/paired_errors.npz')
    err = archive['absolute_errors']
    rowmap = {int(r): i for i, r in enumerate(archive['row_ids'])}
    lock = json.loads((ROOT / 'selection.json').read_text())
    risks = pd.read_csv(ROOT / 'risk_intervals.csv')

    def get(c):
        return tensor(rows, err, rowmap, c['method'], c['allocation'], c['shots'])

    choices = {(c['target'], c['role']): c['configuration'] for c in lock['selections']}
    configs = [('prior', dict(method='neural_prior', allocation='none', shots=0)),
               ('target_0065', choices[.065, 'neural']),
               ('target_0053', choices[.053, 'neural'])]
    checkpoints = []
    for label, c in configs:
        x = get(c)
        names = sorted(rows[(rows.method == c['method']) &
                            (rows.allocation == c['allocation']) &
                            (rows.shots == c['shots'])].checkpoint.unique())
        assert x.shape == (20, 5, 5, 4096)
        averages = x.mean(axis=(0, 1, 3), dtype=np.float64)
        original = risks[(risks.method == c['method']) &
                         (risks.allocation == c['allocation']) &
                         (risks.shots == c['shots'])].iloc[0]
        assert abs(float(averages.mean()) - original.mae) < 3e-8
        for s, (name, value) in enumerate(zip(names, averages)):
            checkpoints.append(dict(configuration=label, checkpoint_index=s,
                                    checkpoint=str(name), shots=c['shots'],
                                    allocation=c['allocation'], mae=float(value)))

    w, ws, qw = resampling_weights(err.shape[1])
    sensitivity = []
    for target in [.065, .053]:
        neural = get(choices[target, 'neural'])
        bases = {role: get(choices[target, role]) for role in ['dfe', 'constant', 'ridge']}
        base_boot = {role: bootstrap(x, w, qw) for role, x in bases.items()}
        for omitted in range(5):
            kept = [s for s in range(5) if s != omitted]
            x = neural[:, :, kept, :]
            mean = float(x.mean(dtype=np.float64))
            expected = (neural.mean(axis=(0, 1, 3), dtype=np.float64)[kept]).mean()
            assert abs(mean - expected) < 1e-12
            boot = bootstrap(x, ws, qw)
            contrasts = {}
            for role, base in bases.items():
                delta = boot - base_boot[role]
                contrasts[role] = dict(difference=mean - float(base.mean(dtype=np.float64)),
                    low=float(np.quantile(delta, .025)), high=float(np.quantile(delta, .975)),
                    upper_matching_primary_quantile=float(np.quantile(delta, 1 - .05 / 6)))
            sensitivity.append(dict(target=target, omitted_checkpoint_index=omitted,
                neural_shots=choices[target, 'neural']['shots'], mae=mean,
                upper95=float(np.quantile(boot, .95)), contrasts=contrasts))
    assert before == {str(p): digest(p) for p in sources}, 'Primary inputs changed'
    OUT.mkdir(parents=True, exist_ok=True)
    result = dict(status='Exploratory analysis after viewing the final primary test',
        family='exchange', bootstrap_seed=BOOT_SEED, bootstrap_draws=NBOOT,
        protocol='Keep final budgets, priors, predictions, calibration and query outcomes fixed. '
                 'Delete each neural checkpoint in turn; resample the four retained checkpoints, '
                 '20 calibration groups, five nested repeats and 4096 queries. '
                 'Comparators are unchanged. Reuse primary 1-0.05/6 quantile for sensitivity only; '
                 'no multiplicity guarantee across omissions, no new confirmatory claims.',
        input_sha256=before, script_sha256=digest(__file__),
        checkpoint_mae=checkpoints, leave_one_out=sensitivity)
    (OUT / 'exchange_checkpoint_sensitivity.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(dict(checkpoint_mae=checkpoints, leave_one_out=sensitivity), indent=2), flush=True)


if __name__ == '__main__':
    main()
