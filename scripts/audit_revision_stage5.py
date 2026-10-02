"""Read-only integrity checks for rerun legacy validation and selection locks."""
import json
from pathlib import Path
import pandas as pd
import numpy as np

for family in ['zz','exchange','two_bath']:
    root=Path('results_revision/stage5')/family
    new=pd.read_csv(root/'validation/summary.csv')
    old=pd.read_csv(Path('results_revision')/('stage4' if family=='two_bath' else 'stage2')/family/'validation/summary.csv')
    paired=new.merge(old,on=['method','allocation','shots'],suffixes=('_new','_old'))
    assert len(paired)>0
    # The older two-bath summaries reduce float32 per-query errors, whereas
    # stage five accumulates float64. Allow only that rounding-scale difference.
    np.testing.assert_allclose(paired.mae_new,paired.mae_old,rtol=0,atol=3e-8 if family=='two_bath' else 2e-10)
    print(family,'legacy validation reproduced',len(paired),'rows; max difference',abs(paired.mae_new-paired.mae_old).max())
    lock=json.loads((root/'selection.json').read_text())
    print([(r['target'],r['role'],r['configuration']) for r in lock['selections'] if r['target'] in [.065,.053]])
