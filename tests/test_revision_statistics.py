import numpy as np
from scripts.analyze_revision_confirmation import paired_bootstrap
from scripts.analyze_revision_confirmation import tensor
import pandas as pd


def test_tensor_preserves_repeat_column_and_archived_row_mapping():
    rows=pd.DataFrame([dict(method="dfe",allocation="fixed",shots=32,cal_draw=d,repeat=r,checkpoint="none")
                       for d in range(20) for r in range(5)])
    errors=np.arange(300,dtype=np.float32).reshape(100,3)
    actual=tensor(rows,errors,{i:i for i in range(100)},"dfe","fixed",32)
    np.testing.assert_array_equal(actual,errors.reshape(20,5,1,3))


def test_shared_baseline_is_paired_across_repeated_checkpoints():
    base=np.random.default_rng(51).uniform(.01,.3,(20,5,1,37)).astype(np.float32)
    model=np.repeat(base,5,axis=2)
    result=paired_bootstrap(model,base,n_boot=128)
    assert abs(result["difference"])<1e-7
    assert abs(result["difference_ci95_low"])<1e-7
    assert abs(result["difference_ci95_high"])<1e-7


def test_calibration_query_resampling_preserves_known_contrast():
    base=np.random.default_rng(52).uniform(.02,.2,(20,5,1,37)).astype(np.float32)
    model=np.repeat(base,5,axis=2)-.01
    result=paired_bootstrap(model,base,n_boot=128)
    np.testing.assert_allclose([result["difference_ci95_low"],result["difference_ci95_high"]],-.01,atol=2e-7)
    assert result["noninferior_bonferroni_two_targets"]
    assert not result["equivalent_bonferroni_two_targets"]
