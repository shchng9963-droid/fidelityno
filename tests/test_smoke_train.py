
import subprocess, sys, os
from pathlib import Path

def test_end_to_end_training_on_100_sample_subset(tmp_path):
    root=Path('/home/wangshuchang/fidelityno')
    env=os.environ.copy(); env['WANDB_MODE']='offline'
    py='/home/wangshuchang/miniforge3/envs/fidelityno/bin/python'
    data_dir=tmp_path/'data'
    ckpt_dir=tmp_path/'checkpoints'
    subprocess.run([py,'scripts/gen_data.py','--outdir',str(data_dir),'--n-train','100','--n-test','40','--seed','123'],cwd=root,env=env,check=True,timeout=120)
    subprocess.run([py,'train.py',f'data.train={data_dir}/train.npz',f'data.val={data_dir}/id_test.npz',f'train.ckpt_dir={ckpt_dir}','train.epochs=1','train.batch_size=32','model.d_model=32','model.depth=1','model.heads=4','device=cpu'],cwd=root,env=env,check=True,timeout=180)
    assert (ckpt_dir/'fidelityno_seed0.pt').exists()
