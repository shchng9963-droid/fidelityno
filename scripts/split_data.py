"""Split train.npz into 90% train_split + 10% val_split, stratified by length."""
import numpy as np

print('Loading train.npz...')
d = np.load('data/train.npz', allow_pickle=True)
keys = list(d.keys())
N = len(d['y'])
print(f'Total samples: {N}, keys: {keys}')

# Stratified split by length
lengths = d['length']
rng = np.random.RandomState(42)
train_idx, val_idx = [], []
for l in np.unique(lengths):
    mask = np.where(lengths == l)[0]
    rng.shuffle(mask)
    split = int(0.9 * len(mask))
    train_idx.extend(mask[:split].tolist())
    val_idx.extend(mask[split:].tolist())

train_idx = np.array(train_idx)
val_idx = np.array(val_idx)
print(f'Train: {len(train_idx)}, Val: {len(val_idx)}')

# Only index per-sample arrays (those with first dim == N)
sample_keys = [k for k in keys if d[k].shape[0] == N]
meta_keys = [k for k in keys if d[k].shape[0] != N]
print(f'Sample keys: {sample_keys}')
print(f'Meta keys: {meta_keys}')

# Save
train_d = {k: d[k][train_idx] for k in sample_keys}
train_d.update({k: d[k] for k in meta_keys})
val_d = {k: d[k][val_idx] for k in sample_keys}
val_d.update({k: d[k] for k in meta_keys})
np.savez('data/train_split.npz', **train_d)
np.savez('data/val_split.npz', **val_d)
print('Saved.')

# Verify
for name in ['train_split', 'val_split']:
    dd = np.load(f'data/{name}.npz', allow_pickle=True)
    lens = dd['length']
    unique, counts = np.unique(lens, return_counts=True)
    print(f'{name}: {len(dd["y"])} samples, lens={dict(zip(unique.tolist(), counts.tolist()))}')
