
from __future__ import annotations
import os, random, math
os.environ.setdefault('WANDB_MODE','offline')
import hydra, numpy as np, torch, wandb
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader, TensorDataset, Subset
from models.fidelityno import FidelityNO
from models.baselines.mlp import FlatMLP
from models.baselines.deepsets import DeepSets
from models.baselines.gnn import LinearChainGNN
from models.baselines.generic_gnn import GenericPathGNN
from models.baselines.bidir import BidirectionalTransformer
from models.heads.quantile import pinball_loss
from models.heads.gaussian import truncated_normal_nll



def prediction_to_quantiles(pred, levels):
    if isinstance(pred, tuple):
        mu, sigma = pred
        q = torch.distributions.Normal(mu.unsqueeze(-1), sigma.unsqueeze(-1)).icdf(levels.to(mu.device).view(1, -1))
        return q.clamp(0.0, 1.0)
    if pred.ndim == 2 and pred.shape[-1] == 1:
        return pred.expand(-1, levels.numel()).clamp(0.0, 1.0)
    return pred


def mean_from_prediction(pred):
    if isinstance(pred, tuple):
        return pred[0]
    if pred.ndim == 2 and pred.shape[-1] == 1:
        return pred.squeeze(-1)
    return pred.mean(-1)


def loss_for_prediction(pred, target, levels, head_type: str):
    if head_type == 'scalar':
        return torch.nn.functional.mse_loss(mean_from_prediction(pred), target)
    if head_type == 'gaussian':
        mu, sigma = pred
        return truncated_normal_nll(mu, sigma, target)
    return pinball_loss(pred, target, levels)

def seed_all(seed): random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def load_npz(path):
    d=np.load(path, allow_pickle=True)
    return TensorDataset(
        torch.tensor(d['x']).float(),
        torch.tensor(d['mask']).float(),
        torch.tensor(d['y']).float(),
        torch.tensor(d['stats']).float(),
    )


def get_seq_lengths(dataset):
    """Return per-sample sequence length from the mask tensor."""
    masks = dataset.tensors[1]  # [N, max_len]
    return masks.sum(dim=1).long()  # [N]


def curriculum_subset(dataset, max_len, seq_lengths):
    """Filter dataset to samples with seq_length <= max_len."""
    indices = (seq_lengths <= max_len).nonzero(as_tuple=True)[0]
    if len(indices) == len(dataset):
        return dataset
    return Subset(dataset, indices.tolist())


def make_model(name, input_dim, max_len, cfg):
    d = cfg.model.get('d_model', 256)
    layers = cfg.model.get('layers', 4)
    if name.startswith('fidelityno_gru'):
        from models.fidelityno_gru import FidelityNO_GRU
        return FidelityNO_GRU(input_dim, d, cfg.model.get('depth', 4),
                              cfg.model.get('head_type','quantile'),
                              bidir=cfg.model.get('bidir', False),
                              aux=cfg.model.get('aux', True))
    if name.startswith('generic_gnn'):
        return GenericPathGNN(input_dim, d, layers)
    if name.startswith('gnn') or name.startswith('fidelityno_gnn'):
        return LinearChainGNN(input_dim, d, layers)
    if name.startswith('fidelityno'):
        return FidelityNO(input_dim, d, cfg.model.get('depth', 4), cfg.model.get('heads', 4),
                          cfg.model.get('head_type','quantile'), cfg.model.get('causal',True), cfg.model.get('aux',True))
    if name.startswith('mlp'): return FlatMLP(input_dim, max_len, d)
    if name.startswith('deepsets'): return DeepSets(input_dim, d)
    if name.startswith('bidir'): return BidirectionalTransformer(input_dim, d, cfg.model.get('depth', 4), cfg.model.get('heads', 4), 'quantile')
    raise ValueError(name)


def get_lr_with_warmup(step, total_steps, warmup_steps, lr_max, lr_min, restart_period=None):
    """Linear warmup then cosine decay with warm restarts.
    
    If restart_period is set, use cosine annealing with warm restarts (SGDR).
    Each restart gives the optimizer a fresh chance to escape local minima.
    This is critical for tasks with phase transitions in training.
    """
    if step < warmup_steps:
        return lr_min + (lr_max - lr_min) * step / max(1, warmup_steps)
    
    post_warmup = step - warmup_steps
    remaining = total_steps - warmup_steps
    
    if restart_period is not None and restart_period > 0:
        # SGDR: cosine with warm restarts
        cycle_pos = post_warmup % restart_period
        progress = cycle_pos / restart_period
    else:
        # Standard cosine
        progress = post_warmup / max(1, remaining)
    
    return lr_min + 0.5 * (lr_max - lr_min) * (1 + math.cos(math.pi * progress))


@hydra.main(version_base=None, config_path='configs', config_name='base')
def main(cfg: DictConfig):
    seed_all(cfg.seed)
    device = 'cuda' if torch.cuda.is_available() and cfg.device == 'cuda' else 'cpu'

    tr = load_npz(cfg.data.train)
    va = load_npz(cfg.data.val)
    input_dim = tr.tensors[0].shape[-1]
    max_len = tr.tensors[0].shape[1]

    model = make_model(cfg.model.name, input_dim, max_len, cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {cfg.model.name}  Params: {n_params:,d}")

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)

    epochs = cfg.train.epochs
    batch_size = cfg.train.batch_size
    warmup_epochs = cfg.train.get('warmup_epochs', 5)
    lr_max = cfg.train.lr
    lr_min = cfg.train.get('lr_min', 1e-6)

    wandb.init(project='fidelityno', mode=cfg.wandb.mode,
               config=OmegaConf.to_container(cfg, resolve=True),
               name=f"{cfg.model.name}-seed{cfg.seed}")
    wandb.log({'n_params': n_params})

    levels = torch.tensor(cfg.model.quantiles, dtype=torch.float32, device=device)
    best = 1e9
    ckpt_dir = str(cfg.train.get('ckpt_dir', 'checkpoints'))
    os.makedirs(ckpt_dir, exist_ok=True)

    head_type = cfg.model.get('head_type', 'quantile')
    aux_weight = cfg.train.aux_weight
    grad_clip = cfg.train.get('grad_clip', 1.0)

    # Curriculum config
    curriculum_enabled = cfg.train.get('curriculum', False)  # Default OFF now
    curriculum_schedule = cfg.train.get('curriculum_schedule', [4, 8, 16, 48])

    # Pre-compute seq lengths for curriculum
    tr_seq_lengths = get_seq_lengths(tr)

    # Total training steps for LR schedule
    steps_per_epoch = max(1, len(tr) // batch_size)
    total_steps = epochs * steps_per_epoch
    warmup_steps = warmup_epochs * steps_per_epoch
    # Restart period: every 20 epochs = gives 4 chances to find phase transition in 80 epochs
    restart_epochs = cfg.train.get('restart_epochs', 20)
    restart_period = restart_epochs * steps_per_epoch if restart_epochs > 0 else None
    global_step = 0

    # Patience for early stopping
    patience = cfg.train.get('patience', 20)
    no_improve = 0

    for ep in range(epochs):
        # Curriculum: determine current max sequence length
        if curriculum_enabled:
            # Smooth curriculum: linearly increase max_len over first 60% of training
            progress = min(1.0, ep / (0.6 * epochs))
            phase_idx = min(int(progress * len(curriculum_schedule)), len(curriculum_schedule) - 1)
            cur_max_len = curriculum_schedule[phase_idx]
            train_data = curriculum_subset(tr, cur_max_len, tr_seq_lengths)
        else:
            cur_max_len = max_len
            train_data = tr

        model.train()
        losses = []
        for x, m, y, stats in DataLoader(train_data, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=4, pin_memory=True):
            x, m, y, stats = x.to(device), m.to(device), y.to(device), stats.to(device)

            # Update LR with warmup + cosine with warm restarts
            lr_now = get_lr_with_warmup(global_step, total_steps, warmup_steps, lr_max, lr_min, restart_period)
            for pg in opt.param_groups:
                pg['lr'] = lr_now

            opt.zero_grad()
            pred, aux = model(x, m)
            loss = loss_for_prediction(pred, y, levels, head_type)
            if aux is not None:
                loss = loss + aux_weight * torch.nn.functional.mse_loss(aux, stats)
            loss.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            opt.step()
            losses.append(loss.item())
            global_step += 1

        # Validation
        model.eval()
        maes = []
        vloss = []
        with torch.no_grad():
            for x, m, y, stats in DataLoader(va, batch_size=512, num_workers=4, pin_memory=True):
                x, m, y = x.to(device), m.to(device), y.to(device)
                pred, _ = model(x, m)
                mean = mean_from_prediction(pred)
                maes.append((mean - y).abs().mean().item())
                vloss.append(loss_for_prediction(pred, y, levels, head_type).item())

        vm = float(np.mean(maes))
        vl = float(np.mean(vloss))
        tl = float(np.mean(losses))

        wandb.log({
            'epoch': ep,
            'train_loss': tl,
            'val_mae': vm,
            'val_pinball': vl,
            'lr': lr_now,
            'curriculum_max_len': cur_max_len if curriculum_enabled else max_len,
            'train_samples': len(train_data),
        })

        if ep % 10 == 0:
            print(f"[{ep:3d}/{epochs}] train_loss={tl:.5f}  val_mae={vm:.5f}  val_pb={vl:.5f}  lr={lr_now:.2e}  max_len={cur_max_len if curriculum_enabled else max_len}")

        if vl < best:
            ckpt_name = cfg.train.get('ckpt_name', None) or f'{cfg.model.name}_seed{cfg.seed}.pt'
            ckpt_name = str(ckpt_name)
            best = vl
            no_improve = 0
            torch.save({
                'model': model.state_dict(),
                'cfg': OmegaConf.to_container(cfg, resolve=True),
                'epoch': ep,
                'best_val_pinball': best,
                'n_params': n_params,
            }, f'{ckpt_dir}/{ckpt_name}')
        else:
            no_improve += 1
            if patience > 0 and no_improve >= patience:
                print(f"Early stopping at epoch {ep} (no improvement for {patience} epochs)")
                break

    wandb.finish()
    print(f'best_val_pinball={best:.6f}  params={n_params:,d}')

if __name__=='__main__': main()
