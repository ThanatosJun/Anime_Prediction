import os

import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from src.model import get_embedding


# ── 單步 forward ──────────────────────────────────────────────────────────────

def _forward_orig(model, pixel_values, device):
    pixel_values = pixel_values.to(device)
    with torch.no_grad():
        return get_embedding(model, pixel_values)


def _forward_aug(model, pixel_values, device):
    pixel_values = pixel_values.to(device)
    return get_embedding(model, pixel_values)


# ── 單步 train / val ──────────────────────────────────────────────────────────

def _train_step(model, orig, aug, optimizer, loss_fn, device):
    orig_emb = _forward_orig(model, orig, device)
    aug_emb  = _forward_aug(model, aug, device)
    loss = loss_fn(aug_emb, orig_emb)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item()


def _val_step(model, orig, aug, loss_fn, device):
    with torch.no_grad():
        orig_emb = _forward_orig(model, orig, device)
        aug_emb  = _forward_aug(model, aug, device)
        loss = loss_fn(aug_emb, orig_emb)
    return loss.item()


# ── epoch 迴圈 ────────────────────────────────────────────────────────────────

def train_one_epoch(model, loader, optimizer, loss_fn, device):
    model.train()
    total = 0.0
    for orig, aug, _ in tqdm(loader, desc='Train', leave=False):
        total += _train_step(model, orig, aug, optimizer, loss_fn, device)
    return total / len(loader)


def validate(model, loader, loss_fn, device):
    model.eval()
    total = 0.0
    for orig, aug, _ in tqdm(loader, desc='Val', leave=False):
        total += _val_step(model, orig, aug, loss_fn, device)
    return total / len(loader)


# ── 評估 ──────────────────────────────────────────────────────────────────────

def _compute_cosine_similarity(emb_a, emb_b) -> float:
    emb_a = F.normalize(emb_a, dim=-1)
    emb_b = F.normalize(emb_b, dim=-1)
    return (emb_a * emb_b).sum(dim=-1).mean().item()


def evaluate_similarity(model, loader, device) -> float:
    model.eval()
    total = 0.0
    for orig, aug, _ in tqdm(loader, desc='Test', leave=False):
        with torch.no_grad():
            orig_emb = _forward_orig(model, orig, device)
            aug_emb  = _forward_aug(model, aug, device)
        total += _compute_cosine_similarity(orig_emb, aug_emb)
    return total / len(loader)


# ── TensorBoard ───────────────────────────────────────────────────────────────

def init_writer(log_dir: str) -> SummaryWriter:
    return SummaryWriter(log_dir=log_dir)


def log_metrics(writer: SummaryWriter, metrics: dict, epoch: int) -> None:
    for key, value in metrics.items():
        writer.add_scalar(key, value, epoch)


def close_writer(writer: SummaryWriter) -> None:
    writer.close()


# ── 存檔 ──────────────────────────────────────────────────────────────────────

def save_checkpoint(model, optimizer, epoch: int, path: str) -> None:
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, path)


def save_best(model, path: str) -> None:
    model.save_pretrained(path)


# ── 主流程 ────────────────────────────────────────────────────────────────────

def train(config: dict) -> None:
    import pandas as pd
    from src.model import load_model
    from src.loss import infonce_loss
    from util.image_process import get_transform_original, get_transform_aug
    from util.dataset import AnimeImageDataset, get_dataloader

    # device
    device = torch.device(
        config['training']['device']
        if torch.cuda.is_available()
        else 'cpu'
    )

    # model, optimizer & scheduler
    model = load_model(config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay'],
    )
    epochs        = config['training']['epochs']
    warmup_epochs = config['training']['warmup_epochs']
    warmup    = LinearLR(optimizer, start_factor=1e-3, end_factor=1.0, total_iters=warmup_epochs)
    cosine    = CosineAnnealingLR(optimizer, T_max=epochs - warmup_epochs)
    scheduler = SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[warmup_epochs])

    # transforms
    transform_orig = get_transform_original(config['data']['image_size'])
    transform_aug  = get_transform_aug(config)

    # datasets
    image_dir  = config['data']['image_dir']
    image_col  = config['data']['image_columns'][0]
    batch_size = config['training']['batch_size']
    split_csv  = config['data']['split_csv']

    train_df = pd.read_csv(split_csv['train'])
    val_df   = pd.read_csv(split_csv['val'])
    test_df  = pd.read_csv(split_csv['test'])

    train_loader = get_dataloader(
        AnimeImageDataset(train_df, image_dir, image_col, transform_orig, transform_aug),
        batch_size, shuffle=True,
    )
    val_loader = get_dataloader(
        AnimeImageDataset(val_df, image_dir, image_col, transform_orig, transform_aug),
        batch_size, shuffle=False,
    )
    test_loader = get_dataloader(
        AnimeImageDataset(test_df, image_dir, image_col, transform_orig, transform_aug),
        batch_size, shuffle=False,
    )

    # output paths
    run_id         = config['output']['run_id']
    results_dir    = config['output']['results_dir']
    log_dir        = os.path.join(results_dir, run_id, 'logs')
    best_dir       = os.path.join(results_dir, run_id, 'best')
    checkpoint_dir = os.path.join(results_dir, run_id, 'checkpoint')
    for d in (log_dir, best_dir, checkpoint_dir):
        os.makedirs(d, exist_ok=True)

    writer = init_writer(log_dir)

    # hyperparams
    val_interval         = config['training']['val_interval']
    checkpoint_interval  = config['training']['checkpoint_interval']
    tau                  = config['training']['tau']
    loss_fn = lambda aug_emb, orig_emb: infonce_loss(aug_emb, orig_emb, tau)

    best_val_loss = float('inf')

    for epoch in tqdm(range(1, epochs + 1), desc='Epochs'):
        train_loss = train_one_epoch(model, train_loader, optimizer, loss_fn, device)

        if epoch % val_interval == 0:
            val_loss   = validate(model, val_loader, loss_fn, device)
            cosine_sim = evaluate_similarity(model, val_loader, device)

            log_metrics(writer, {
                'train_loss': train_loss,
                'val_loss':   val_loss,
                'cosine_sim': cosine_sim,
                'lr':         optimizer.param_groups[0]['lr'],
            }, epoch)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_best(model, best_dir)

        if epoch % checkpoint_interval == 0:
            ckpt_path = os.path.join(checkpoint_dir, f'epoch_{epoch}.pt')
            save_checkpoint(model, optimizer, epoch, ckpt_path)

        scheduler.step()

    test_cosine_sim = evaluate_similarity(model, test_loader, device)
    log_metrics(writer, {'test_cosine_sim': test_cosine_sim}, epochs)
    close_writer(writer)
