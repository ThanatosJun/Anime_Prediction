"""
Fine-tune top-N transformer encoder layers for anime text regression.

Strategy (Optimization #3 — Layer-Wise Rate Tuning):
  - Start from a pretrained encoder (default: intfloat/e5-base-v2).
  - Freeze ALL encoder layers.
  - Unfreeze only the top N transformer blocks (--unfreeze-layers).
  - Attach a tiny regression head; train with discriminative learning rates:
      head params    : lr_head  (default 1e-4)
      unfrozen layers: lr_top   (default 1e-5)
      frozen layers  : no grad  (effectively lr = 0)
  - Track val Spearman on popularity (primary metric); save best encoder.
  - The head is discarded at save time; the encoder is written as a
    SentenceTransformer so the existing EmbeddingGenerator loads it unchanged.

Workflow (A1 = top-2, A2 = top-3):

  # 1. Fine-tune
  python -m src.text_branch.finetune_encoder --unfreeze-layers 2 --run-id A1
  python -m src.text_branch.finetune_encoder --unfreeze-layers 3 --run-id A2

  # 2. Re-generate embeddings with fine-tuned encoder
  python -m src.text_branch.run_text_embedding_pipeline \\
      --finetuned-model-path artifacts/finetuned_encoder_A1 \\
      --output-prefix text_embeddings_A1 \\
      --report-name text_embedding_pipeline_summary_A1.json

  # 3. Evaluate (unchanged Ridge flow)
  python -m src.text_branch.baseline_model \\
      --embedding-prefix text_embeddings_A1 \\
      --experiment-name A1_top2layers \\
      --report-name text_branch_metrics_A1.json
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.stats import spearmanr
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer

try:
    from sentence_transformers import SentenceTransformer
    from sentence_transformers import models as st_models
    _ST_AVAILABLE = True
except ImportError:
    _ST_AVAILABLE = False

try:
    from .text_preprocessor import TextPreprocessor
except ImportError:
    from text_preprocessor import TextPreprocessor


# ──────────────────────────────────────────────────────────────────────────────
# Dataset
# ──────────────────────────────────────────────────────────────────────────────

class _TextRegressionDataset(Dataset):
    """Tokenises texts once at construction and stores as tensors."""

    def __init__(
        self,
        texts: List[str],
        targets: np.ndarray,      # shape (N, n_targets), already z-scored
        tokenizer,
        max_length: int,
    ) -> None:
        self.encodings = tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=max_length,
            return_tensors="pt",
        )
        self.targets = torch.tensor(targets, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, idx: int):
        return {k: v[idx] for k, v in self.encodings.items()}, self.targets[idx]


# ──────────────────────────────────────────────────────────────────────────────
# Model
# ──────────────────────────────────────────────────────────────────────────────

class _EncoderWithHead(nn.Module):
    """Encoder (frozen/partially unfrozen) + mean-pool + linear regression head."""

    def __init__(self, encoder: nn.Module, hidden_size: int, n_targets: int) -> None:
        super().__init__()
        self.encoder = encoder
        self.head = nn.Linear(hidden_size, n_targets)
        nn.init.normal_(self.head.weight, std=0.02)
        nn.init.zeros_(self.head.bias)

    @staticmethod
    def _mean_pool(last_hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        mask = attention_mask.unsqueeze(-1).float()
        return (last_hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-9)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, **kwargs) -> torch.Tensor:
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask, **kwargs)
        pooled = self._mean_pool(out.last_hidden_state, attention_mask)
        return self.head(pooled)


# ──────────────────────────────────────────────────────────────────────────────
# Layer freezing helpers
# ──────────────────────────────────────────────────────────────────────────────

def _freeze_all(model: nn.Module) -> None:
    for p in model.parameters():
        p.requires_grad_(False)


def _unfreeze_top_n_layers(encoder: nn.Module, n: int) -> int:
    """
    Unfreeze the top-N transformer encoder blocks for BERT-style models.
    Returns the index of the first unfrozen layer.
    Raises RuntimeError if the layer structure is unrecognised.
    """
    layers = None
    if hasattr(encoder, "encoder") and hasattr(encoder.encoder, "layer"):
        layers = encoder.encoder.layer          # BERT / RoBERTa / e5
    elif hasattr(encoder, "layers"):
        layers = encoder.layers                 # some other HF models

    if layers is None:
        raise RuntimeError(
            "Cannot locate transformer layers. "
            "Expected encoder.encoder.layer (BERT-style) or encoder.layers."
        )

    total = len(layers)
    if n > total:
        raise ValueError(f"--unfreeze-layers {n} > total layers {total}")

    start = total - n
    for layer in layers[start:]:
        for p in layer.parameters():
            p.requires_grad_(True)

    return start


def _count_trainable(model: nn.Module) -> Tuple[int, int]:
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total


def _build_param_groups(model: _EncoderWithHead, lr_head: float, lr_top: float) -> List[Dict]:
    """
    Two parameter groups with different learning rates:
      - regression head  : lr_head
      - unfrozen encoder : lr_top
    Frozen params (requires_grad=False) are excluded automatically by AdamW.
    """
    head_ids = {id(p) for p in model.head.parameters()}
    return [
        {"params": [p for p in model.parameters() if id(p) in head_ids and p.requires_grad],
         "lr": lr_head},
        {"params": [p for p in model.parameters() if id(p) not in head_ids and p.requires_grad],
         "lr": lr_top},
    ]


# ──────────────────────────────────────────────────────────────────────────────
# Training / evaluation helpers
# ──────────────────────────────────────────────────────────────────────────────

def _train_epoch(
    model: _EncoderWithHead,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: str,
    grad_clip: float,
) -> float:
    model.train()
    criterion = nn.MSELoss()
    total_loss = 0.0
    for batch_enc, targets in loader:
        batch_enc = {k: v.to(device) for k, v in batch_enc.items()}
        targets = targets.to(device)
        optimizer.zero_grad()
        preds = model(**batch_enc)
        loss = criterion(preds, targets)
        loss.backward()
        nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad], grad_clip
        )
        optimizer.step()
        total_loss += loss.item() * len(targets)
    return total_loss / max(len(loader.dataset), 1)


@torch.no_grad()
def _eval_epoch(
    model: _EncoderWithHead,
    loader: DataLoader,
    device: str,
    scalers: List[StandardScaler],
    target_names: List[str],
) -> Dict[str, float]:
    model.eval()
    all_preds, all_targets = [], []
    for batch_enc, targets in loader:
        batch_enc = {k: v.to(device) for k, v in batch_enc.items()}
        preds = model(**batch_enc).cpu().numpy()
        all_preds.append(preds)
        all_targets.append(targets.numpy())

    preds_arr = np.concatenate(all_preds, axis=0)    # (N, n_targets) scaled
    targets_arr = np.concatenate(all_targets, axis=0)

    metrics: Dict[str, float] = {}
    for i, (name, scaler) in enumerate(zip(target_names, scalers)):
        pred_orig = scaler.inverse_transform(preds_arr[:, i : i + 1]).ravel()
        tgt_orig = scaler.inverse_transform(targets_arr[:, i : i + 1]).ravel()
        rho, _ = spearmanr(tgt_orig, pred_orig)
        metrics[name] = float(rho) if np.isfinite(rho) else 0.0

    return metrics


# ──────────────────────────────────────────────────────────────────────────────
# Save as SentenceTransformer
# ──────────────────────────────────────────────────────────────────────────────

def _save_as_sentence_transformer(
    encoder: nn.Module,
    tokenizer_source: str,
    hidden_size: int,
    save_path: Path,
) -> None:
    """
    Persist the fine-tuned encoder as a SentenceTransformer so that the
    existing EmbeddingGenerator can load it with no code changes.

    Layout on disk:
      save_path/
        modules.json          ← ST metadata
        sentence_bert_config.json
        0_Transformer/        ← the fine-tuned HF model + tokenizer
        1_Pooling/            ← mean-pooling config
    """
    if not _ST_AVAILABLE:
        raise RuntimeError(
            "sentence-transformers is not installed. "
            "Run: pip install sentence-transformers"
        )

    # --- Save HF model and tokenizer to a sub-folder ---
    hf_path = save_path / "0_Transformer"
    hf_path.mkdir(parents=True, exist_ok=True)
    encoder.save_pretrained(str(hf_path))
    tok = AutoTokenizer.from_pretrained(tokenizer_source)
    tok.save_pretrained(str(hf_path))

    # --- Wrap as SentenceTransformer and save ---
    word_model = st_models.Transformer(str(hf_path), max_seq_length=512)
    pooling = st_models.Pooling(hidden_size, pooling_mode_mean_tokens=True)
    st_model = SentenceTransformer(modules=[word_model, pooling])
    st_model.save(str(save_path))
    print(f"SentenceTransformer saved → {save_path}")


# ──────────────────────────────────────────────────────────────────────────────
# Data loading
# ──────────────────────────────────────────────────────────────────────────────

def _load_split(
    csv_path: Path,
    text_col: str,
    target_cols: List[str],
    preprocessor: TextPreprocessor,
) -> Tuple[List[str], np.ndarray]:
    """Load CSV, apply text preprocessing, drop rows with missing targets."""
    df = pd.read_csv(csv_path)
    for col in [text_col] + target_cols:
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found in {csv_path}")

    mask = df[text_col].notna()
    for col in target_cols:
        mask &= df[col].notna()
    df = df.loc[mask].copy()

    texts = df[text_col].apply(preprocessor.clean).fillna("").tolist()
    targets = df[target_cols].to_numpy(dtype=np.float64)
    return texts, targets


# ──────────────────────────────────────────────────────────────────────────────
# Main fine-tune routine
# ──────────────────────────────────────────────────────────────────────────────

def finetune(
    model_name: str,
    data_dir: Path,
    unfreeze_layers: int,
    run_id: str,
    artifact_dir: Path,
    target_cols: List[str],
    text_col: str,
    epochs: int,
    batch_size: int,
    lr_head: float,
    lr_top: float,
    grad_clip: float,
    patience: int,
    max_length: int,
    device: str,
    random_seed: int,
) -> Dict:
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)

    preprocessor = TextPreprocessor(
        lowercase=True,
        remove_urls=True,
        remove_marketing=False,   # keep signal (see Experiment 01 findings)
        remove_extra_whitespace=True,
        min_length=10,
        max_length=max_length,
    )

    # ── Load data ────────────────────────────────────────────────────────────
    train_csv = data_dir / "anilist_anime_multimodal_input_train.csv"
    val_csv = data_dir / "anilist_anime_multimodal_input_val.csv"

    print("Loading train/val data...")
    train_texts, train_targets = _load_split(train_csv, text_col, target_cols, preprocessor)
    val_texts, val_targets = _load_split(val_csv, text_col, target_cols, preprocessor)
    print(f"  Train: {len(train_texts)} rows | Val: {len(val_texts)} rows")

    # ── Scale targets (z-score per target, fit on train only) ────────────────
    scalers: List[StandardScaler] = [StandardScaler() for _ in target_cols]
    train_scaled = np.column_stack(
        [s.fit_transform(train_targets[:, i : i + 1]).ravel() for i, s in enumerate(scalers)]
    ).astype(np.float32)
    val_scaled = np.column_stack(
        [s.transform(val_targets[:, i : i + 1]).ravel() for i, s in enumerate(scalers)]
    ).astype(np.float32)

    # ── Load tokenizer and base encoder ──────────────────────────────────────
    print(f"\nLoading tokenizer & encoder: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    encoder = AutoModel.from_pretrained(model_name)
    hidden_size: int = encoder.config.hidden_size

    # ── Freeze all; then unfreeze top N layers ────────────────────────────────
    _freeze_all(encoder)
    first_unfrozen = _unfreeze_top_n_layers(encoder, unfreeze_layers)
    total_layers = len(encoder.encoder.layer)
    trainable, total_params = _count_trainable(encoder)
    print(
        f"Unfroze layers {first_unfrozen}–{total_layers - 1} "
        f"(top {unfreeze_layers} of {total_layers})"
    )
    print(f"Trainable encoder params: {trainable:,} / {total_params:,} ({100 * trainable / total_params:.1f}%)")

    # ── Build model + optimizer ───────────────────────────────────────────────
    model = _EncoderWithHead(encoder, hidden_size, n_targets=len(target_cols)).to(device)
    param_groups = _build_param_groups(model, lr_head=lr_head, lr_top=lr_top)
    optimizer = torch.optim.AdamW(param_groups, weight_decay=0.01)

    # ── DataLoaders ───────────────────────────────────────────────────────────
    train_ds = _TextRegressionDataset(train_texts, train_scaled, tokenizer, max_length)
    val_ds = _TextRegressionDataset(val_texts, val_scaled, tokenizer, max_length)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    # ── Training loop ─────────────────────────────────────────────────────────
    best_val_spearman = -np.inf
    best_epoch = -1
    patience_counter = 0
    history: List[Dict] = []
    save_path = artifact_dir / f"finetuned_encoder_{run_id}"

    col_w = 25
    header = (
        f"{'Epoch':>5} | {'Train MSE Loss':>14} | "
        f"{'Val Spearman (' + target_cols[0] + ')':>{col_w}} | "
        f"{'Val Spearman (' + (target_cols[1] if len(target_cols) > 1 else '') + ')':>{col_w}}"
    )
    print(f"\n{header}")
    print("-" * len(header))

    for epoch in range(1, epochs + 1):
        train_loss = _train_epoch(model, train_loader, optimizer, device, grad_clip)
        val_metrics = _eval_epoch(model, val_loader, device, scalers, target_cols)

        sp0 = val_metrics.get(target_cols[0], 0.0)
        sp1 = val_metrics.get(target_cols[1], float("nan")) if len(target_cols) > 1 else float("nan")

        print(f"{epoch:>5} | {train_loss:>14.4f} | {sp0:>{col_w}.4f} | {sp1:>{col_w}.4f}")

        history.append({
            "epoch": epoch,
            "train_loss": round(train_loss, 6),
            f"val_spearman_{target_cols[0]}": round(sp0, 6),
            f"val_spearman_{target_cols[1] if len(target_cols) > 1 else 'n/a'}": round(sp1, 6),
        })

        if sp0 > best_val_spearman:
            best_val_spearman = sp0
            best_epoch = epoch
            patience_counter = 0
            # Save encoder only (head is discarded)
            _save_as_sentence_transformer(
                encoder=model.encoder,
                tokenizer_source=model_name,
                hidden_size=hidden_size,
                save_path=save_path,
            )
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\nEarly stopping at epoch {epoch} (patience={patience}).")
                break

    print(
        f"\nBest epoch: {best_epoch} | "
        f"Val Spearman ({target_cols[0]}): {best_val_spearman:.4f}"
    )
    print(f"Saved encoder → {save_path}")

    return {
        "run_id": run_id,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "model_name": model_name,
        "unfreeze_layers": unfreeze_layers,
        "first_unfrozen_layer": first_unfrozen,
        "total_encoder_layers": total_layers,
        "trainable_encoder_params": trainable,
        "total_encoder_params": total_params,
        "lr_head": lr_head,
        "lr_top": lr_top,
        "batch_size": batch_size,
        "epochs_run": len(history),
        "best_epoch": best_epoch,
        f"best_val_spearman_{target_cols[0]}": round(best_val_spearman, 6),
        "history": history,
        "encoder_path": str(save_path.as_posix()),
    }


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Fine-tune top-N encoder layers for anime text regression (Opt #3)."
    )
    p.add_argument("--model-name", type=str, default="intfloat/e5-base-v2",
                   help="HuggingFace model id or local path.")
    p.add_argument("--data-dir", type=Path, default=Path("data/processed"),
                   help="Directory containing split CSV files.")
    p.add_argument("--unfreeze-layers", type=int, default=2, choices=[1, 2, 3, 4],
                   help="Number of top transformer layers to unfreeze.")
    p.add_argument("--run-id", type=str, default="A1",
                   help="Run identifier used in artifact names (e.g. A1, A2).")
    p.add_argument("--artifact-dir", type=Path, default=Path("artifacts"),
                   help="Where to save the fine-tuned SentenceTransformer.")
    p.add_argument("--report-dir", type=Path, default=Path("reports"))
    p.add_argument("--targets", nargs="+", default=["popularity", "meanScore"])
    p.add_argument("--text-column", type=str, default="description")
    p.add_argument("--epochs", type=int, default=6,
                   help="Maximum training epochs (early stopping may end sooner).")
    p.add_argument("--batch-size", type=int, default=8,
                   help="Batch size. Reduce if GPU OOM (default 8 is conservative).")
    p.add_argument("--lr-head", type=float, default=1e-4,
                   help="Learning rate for the regression head.")
    p.add_argument("--lr-top", type=float, default=1e-5,
                   help="Learning rate for the unfrozen encoder layers.")
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--patience", type=int, default=3,
                   help="Early stopping patience (epochs without val improvement).")
    p.add_argument("--max-length", type=int, default=512)
    p.add_argument("--device", type=str, default="auto",
                   help="auto | cuda | cpu")
    p.add_argument("--random-seed", type=int, default=42)
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    args.artifact_dir.mkdir(parents=True, exist_ok=True)
    args.report_dir.mkdir(parents=True, exist_ok=True)

    result = finetune(
        model_name=args.model_name,
        data_dir=args.data_dir,
        unfreeze_layers=args.unfreeze_layers,
        run_id=args.run_id,
        artifact_dir=args.artifact_dir,
        target_cols=args.targets,
        text_col=args.text_column,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr_head=args.lr_head,
        lr_top=args.lr_top,
        grad_clip=args.grad_clip,
        patience=args.patience,
        max_length=args.max_length,
        device=device,
        random_seed=args.random_seed,
    )

    report_path = args.report_dir / f"finetune_{args.run_id}.json"
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(f"Run report → {report_path}")


if __name__ == "__main__":
    main()
