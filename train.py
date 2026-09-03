from __future__ import annotations

import argparse
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.data import (
    N_DYNAMIC_FEATURES,
    N_STATIC_FEATURES,
    Panel,
    Scaler,
    WindowDataset,
    build_inference_batch,
    fit_scaler,
    load_panel,
    target_floor,
    window_starts,
)
from src.metrics import all_metrics, format_metrics
from src.model import LSTMForecaster, denormalize
from src.prism import PRISMForecaster

HISTORY = 168
HORIZON = 336


def build_model(args, n_series: int):
    if args.model == "lstm":
        return LSTMForecaster(
            n_dynamic=N_DYNAMIC_FEATURES,
            n_static=N_STATIC_FEATURES,
            n_series=n_series,
            hidden_size=args.hidden_size,
            num_layers=args.num_layers,
            dropout=args.dropout,
            embedding_dim=16,
            static_dim = 32
        )
    return PRISMForecaster(
        n_static=N_STATIC_FEATURES,
        n_series=n_series,
        history=HISTORY,
        horizon=HORIZON,
        d_model=args.hidden_size,
        patch_len=args.patch_len,
        dropout=args.dropout,
        mode=args.mode,
        pooling=args.pooling,
    )


def run_tag(args):
    if args.model == "lstm":
        base = "lstm"
    else:
        base = f"prism_{args.mode}"

        if args.mode != "stage2" and args.pooling != "mean":
            base += f"_{args.pooling}"

    return f"{base}_full" if args.full else base


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def select_device(requested):
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def move(batch,device):
    return {
        key: value.to(device, non_blocking=True) for key, value in batch.items()
        }


def predict(model, batch, device):
    batch = move(batch, device)
    normalized = model(
        batch["past_dynamic"],
        batch["past_target"],
        batch["future_dynamic"],
        batch["static"],
        batch["series_index"],
    )
    return denormalize(normalized, batch["location"], batch["scale"])


@torch.no_grad()
def evaluate(model, panel, scaler, forecast_start, device):
    model.eval()

    batch = build_inference_batch(panel, scaler, HISTORY, HORIZON, forecast_start)
    predictions = predict(model, batch, device).cpu().numpy()
    predictions = np.maximum(predictions, target_floor(panel, forecast_start))

    truth = panel.target[:, forecast_start : forecast_start + HORIZON]

    return all_metrics(truth, predictions)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, default=Path("checkpoints"))
    parser.add_argument("--model", choices=["lstm", "prism"], default="prism")
    parser.add_argument(
        "--mode", choices=["full", "stage1", "stage2"], default="full",
        help="prism ablation: full pipeline, stage 1 or stage 2 only",
    )
    parser.add_argument("--patch-len", type=int, default=24, help="prism patch length.")
    parser.add_argument("--full", action="store_true", help="train on all labelled hours.")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--stride", type=int, default=6, help="window stride in hours.")
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument(
        "--pooling",
        choices=["mean", "attention", "patches"],
        default="mean",
        help="pooling between PRISM stage 1 and stage 2",
    )
    args = parser.parse_args()

    set_seed(args.seed)
    device = select_device(args.device)

    panel = load_panel(args.data_dir)
    train_end = panel.n_train_steps if args.full else panel.n_train_steps - HORIZON
    scaler = fit_scaler(panel, train_end)
    starts = window_starts(train_end, HISTORY, HORIZON, args.stride)

    dataset = WindowDataset(panel, scaler, starts, HISTORY, HORIZON)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=True,
    )

    print(f"device={device}  series={panel.n_series}  steps={panel.n_steps}")
    print(f"train region=[0,{train_end})  windows={len(dataset)}  batches/epoch={len(loader)}")
    if not args.full:
        print(f"holdout=[{train_end},{train_end + HORIZON})")

    model = build_model(args, panel.n_series).to(device)
    tag = run_tag(args)
    print(f"model={tag}  parameters={sum(p.numel() for p in model.parameters()):,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = args.output_dir / f"{tag}.pt"
    scaler.save(args.output_dir / f"scaler{'_full' if args.full else ''}.npz")

    best_score = float("inf")
    epochs_without_gain = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        started = time.time()
        running_loss = 0.0

        for batch in loader:
            optimizer.zero_grad(set_to_none=True)

            predictions = predict(model, batch, device)

            loss = torch.nn.functional.l1_loss(predictions, batch["target"].to(device))

            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            running_loss += loss.item()

        schedule.step()
        train_loss = running_loss / len(loader)
        elapsed = time.time() - started

        if args.full:
            print(f"epoch {epoch:3d}  train_l1={train_loss:.4f}  {elapsed:.1f}s")
            torch.save({"state_dict": model.state_dict(), "args": vars(args)}, checkpoint_path)
            continue

        scores = evaluate(model, panel, scaler, train_end, device)
        marker = ""
        if scores["wape"] < best_score:
            best_score = scores["wape"]
            epochs_without_gain = 0
            torch.save({"state_dict": model.state_dict(), "args": vars(args)}, checkpoint_path)
            marker = "  *"
        else:
            epochs_without_gain += 1

        print(f"epoch {epoch:3d}  train_l1={train_loss:.4f}  "f"{format_metrics(scores)}  {elapsed:.1f}s{marker}")

        if epochs_without_gain >= args.patience:
            print(f"early stop: no wape gain for {args.patience} epochs")
            break

    print(f"\ncheckpoint: {checkpoint_path}")
    if not args.full:
        print(f"best holdout wape: {best_score:.4f}")


if __name__ == "__main__":
    main()
