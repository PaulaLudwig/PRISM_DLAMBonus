from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from PRISM_DLAMBonus.predict_validation import build_from_checkpoint
from src.data import Scaler, build_inference_batch, load_panel, target_floor
from src.metrics import all_metrics, format_metrics
from src.model import denormalize

HISTORY = 168
HORIZON = 336


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description="Score a checkpoint on the local holdout.")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--scaler", type=Path, default=Path("checkpoints/scaler.npz"))
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    panel = load_panel(args.data_dir)
    scaler = Scaler.load(args.scaler)
    forecast_start = panel.n_train_steps - HORIZON 

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = build_from_checkpoint(checkpoint.get("args", {}), panel.n_series).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    batch = build_inference_batch(panel, scaler, HISTORY, HORIZON, forecast_start)
    batch = {key: value.to(device) for key, value in batch.items()}
    normalized = model(
        batch["past_dynamic"],
        batch["past_target"],
        batch["future_dynamic"],
        batch["static"],
        batch["series_index"],
    )
    predictions = denormalize(normalized, batch["location"], batch["scale"]).cpu().numpy()
    predictions = np.maximum(predictions, target_floor(panel, forecast_start))
    truth = panel.target[:, forecast_start : forecast_start + HORIZON]

    print(f"{args.checkpoint.name:<24} {format_metrics(all_metrics(truth, predictions))}")


if __name__ == "__main__":
    main()
