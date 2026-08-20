from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.data import (
    N_DYNAMIC_FEATURES,
    N_STATIC_FEATURES,
    SERIES_COL,
    TIME_COL,
    Scaler,
    build_inference_batch,
    load_panel,
    target_floor,
)
from src.model import LSTMForecaster, denormalize
from src.prism import PRISMForecaster

HISTORY = 168
HORIZON = 336


def build_from_checkpoint(saved, n_series):
    if saved.get("model", "lstm") == "lstm":
        return LSTMForecaster(
            n_dynamic=N_DYNAMIC_FEATURES,
            n_static=N_STATIC_FEATURES,
            n_series=n_series,
            hidden_size=saved.get("hidden_size", 128),
            num_layers=saved.get("num_layers", 2),
            dropout=saved.get("dropout", 0.2),
        )
    return PRISMForecaster(
        n_static=N_STATIC_FEATURES,
        n_series=n_series,
        history=HISTORY,
        horizon=HORIZON,
        d_model=saved.get("hidden_size", 128),
        patch_len=saved.get("patch_len", 24),
        dropout=saved.get("dropout", 0.2),
        mode=saved.get("mode", "full"),
    )


def main():
    parser = argparse.ArgumentParser(description="Write validation predictions.")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/lstm.pt"))
    parser.add_argument("--scaler", type=Path, default=Path("checkpoints/scaler.npz"))
    parser.add_argument("--output-file", type=Path, default=Path("predictions/validation.csv"))
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    panel = load_panel(args.data_dir)
    scaler = Scaler.load(args.scaler)

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    saved_args = checkpoint.get("args", {})
    model = build_from_checkpoint(saved_args, panel.n_series).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    batch = build_inference_batch(panel, scaler, HISTORY, HORIZON, panel.n_train_steps)
    with torch.no_grad():
        batch = {key: value.to(device) for key, value in batch.items()}
        normalized = model(
            batch["past_dynamic"],
            batch["past_target"],
            batch["future_dynamic"],
            batch["static"],
            batch["series_index"],
        )
        predictions = denormalize(normalized, batch["location"], batch["scale"]).cpu().numpy()

    predictions = np.maximum(predictions, target_floor(panel, panel.n_train_steps))

    forecast_timestamps = panel.timestamps[panel.n_train_steps :]
    predicted = pd.DataFrame(
        {
            SERIES_COL: [sid for sid in panel.series_ids for _ in range(HORIZON)],
            TIME_COL: list(forecast_timestamps) * panel.n_series,
            "prediction": predictions.reshape(-1),
        }
    )

    index = pd.read_csv(args.data_dir / "forecast_index_validation.csv", parse_dates=[TIME_COL])
    output = index.merge(predicted, on=[SERIES_COL, TIME_COL], how="left")

    missing = int(output["prediction"].isna().sum())
    if missing:
        raise ValueError(f"{missing} forecast-index rows received no prediction.")
    if len(output) != len(index):
        raise ValueError(f"Row count changed during merge: {len(index)} -> {len(output)}.")

    args.output_file.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output_file, index=False)
    print(f"wrote {len(output)} rows to {args.output_file}")
    print(output["prediction"].describe().to_string())


if __name__ == "__main__":
    main()
