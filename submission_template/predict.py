"""Inference entrypoint for final private evaluation."""

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
from src.prism import PRISMForecaster

HISTORY = 168
HORIZON = 336


def load_forecast_index(input_dir: Path) -> pd.DataFrame:
    candidates = [
        input_dir / "forecast_index_test.csv",
        input_dir / "forecast_index_validation.csv",
    ]
    for forecast_index in candidates:
        if forecast_index.exists():
            return pd.read_csv(forecast_index, parse_dates=[TIME_COL])
    expected = ", ".join(path.name for path in candidates)
    raise FileNotFoundError(f"Expected one of {expected} in input_dir.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate private test predictions.")
    parser.add_argument("--input_dir", required=True, type=Path)
    parser.add_argument("--output_file", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    args = parser.parse_args()

    if not args.checkpoint.exists():
        raise FileNotFoundError(f"Missing checkpoint: {args.checkpoint}")

    scaler_path = args.checkpoint.parent / "scaler.npz"
    if not scaler_path.exists():
        raise FileNotFoundError(f"Missing {scaler_path} (must sit next to checkpoint.pt).")
    scaler = Scaler.load(scaler_path)

    panel = load_panel(args.input_dir)
    forecast_index = load_forecast_index(args.input_dir)

    model = PRISMForecaster(
        n_static=N_STATIC_FEATURES,
        n_series=panel.n_series,
        history=HISTORY,
        horizon=HORIZON,
        d_model=128,
        patch_len=24,
        dropout=0.2,
        mode="full",
        pooling="patches",
    )
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    batch = build_inference_batch(panel, scaler, HISTORY, HORIZON, panel.n_train_steps)
    with torch.no_grad():
        normalized = model(
            batch["past_dynamic"],
            batch["past_target"],
            batch["future_dynamic"],
            batch["static"],
            batch["series_index"],
        )
        predictions = (normalized * batch["scale"].unsqueeze(1) + batch["location"].unsqueeze(1)).numpy()

    predictions = np.maximum(predictions, target_floor(panel, panel.n_train_steps))

    forecast_timestamps = panel.timestamps[panel.n_train_steps:]
    predicted = pd.DataFrame({
        SERIES_COL: [sid for sid in panel.series_ids for _ in range(HORIZON)],
        TIME_COL: list(forecast_timestamps) * panel.n_series,
        "prediction": predictions.reshape(-1),
    })

    output = forecast_index.merge(predicted, on=[SERIES_COL, TIME_COL], how="left")

    missing = int(output["prediction"].isna().sum())
    if missing:
        raise ValueError(f"{missing} forecast-index rows received no prediction.")
    if len(output) != len(forecast_index):
        raise ValueError(f"Row count changed during merge: {len(forecast_index)} -> {len(output)}.")

    args.output_file.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output_file, index=False)
    print(f"wrote {len(output)} rows to {args.output_file}")


if __name__ == "__main__":
    main()