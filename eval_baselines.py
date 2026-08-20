from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent / "baseline"))

from baselines import make_all_baselines  

from src.metrics import all_metrics, format_metrics 
HORIZON = 336


def main():
    parser = argparse.ArgumentParser(description="Score baselines on the local holdout.")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    args = parser.parse_args()

    frame = pd.read_csv(args.data_dir / "train.csv", parse_dates=["timestamp"])
    cutoff = frame["timestamp"].sort_values().unique()[-HORIZON]

    history = frame[frame["timestamp"] < cutoff]
    holdout = frame[frame["timestamp"] >= cutoff]
    index = holdout[["series_id", "timestamp"]].copy()
    print(f"history ends {history['timestamp'].max()}   holdout starts {cutoff}")
    print(f"holdout rows {len(holdout)} = {holdout['series_id'].nunique()} series x {HORIZON}\n")

    truth = holdout[["series_id", "timestamp", "target"]]
    for name, predictions in make_all_baselines(history, index).items():
        merged = truth.merge(predictions, on=["series_id", "timestamp"], how="left")
        if merged["prediction"].isna().any():
            raise ValueError(f"Baseline {name} left gaps in the holdout index.")
        scores = all_metrics(merged["target"].to_numpy(), merged["prediction"].to_numpy())
        print(f"{name:<20} {format_metrics(scores)}")


if __name__ == "__main__":
    main()
