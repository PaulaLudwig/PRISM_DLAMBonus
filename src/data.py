from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

SERIES_COL = "series_id"
TIME_COL = "timestamp"
TARGET_COL = "target"

DYNAMIC_COLUMNS = (
    "hour_sin",
    "hour_cos",
    "dow_sin",
    "dow_cos",
    "is_weekend",
    "workload_intensity",
    "promotion_intensity",
    "maintenance_known",
    "demand_forecast",
    "staffing_forecast",
    "upstream_quality_forecast",
    "shock_risk",
    "unit_reliability_forecast",
    "queue_pressure_forecast",
    "network_pressure_forecast",
    "event_load_forecast",
    "service_irregularity_risk_forecast",
    "throughput_disruption_risk_forecast",
)

# ten columns get binary observed feature so the model can tell a value is imputed
NULLABLE_COLUMNS = (
    "demand_forecast",
    "staffing_forecast",
    "upstream_quality_forecast",
    "shock_risk",
    "unit_reliability_forecast",
    "queue_pressure_forecast",
    "network_pressure_forecast",
    "event_load_forecast",
    "service_irregularity_risk_forecast",
    "throughput_disruption_risk_forecast",
)

# constant within each series
STATIC_COLUMNS = ("nominal_capacity", "zone_sin", "zone_cos")

# trend is excluded because it is increasing linearly and validation range lies above training range
# leads to out of distribution 
EXCLUDED_COLUMNS = ("trend",)

N_DYNAMIC_FEATURES = len(DYNAMIC_COLUMNS) + len(NULLABLE_COLUMNS)
N_STATIC_FEATURES = len(STATIC_COLUMNS)

# organize series for easier manipulation and access
@dataclass
class Panel:
    series_ids: list[str]
    timestamps: np.ndarray  # (n_steps)
    dynamic: np.ndarray  # (n_series, n_steps, N_DYNAMIC_FEATURES)
    static: np.ndarray  # (n_series, N_STATIC_FEATURES)
    target: np.ndarray  # (n_series, n_steps)
    n_train_steps: int  # 

    @property
    def n_series(self):
        return len(self.series_ids)

    @property
    def n_steps(self):
        return len(self.timestamps)


@dataclass
class Scaler:
    dynamic_mean: np.ndarray  # (N_DYNAMIC_FEATURES)
    dynamic_std: np.ndarray  # (N_DYNAMIC_FEATURES)
    static_mean: np.ndarray  # (N_STATIC_FEATURES)
    static_std: np.ndarray  # (N_STATIC_FEATURES)

    def transform_dynamic(self, values):
        return (values - self.dynamic_mean) / self.dynamic_std

    def transform_static(self, values):
        return (values - self.static_mean) / self.static_std

    def save(self, path):
        np.savez(
            path,
            dynamic_mean=self.dynamic_mean,
            dynamic_std=self.dynamic_std,
            static_mean=self.static_mean,
            static_std=self.static_std,
        )

    @classmethod
    def load(cls, path):
        with np.load(path) as data:
            return cls(
                dynamic_mean=data["dynamic_mean"],
                dynamic_std=data["dynamic_std"],
                static_mean=data["static_mean"],
                static_std=data["static_std"],
            )


def forward_fill(values):
    missing = np.isnan(values)
    step_index = np.broadcast_to(
        np.arange(values.shape[1], dtype=np.int64)[None, :, None], values.shape
    )
    last_valid = np.where(missing, 0, step_index)
    np.maximum.accumulate(last_valid, axis=1, out=last_valid)
    filled = np.take_along_axis(values, last_valid, axis=1)
    # positions before index 1 stay nan because there is no possibility for forward fill
    leading = np.take_along_axis(missing, last_valid, axis=1)
    return np.where(leading, np.nan, filled)


def load_panel(data_dir):
    # load training data and future covariates
    train = pd.read_csv(data_dir / "train.csv", parse_dates=[TIME_COL])
    future = pd.read_csv(data_dir / "validation_input.csv", parse_dates=[TIME_COL])

    # dummy target column to future data and concatenate together, sorting by unit then time
    n_train_steps = train[TIME_COL].nunique()
    future[TARGET_COL] = np.nan
    frame = pd.concat([train, future], ignore_index=True)
    frame = frame.sort_values([SERIES_COL, TIME_COL], kind="mergesort").reset_index(drop=True)

    series_ids = sorted(frame[SERIES_COL].unique())
    timestamps = np.sort(frame[TIME_COL].unique())
    n_series, n_steps = len(series_ids), len(timestamps)

    # (n_series, n_steps,  N_DYNAMIC_FEATURES)
    raw = frame[list(DYNAMIC_COLUMNS)].to_numpy(np.float32).reshape(n_series, n_steps, -1) 
    # missingness mask, go over nullable columns and create boolean missing mask
    observed = (~np.isnan(raw[:, :, [DYNAMIC_COLUMNS.index(c) for c in NULLABLE_COLUMNS]])).astype(
        np.float32
    )
   
    filled = forward_fill(raw)
    # for columns with nans at index 0 , fill by overall median of the column from training set
    train_median = np.nanmedian(filled[:, :n_train_steps, :], axis=(0, 1))
    filled = np.where(np.isnan(filled), train_median[None, None, :], filled)

    # concatenate imputed features and binary missing mask along the feature (dim=2) axis
    dynamic = np.concatenate([filled, observed], axis=2).astype(np.float32)
    static = (
        frame.groupby(SERIES_COL, sort=True)[list(STATIC_COLUMNS)]
        .first()
        .to_numpy(np.float32)
    )
    target = frame[TARGET_COL].to_numpy(np.float32).reshape(n_series, n_steps)

    return Panel(
        series_ids=list(series_ids),
        timestamps=timestamps,
        dynamic=dynamic,
        static=static,
        target=target,
        n_train_steps=int(n_train_steps),
    )


def fit_scaler(panel, train_end):
    region = panel.dynamic[:, :train_end, :]
    dynamic_mean = region.mean(axis=(0, 1))
    dynamic_std = region.std(axis=(0, 1))
    static_mean = panel.static.mean(axis=0)
    static_std = panel.static.std(axis=0)
    dynamic_std = np.where(dynamic_std < 1e-6, 1.0, dynamic_std)
    static_std = np.where(static_std < 1e-6, 1.0, static_std)
    return Scaler(
        dynamic_mean=dynamic_mean.astype(np.float32),
        dynamic_std=dynamic_std.astype(np.float32),
        static_mean=static_mean.astype(np.float32),
        static_std=static_std.astype(np.float32),
    )


def window_starts(train_end, history, horizon, stride):
    last_start = train_end - history - horizon
    return np.arange(0, last_start + 1, stride, dtype=np.int64)


class WindowDataset(Dataset):

    def __init__(self, panel, scaler, starts, history, horizon):
        self.history = history
        self.horizon = horizon
        self.dynamic = scaler.transform_dynamic(panel.dynamic).astype(np.float32)
        self.static = scaler.transform_static(panel.static).astype(np.float32)
        self.target = panel.target
        self.pairs = np.stack(
            np.meshgrid(np.arange(panel.n_series), starts, indexing="ij"), axis=-1
        ).reshape(-1, 2)

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        series_index, start = self.pairs[index]
        past_end = start + self.history
        future_end = past_end + self.horizon

        past_target = self.target[series_index, start:past_end]
        # instance normalization revin style
        # each window is centered and scaled by its own history
        location = np.float32(past_target.mean())
        scale = np.float32(max(past_target.std(), 1e-2))

        return {
            "past_dynamic": torch.from_numpy(self.dynamic[series_index, start:past_end]),
            "past_target": torch.from_numpy(((past_target - location) / scale)[:, None]),
            "future_dynamic": torch.from_numpy(self.dynamic[series_index, past_end:future_end]),
            "static": torch.from_numpy(self.static[series_index]),
            "series_index": torch.tensor(series_index, dtype=torch.long),
            "target": torch.from_numpy(self.target[series_index, past_end:future_end]),
            "location": torch.tensor(location),
            "scale": torch.tensor(scale),
        }


def build_inference_batch(panel, scaler, history, horizon, forecast_start):
    starts = np.full(panel.n_series, forecast_start - history, dtype=np.int64)
    dataset = WindowDataset(panel, scaler, np.array([0]), history, horizon)
    dataset.pairs = np.stack([np.arange(panel.n_series), starts], axis=-1)
    samples = [dataset[i] for i in range(panel.n_series)]
    return {key: torch.stack([sample[key] for sample in samples]) for key in samples[0]}


def target_floor(panel, train_end) -> float:
    return float(np.nanmin(panel.target[:, :train_end]))


def read_metadata(data_dir):
    return json.loads((data_dir / "metadata.json").read_text())
