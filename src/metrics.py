from __future__ import annotations

import numpy as np


def wape(y_true, y_pred):
    return float(np.abs(y_true - y_pred).sum() / np.abs(y_true).sum())


def mae(y_true, y_pred):
    return float(np.abs(y_true - y_pred).mean())


def mse(y_true, y_pred):
    return float(((y_true - y_pred) ** 2).mean())


def rmse(y_true, y_pred):
    return float(np.sqrt(mse(y_true, y_pred)))


def mape(y_true, y_pred):
    return float((np.abs(y_true - y_pred) / np.abs(y_true)).mean())


def smape(y_true, y_pred):
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    return float((np.abs(y_true - y_pred) / denominator).mean())


def all_metrics(y_true, y_pred):
    return {
        "wape": wape(y_true, y_pred),
        "mae": mae(y_true, y_pred),
        "mse": mse(y_true, y_pred),
        "rmse": rmse(y_true, y_pred),
        "mape": mape(y_true, y_pred),
        "smape": smape(y_true, y_pred),
    }


def format_metrics(values):
    return "  ".join(f"{name}={value:.4f}" for name, value in values.items())
