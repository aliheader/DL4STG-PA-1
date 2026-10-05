from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "Data"
RESULTS_DIR = ROOT / "results"

FEATURES = [f"feature_{c}" for c in "ABCDEFGHIJ"]
PRED_LEN = 168

PROFILE_PERIODS = {"seasonal_day": 24, "seasonal_week": 168, "seasonal_year": 8766}
PROFILE_FEATURES = list(PROFILE_PERIODS)


def load_target() -> np.ndarray:

    df = pd.read_csv(DATA_DIR / "student_train.csv")
    assert (
        df["time_idx"].to_numpy() == np.arange(1, len(df) + 1)
    ).all(), "time_idx not 1..N"
    return df["value"].to_numpy(dtype=np.float64)


def load_external(columns=FEATURES) -> np.ndarray:

    df = pd.read_csv(DATA_DIR / "optional_external_data.csv")
    assert (
        df["time_idx"].to_numpy() == np.arange(1, len(df) + 1)
    ).all(), "time_idx not 1..N"
    return df[list(columns)].to_numpy(dtype=np.float64)


def seasonal_profile(
    y: np.ndarray, train_end: int, n_total: int, period: int
) -> np.ndarray:

    phase_hist = np.arange(train_end) % period
    means = pd.Series(y[:train_end]).groupby(phase_hist).mean()
    phase_full = np.arange(n_total) % period
    return means.loc[phase_full].to_numpy(dtype=np.float64)


def load_covariates(columns, y=None, train_end=None) -> np.ndarray:

    columns = list(columns)
    n_total = len(load_target()) + PRED_LEN
    if not columns:
        return np.zeros((n_total, 0))
    raw_cols = [c for c in columns if c in FEATURES]
    prof_cols = [c for c in columns if c in PROFILE_FEATURES]
    assert set(raw_cols) | set(prof_cols) == set(
        columns
    ), f"unknown covariate columns: {columns}"
    parts = {}
    if raw_cols:
        parts.update(zip(raw_cols, load_external(raw_cols).T))
    if prof_cols:
        assert (
            y is not None and train_end is not None
        ), "profile features need y and train_end"
        parts.update(
            (c, seasonal_profile(y, train_end, n_total, PROFILE_PERIODS[c]))
            for c in prof_cols
        )
    return np.stack([parts[c] for c in columns], axis=1)


def standardise_covariates(cov: np.ndarray, train_end: int) -> np.ndarray:
    """Zero-mean / unit-std per column, using ONLY rows before `train_end` (no peeking)."""
    mean = cov[:train_end].mean(axis=0)
    std = cov[:train_end].std(axis=0)
    std[std == 0] = 1.0
    return (cov - mean) / std


def make_window(y, cov, origin, L_in, label_len, pred_len, past_cov, future_cov):

    hist = y[origin - L_in : origin]
    mu, sd = hist.mean(), max(hist.std(), 1.0)

    x_enc = ((hist - mu) / sd)[:, None]
    if past_cov:
        x_enc = np.concatenate([x_enc, cov[origin - L_in : origin]], axis=1)

    span = label_len + pred_len
    if future_cov:
        x_dec_cov = cov[origin - label_len : origin + pred_len]
    else:
        x_dec_cov = np.zeros((span, 0))
    return x_enc, x_dec_cov, mu, sd


class WindowDataset(Dataset):
    """All training windows, sliced on the fly (cheap) instead of stored (would be ~200 MB)."""

    def __init__(
        self, y, cov, origins, L_in, label_len, pred_len, past_cov, future_cov
    ):
        self.y, self.cov, self.origins = y, cov, list(origins)
        self.kw = dict(
            L_in=L_in,
            label_len=label_len,
            pred_len=pred_len,
            past_cov=past_cov,
            future_cov=future_cov,
        )
        self.pred_len = pred_len

    def __len__(self):
        return len(self.origins)

    def __getitem__(self, i):
        o = self.origins[i]
        x_enc, x_dec, mu, sd = make_window(self.y, self.cov, o, **self.kw)
        target = (
            self.y[o : o + self.pred_len] - mu
        ) / sd  # same normalisation as input
        as_t = lambda a: torch.as_tensor(a, dtype=torch.float32)
        return (
            as_t(x_enc),
            as_t(x_dec),
            as_t(target),
            torch.tensor(sd, dtype=torch.float32),
        )


def acf(x: np.ndarray, lag: int) -> float:
    xc = x - x.mean()
    return float(np.dot(xc[:-lag], xc[lag:]) / np.dot(xc, xc))
