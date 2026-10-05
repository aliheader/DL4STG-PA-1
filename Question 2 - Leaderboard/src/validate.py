import numpy as np
import pandas as pd

from data import PRED_LEN, RESULTS_DIR, load_target

K_BLOCKS = 12
ES_BLOCKS = 4


def rmse(y, p):
    return float(np.sqrt(np.mean((y - p) ** 2)))


def mae(y, p):
    return float(np.mean(np.abs(y - p)))


def smape(y, p):
    return float(
        100 * np.mean(2 * np.abs(p - y) / np.maximum(np.abs(y) + np.abs(p), 1e-9))
    )


def block_origins(n: int, k: int = K_BLOCKS, skip_last: int = 0) -> list[int]:

    end = n - skip_last * PRED_LEN
    return [end - (k - b) * PRED_LEN for b in range(k)]


def evaluate(forecast_fn, y: np.ndarray, origins: list[int]) -> pd.DataFrame:
    """Call forecast_fn(origin) -> 168 values for each block; return one row of metrics per block."""
    rows = []
    for b, o in enumerate(origins):
        pred = np.asarray(forecast_fn(o), dtype=np.float64)
        truth = y[o : o + PRED_LEN]
        assert pred.shape == truth.shape == (PRED_LEN,)
        rows.append(
            dict(
                block=b,
                origin=o,
                rmse=rmse(truth, pred),
                mae=mae(truth, pred),
                smape=smape(truth, pred),
            )
        )
    return pd.DataFrame(rows)


def paired_diff(
    a: pd.DataFrame, b: pd.DataFrame, metric: str = "rmse", keys=("block",)
):
    m = a.merge(b, on=list(keys), suffixes=("_a", "_b"))
    d = m[f"{metric}_a"] - m[f"{metric}_b"]
    return float(d.mean()), float(d.std(ddof=1)), int(len(d))


def hour_of_day_mean(h):
    profile = h[-28 * 24 :].reshape(28, 24).mean(axis=0)  # 28 days x 24 hours
    return profile[np.arange(PRED_LEN) % 24]


def damped_persistence(h, phi=0.966):

    m = h[-720:].mean()
    return m + (h[-1] - m) * phi ** np.arange(1, PRED_LEN + 1)


def anchor(pred, h, phi=0.966):
    w = phi ** np.arange(1, PRED_LEN + 1)
    return w * damped_persistence(h, phi) + (1 - w) * np.asarray(pred)


BASELINES = {
    "damped persistence -> mean last 720": damped_persistence,
    "global mean": lambda h: np.full(PRED_LEN, h.mean()),
    "mean of last 168": lambda h: np.full(PRED_LEN, h[-168:].mean()),
    "hour-of-day mean, last 28 days": hour_of_day_mean,
    "median of last 720": lambda h: np.full(PRED_LEN, np.median(h[-720:])),
    "mean of last 720": lambda h: np.full(PRED_LEN, h[-720:].mean()),
    "seasonal naive (repeat last 168)": lambda h: h[-168:].copy(),
    "persistence (repeat last value)": lambda h: np.full(PRED_LEN, h[-1]),
}


EXPECTED_RMSE = {
    "damped persistence -> mean last 720": 107.31,
    "global mean": 109.15,
    "mean of last 168": 111.64,
    "hour-of-day mean, last 28 days": 113.12,
    "median of last 720": 114.67,
    "mean of last 720": 114.73,
    "seasonal naive (repeat last 168)": 157.85,
    "persistence (repeat last value)": 173.98,
}


def run_baselines(y: np.ndarray, origins: list[int]) -> pd.DataFrame:
    frames = []
    for name, fn in BASELINES.items():
        df = evaluate(lambda o, fn=fn: fn(y[:o]), y, origins)
        df.insert(0, "method", name)
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def summarise(per_block: pd.DataFrame, group: str = "method") -> pd.DataFrame:
    """mean ± std of each metric across blocks, plus mean rank across blocks (1 = best)."""
    df = per_block.copy()
    df["rank"] = df.groupby("block")["rmse"].rank()
    out = df.groupby(group).agg(
        rmse_mean=("rmse", "mean"),
        rmse_std=("rmse", "std"),
        mae_mean=("mae", "mean"),
        smape_mean=("smape", "mean"),
        mean_rank=("rank", "mean"),
    )
    return out.sort_values("rmse_mean").round(2)
