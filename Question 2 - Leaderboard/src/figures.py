import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from autoformer import SeriesDecomp
from data import FEATURES, PRED_LEN, acf


def data_profile(y: np.ndarray) -> pd.DataFrame:
    q = np.quantile(y, [0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99])
    s = pd.Series(y)
    rows = [
        ("observations", len(y)),
        ("mean", y.mean()),
        ("std", y.std()),
        ("min", y.min()),
        ("max", y.max()),
        ("exact zeros", int((y == 0).sum())),
        ("skewness", s.skew()),
        ("excess kurtosis", s.kurt()),
    ]
    rows += [(f"quantile {p}", v) for p, v in zip((1, 5, 25, 50, 75, 95, 99), q)]
    rows += [(f"ACF lag {k}", acf(y, k)) for k in (1, 24, 48, 72, 168, 336, 720, 8766)]
    return pd.DataFrame(rows, columns=["statistic", "value"])


def acf_and_recent(y: np.ndarray, max_lag: int = 200):
    lags = np.arange(1, max_lag + 1)
    fig, (a, b) = plt.subplots(1, 2, figsize=(13, 3.6))
    a.bar(lags, [acf(y, k) for k in lags], width=1.0)
    for k in (24, 48, 72, 168):
        a.axvline(k, color="grey", ls=":", lw=0.8)
    a.set(title="Autocorrelation of the target", xlabel="lag (steps)", ylabel="ACF")
    b.plot(np.arange(len(y) - 2000, len(y)) + 1, y[-2000:], lw=0.7)
    b.set(title="Last 2,000 observations", xlabel="time_idx", ylabel="value")
    fig.tight_layout()


def periodicity(y: np.ndarray) -> pd.DataFrame:
    """Share of variance explained by the period-P mean profile, for candidate periods."""
    rows = []
    for p in (12, 24, 48, 168, 720, 8766):
        phase = np.arange(len(y)) % p
        profile = pd.Series(y).groupby(phase).transform("mean").to_numpy()
        rows.append(
            dict(
                period=p,
                variance_share_pct=100 * profile.var() / y.var(),
                profile_peak_to_trough=profile.max() - profile.min(),
            )
        )
    return pd.DataFrame(rows)


def covariate_correlations(
    y: np.ndarray, cov: np.ndarray, columns=FEATURES
) -> pd.DataFrame:
    hist = pd.DataFrame(cov[: len(y)], columns=columns)
    target = pd.Series(y)
    out = pd.DataFrame(
        dict(
            feature=columns,
            pearson=[hist[c].corr(target) for c in columns],
            rank=[hist[c].rank().corr(target.rank()) for c in columns],
        )
    )
    return out.reindex(
        out["rank"].abs().sort_values(ascending=False).index
    ).reset_index(drop=True)


def baseline_blocks(per_block: pd.DataFrame, methods):
    fig, ax = plt.subplots(figsize=(11, 3.6))
    for m in methods:
        d = per_block[per_block.method == m]
        ax.plot(d.block + 1, d.rmse, marker="o", label=m)
    ax.set(
        title="RMSE of each evaluation week — the week matters more than the method",
        xlabel="evaluation week (1 = oldest)",
        ylabel="RMSE",
    )
    ax.legend(fontsize=8)
    fig.tight_layout()


def decomposition(y: np.ndarray, origin: int, L: int = 336, kernels=(25, 169)):
    window = torch.as_tensor(y[origin - L : origin], dtype=torch.float32).view(1, L, 1)
    t = np.arange(origin - L, origin) + 1
    fig, axes = plt.subplots(
        len(kernels), 2, figsize=(13, 3.0 * len(kernels)), squeeze=False
    )
    for row, k in enumerate(kernels):
        seasonal, trend = (part.view(-1).numpy() for part in SeriesDecomp(k)(window))
        axes[row, 0].plot(t, window.view(-1), lw=0.7, color="grey", label="input")
        axes[row, 0].plot(t, trend, lw=1.8, label=f"trend (kernel {k})")
        axes[row, 0].legend(fontsize=8)
        axes[row, 1].plot(t, seasonal, lw=0.8, color="tab:green")
        axes[row, 1].axhline(0, color="grey", lw=0.5)
        axes[row, 0].set_title(f"kernel {k}: input and trend")
        axes[row, 1].set_title(f"kernel {k}: seasonal = input - trend")
    fig.tight_layout()


def delay_scores_raw(window: np.ndarray) -> np.ndarray:
    """The same FFT delay score AutoCorrelation uses, on a raw series (no learned projections)."""
    x = torch.as_tensor(window - window.mean(), dtype=torch.float32)
    return torch.fft.irfft(
        torch.fft.rfft(x) * torch.fft.rfft(x).conj(), n=len(x)
    ).numpy()


def delay_selection(
    y: np.ndarray, origins, L: int = 336, factor: float = 2.0
) -> pd.DataFrame:

    k = max(1, min(int(factor * math.log(L)), L))
    counts = pd.Series(0, index=range(L))
    for o in origins:
        scores = delay_scores_raw(y[o - L : o])
        counts[np.argsort(scores)[::-1][:k]] += 1
    out = (
        counts[counts > 0]
        .sort_values(ascending=False)
        .rename("windows_selecting")
        .reset_index()
    )
    out = out.rename(columns={"index": "delay"})
    out["delay_mod_24"] = out["delay"] % 24
    out["share_of_windows"] = out["windows_selecting"] / len(origins)
    return out


def delay_score_plot(y: np.ndarray, origin: int, L: int = 336):
    scores = delay_scores_raw(y[origin - L : origin])
    fig, ax = plt.subplots(figsize=(11, 3.2))
    ax.plot(np.arange(L), scores / scores[0], lw=1)
    for k in range(24, L, 24):
        ax.axvline(k, color="grey", ls=":", lw=0.6)
    ax.set(
        title=f"Delay scores for the encoder window before week starting {origin + 1} "
        f"(dotted = multiples of 24)",
        xlabel="delay",
        ylabel="score / score at delay 0",
    )
    fig.tight_layout()


def paired_differences(runs: pd.DataFrame, reference: str = "A"):
    ref = runs[runs.config == reference][["seed", "block", "rmse"]]
    others = [c for c in runs.config.unique() if c != reference]
    fig, ax = plt.subplots(figsize=(11, 3.6))
    for i, c in enumerate(others):
        m = runs[runs.config == c].merge(
            ref, on=["seed", "block"], suffixes=("", "_ref")
        )
        jitter = (i - (len(others) - 1) / 2) * 0.15
        ax.scatter(
            m.block + 1 + jitter, m.rmse - m.rmse_ref, s=18, label=f"{c} - {reference}"
        )
    ax.axhline(0, color="black", lw=0.8)
    ax.set(
        title=f"Paired RMSE difference vs config {reference} (below 0 = better), "
        "one dot per seed",
        xlabel="evaluation week",
        ylabel="RMSE difference",
    )
    ax.legend(fontsize=8)
    fig.tight_layout()


def horizon_curve(curves: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(11, 3.6))
    for c in curves.columns:
        ax.plot(
            curves.index,
            curves[c],
            lw=1.4 if c.startswith("config") else 1.0,
            ls="-" if c.startswith("config") else "--",
            label=c,
        )
    ax.set(
        title="RMSE at each forecast step (pooled over weeks and seeds)",
        xlabel="steps ahead",
        ylabel="RMSE",
    )
    ax.legend(fontsize=8)
    fig.tight_layout()


def forecast_cases(
    study, configs, seed: int, blocks, baseline="damped persistence -> mean last 720"
):
    truth = study.truth()
    base = study.baseline_preds(baseline)
    fig, axes = plt.subplots(
        len(blocks), 1, figsize=(12, 2.8 * len(blocks)), squeeze=False
    )
    for ax, b in zip(axes[:, 0], blocks):
        o = study.eval_origins[b]
        ctx = np.arange(o - 168, o) + 1
        hor = np.arange(o, o + PRED_LEN) + 1
        ax.plot(ctx, study.y[o - 168 : o], color="grey", lw=0.7, label="history")
        ax.plot(hor, truth[b], color="black", lw=0.9, label="actual")
        ax.plot(hor, base[b], ls="--", lw=1.0, label="damped persistence")
        for c in configs:
            if (c, seed) in study.records:
                ax.plot(
                    hor,
                    study.records[(c, seed)]["preds"][b],
                    lw=1.3,
                    label=f"config {c}",
                )
        ax.set_title(
            f"evaluation week {b + 1} (time_idx {o + 1}..{o + PRED_LEN})", fontsize=9
        )
    axes[0, 0].legend(fontsize=8, ncol=3)
    fig.tight_layout()


def submission(y: np.ndarray, forecast: np.ndarray, label: str):
    n = len(y)
    fig, ax = plt.subplots(figsize=(12, 3.6))
    ax.plot(np.arange(n - 500, n) + 1, y[-500:], lw=0.8, label="history (last 500)")
    ax.plot(np.arange(n, n + PRED_LEN) + 1, forecast, lw=1.5, label=label)
    ax.axvline(n + 1, color="grey", ls="--", lw=0.8)
    ax.set(
        title="Submitted forecast appended to history — the join must look natural",
        xlabel="time_idx",
        ylabel="value",
    )
    ax.legend(fontsize=8)
    fig.tight_layout()
