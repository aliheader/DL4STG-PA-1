import argparse
import copy
import random

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

from autoformer import Autoformer, count_params
from data import (
    FEATURES,
    PRED_LEN,
    RESULTS_DIR,
    WindowDataset,
    load_covariates,
    load_target,
    make_window,
    standardise_covariates,
)
from validate import (
    ES_BLOCKS,
    K_BLOCKS,
    block_origins,
    evaluate,
    paired_diff,
    BASELINES,
)

HP = dict(
    L_in=336,
    label_len=168,
    pred_len=PRED_LEN,
    d=32,
    heads=4,
    e_layers=1,
    d_layers=1,
    kernel=25,
    dropout=0.1,
    lr=1e-3,
    weight_decay=1e-4,
    batch=32,
)

# The ablation. Only the covariate inputs differ between configs.
CONFIGS = {
    "A": dict(features=[], past_cov=False, future_cov=False),
    "B": dict(features=FEATURES, past_cov=True, future_cov=False),
    "C": dict(features=FEATURES, past_cov=True, future_cov=True),
    "C_subset": dict(
        features=["feature_D", "feature_H", "feature_A", "feature_I"],
        past_cov=True,
        future_cov=True,
    ),
    "C_cal": dict(
        features=FEATURES + ["seasonal_year"], past_cov=True, future_cov=True
    ),
    "C_small": dict(
        features=FEATURES, past_cov=True, future_cov=True, hp_overrides=dict(d=16)
    ),
    "C_small_wl": dict(
        features=FEATURES,
        past_cov=True,
        future_cov=True,
        hp_overrides=dict(d=16, loss="scaled"),
    ),
    "C_small_wlr": dict(
        features=FEATURES,
        past_cov=True,
        future_cov=True,
        hp_overrides=dict(d=16, loss="scaled", dropout=0.2, weight_decay=1e-3),
    ),
    "C_small_wlrd": dict(
        features=FEATURES,
        past_cov=True,
        future_cov=True,
        hp_overrides=dict(
            d=16, loss="scaled", dropout=0.2, weight_decay=1e-3, damped_trend=True
        ),
    ),
    "C_small_d": dict(
        features=FEATURES,
        past_cov=True,
        future_cov=True,
        hp_overrides=dict(d=16, damped_trend=True),
    ),
}


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def prepare_covariates(cfg: dict, train_end: int) -> np.ndarray:
    """Selected covariate columns, standardised with rows before train_end only."""
    if not cfg["features"]:
        return np.zeros((len(load_target()) + PRED_LEN, 0))
    cov = load_covariates(cfg["features"], y=load_target(), train_end=train_end)
    return standardise_covariates(cov, train_end)


def hp_for(cfg: dict) -> dict:
    """HP, with per-config overrides (e.g. a smaller `d`) layered on top -- lets a config test a
    capacity change without touching HP itself, so every other config's checkpoints stay valid.
    """
    return {**HP, **cfg.get("hp_overrides", {})}


def build_model(cfg: dict) -> Autoformer:
    n_feat = len(cfg["features"])
    hp = hp_for(cfg)
    return Autoformer(
        enc_in=1 + (n_feat if cfg["past_cov"] else 0),
        dec_in=1 + (n_feat if cfg["future_cov"] else 0),
        d=hp["d"],
        heads=hp["heads"],
        e_layers=hp["e_layers"],
        d_layers=hp["d_layers"],
        kernel=hp["kernel"],
        label_len=hp["label_len"],
        pred_len=hp["pred_len"],
        dropout=hp["dropout"],
        damped_trend=hp.get("damped_trend", False),
    )


def window_kwargs(cfg: dict) -> dict:
    hp = hp_for(cfg)
    return dict(
        L_in=hp["L_in"],
        label_len=hp["label_len"],
        pred_len=hp["pred_len"],
        past_cov=cfg["past_cov"],
        future_cov=cfg["future_cov"],
    )


@torch.no_grad()
def predict(model, y, cov, origin, cfg) -> np.ndarray:
    """168-step forecast made at `origin`, in ORIGINAL units, clamped at 0."""
    model.eval()
    x_enc, x_dec, mu, sd = make_window(y, cov, origin, **window_kwargs(cfg))
    as_t = lambda a: torch.as_tensor(a, dtype=torch.float32).unsqueeze(0)
    out = model(as_t(x_enc), as_t(x_dec)).squeeze(0).numpy()
    return np.clip(out * sd + mu, 0.0, None)


def train_one(
    cfg: dict,
    seed: int,
    epochs: int,
    stride: int,
    y,
    cov,
    train_end: int,
    es_origins=None,
    patience: int = 3,
    verbose: bool = True,
    schedule_epochs=None,
):

    set_seed(seed)
    model = build_model(cfg)
    hp = hp_for(cfg)
    origins = range(hp["L_in"], train_end - hp["pred_len"] + 1, stride)
    ds = WindowDataset(y, cov, origins, **window_kwargs(cfg))
    gen = torch.Generator().manual_seed(seed)
    loader = DataLoader(ds, batch_size=hp["batch"], shuffle=True, generator=gen)

    opt = torch.optim.AdamW(
        model.parameters(), lr=hp["lr"], weight_decay=hp["weight_decay"]
    )
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt,
        max_lr=hp["lr"],
        epochs=schedule_epochs or epochs,
        steps_per_epoch=len(loader),
    )
    loss_fn = nn.MSELoss()  # RMSE is the metric, so MSE is the matching loss

    scaled = hp.get("loss", "mse") == "scaled"
    if scaled:
        mean_sd2 = float(
            np.mean([max(y[o - hp["L_in"] : o].std(), 1.0) ** 2 for o in origins])
        )

    best, best_state, best_epoch, bad, history = float("inf"), None, 0, 0, []
    for epoch in range(1, epochs + 1):
        model.train()
        total = 0.0
        for x_enc, x_dec, target, scale in loader:
            opt.zero_grad()
            pred = model(x_enc, x_dec)
            if scaled:
                loss = ((pred - target) ** 2).mean(1).mul(scale**2).mean() / mean_sd2
            else:
                loss = loss_fn(pred, target)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            total += loss.item() * len(target)
        train_loss = total / len(ds)

        row = dict(epoch=epoch, train_loss=train_loss)
        if es_origins is not None:
            es = evaluate(lambda o: predict(model, y, cov, o, cfg), y, es_origins)
            row["es_rmse"] = es["rmse"].mean()
            if row["es_rmse"] < best:
                best, best_epoch, bad = row["es_rmse"], epoch, 0
                best_state = copy.deepcopy(model.state_dict())
            else:
                bad += 1
        history.append(row)
        if verbose:
            print(
                "   "
                + "  ".join(
                    f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
                    for k, v in row.items()
                ),
                flush=True,
            )
        if es_origins is not None and bad >= patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    else:
        best_epoch = epoch
    return model, dict(epochs_run=epoch, best_epoch=best_epoch, history=history)


def summarise_ablation(runs: pd.DataFrame, y, eval_origins) -> pd.DataFrame:

    rows = []
    for name, g in runs.groupby("config", sort=False):
        seed_means = g.groupby("seed")["rmse"].mean()
        row = dict(
            config=name,
            rmse_mean=g["rmse"].mean(),
            rmse_std_blocks=g["rmse"].std(),
            rmse_std_seeds=seed_means.std(ddof=1) if len(seed_means) > 1 else np.nan,
            mae_mean=g["mae"].mean(),
            smape_mean=g["smape"].mean(),
            P=int(g["P"].iloc[0]),
            E_mean=g["E"].mean(),
            best_epoch_mean=g["best_epoch"].mean(),
        )
        if "A" in set(runs.config) and name != "A":
            d_mean, d_std, n_pairs = paired_diff(
                g, runs[runs.config == "A"], keys=("seed", "block")
            )
            row.update(diff_vs_A_mean=d_mean, diff_vs_A_std=d_std, n_pairs=n_pairs)
        rows.append(row)
    for ref in ("damped persistence -> mean last 720", "mean of last 720"):
        b = evaluate(lambda o: BASELINES[ref](y[:o]), y, eval_origins)
        rows.append(
            dict(
                config=f"[baseline] {ref}",
                rmse_mean=b["rmse"].mean(),
                rmse_std_blocks=b["rmse"].std(),
                mae_mean=b["mae"].mean(),
                smape_mean=b["smape"].mean(),
            )
        )
    return pd.DataFrame(rows).round(2)
