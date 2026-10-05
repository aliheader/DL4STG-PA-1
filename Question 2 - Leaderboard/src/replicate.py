import hashlib
import json

import numpy as np
import pandas as pd

from data import PRED_LEN, load_target
from study import CHECKPOINT_DIR, PRESETS
from train import CONFIGS, HP, predict, prepare_covariates, train_one
from validate import ES_BLOCKS, K_BLOCKS, anchor, block_origins, rmse

SHIFT = 8766


def fold(shift: int = SHIFT):
    """(y, eval_origins, es_origins, train_end) for the history cut `shift` steps earlier."""
    y = load_target()[:-shift]
    eval_origins = block_origins(len(y), K_BLOCKS)
    es_origins = block_origins(len(y), ES_BLOCKS, skip_last=K_BLOCKS)
    return y, eval_origins, es_origins, es_origins[0]


def _path(name: str, seed: int, preset: str, shift: int):
    blob = json.dumps(
        dict(preset=PRESETS[preset], hp=HP, cfg=CONFIGS[name], seed=seed, shift=shift),
        sort_keys=True,
    )
    ident = hashlib.sha1(blob.encode()).hexdigest()[:10]
    return CHECKPOINT_DIR / f"replication-{preset}-{name}-seed{seed}-{ident}.npz"


def run(name: str, seed: int, preset: str = "full", shift: int = SHIFT) -> np.ndarray:
    """Eval-week forecasts [12, 168] of one config/seed on the replication fold (trained once)."""
    path = _path(name, seed, preset, shift)
    if path.exists():
        print(f"loaded {path.name}")
        return np.load(path)["preds"]
    p = PRESETS[preset]
    y, eval_origins, es_origins, train_end = fold(shift)
    cfg = CONFIGS[name]
    cov = prepare_covariates(cfg, train_end)
    print(
        f"replication: training config {name}, seed {seed} (history cut {shift} steps earlier)"
    )
    model, info = train_one(
        cfg,
        seed,
        p["epochs"],
        p["stride"],
        y,
        cov,
        train_end,
        es_origins=es_origins,
        patience=p["patience"],
    )
    preds = np.stack([predict(model, y, cov, o, cfg) for o in eval_origins])
    CHECKPOINT_DIR.mkdir(exist_ok=True)
    np.savez(
        path, preds=preds, eval_origins=eval_origins, best_epoch=info["best_epoch"]
    )
    return preds


def summary(
    configs,
    seeds=(0, 1, 2),
    phi: float = 0.83,
    preset: str = "full",
    shift: int = SHIFT,
) -> pd.DataFrame:
    """Per config on the replication fold: mean single-seed RMSE, seed-ensemble RMSE raw and
    anchored, and the paired per-week ensemble difference against the FIRST config."""
    y, eval_origins, _, _ = fold(shift)
    truth = np.stack([y[o : o + PRED_LEN] for o in eval_origins])
    blocks = lambda p: np.array([rmse(truth[b], p[b]) for b in range(len(truth))])
    anchored = lambda p: np.stack(
        [anchor(p[b], y[:o], phi) for b, o in enumerate(eval_origins)]
    )
    rows, ref = [], None
    for name in configs:
        preds = [run(name, s, preset, shift) for s in seeds]
        ens = np.mean(preds, axis=0)
        e_raw, e_anc = blocks(ens), blocks(anchored(ens))
        ref = e_raw if ref is None else ref
        rows.append(
            dict(
                config=name,
                single_seed_rmse=np.mean([blocks(p).mean() for p in preds]),
                ensemble_rmse=e_raw.mean(),
                ensemble_anchored_rmse=e_anc.mean(),
                ensemble_diff_vs_first=(e_raw - ref).mean(),
                weeks_better_than_first=f"{(e_raw < ref).sum()}/{len(ref)}",
            )
        )
    return pd.DataFrame(rows).round(2)
