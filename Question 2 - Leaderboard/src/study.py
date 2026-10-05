import hashlib
import json
import time

import numpy as np
import pandas as pd
import torch

from autoformer import count_params
from data import PRED_LEN, ROOT, load_target
from train import (
    CONFIGS,
    HP,
    build_model,
    predict,
    prepare_covariates,
    summarise_ablation,
    train_one,
)
from validate import (
    BASELINES,
    ES_BLOCKS,
    K_BLOCKS,
    anchor,
    block_origins,
    mae,
    rmse,
    smape,
)

CHECKPOINT_DIR = ROOT / "checkpoints"

PRESETS = {
    "smoke": dict(configs=["A", "B", "C"], seeds=[0], epochs=2, stride=24, patience=3),
    "full": dict(
        configs=["A", "B", "C"], seeds=[0, 1, 2], epochs=12, stride=3, patience=3
    ),
}


class Study:
    def __init__(self, preset: str = "smoke"):
        if preset not in PRESETS:
            raise ValueError(f"preset must be one of {list(PRESETS)}")
        self.preset_name, self.preset = preset, PRESETS[preset]
        self.y = load_target()
        self.n = len(self.y)
        self.eval_origins = block_origins(self.n, K_BLOCKS)
        self.es_origins = block_origins(self.n, ES_BLOCKS, skip_last=K_BLOCKS)
        self.train_end = self.es_origins[0]
        self.records = {}
        self._cov = {}

    # ---------------------------------------------------------------------------------------
    def covariates(self, name: str, train_end=None) -> np.ndarray:
        train_end = self.train_end if train_end is None else train_end
        key = (name, train_end)
        if key not in self._cov:
            self._cov[key] = prepare_covariates(CONFIGS[name], train_end)
        return self._cov[key]

    def _identity(self, name: str, seed: int, **extra) -> str:
        blob = json.dumps(
            dict(preset=self.preset, hp=HP, cfg=CONFIGS[name], seed=seed, **extra),
            sort_keys=True,
        )
        return hashlib.sha1(blob.encode()).hexdigest()[:10]

    def run(self, name: str, seed: int):
        """Train (or load) one config/seed on the validation split; store its 12-week forecasts."""
        if (name, seed) in self.records:
            return self.records[(name, seed)]
        CHECKPOINT_DIR.mkdir(exist_ok=True)
        path = (
            CHECKPOINT_DIR
            / f"{self.preset_name}-{name}-seed{seed}-{self._identity(name, seed)}.pt"
        )
        if path.exists():
            record = torch.load(path, weights_only=False)
            print(f"loaded {path.name}")
        else:
            print(f"training config {name}, seed {seed} ({self.preset_name} preset)")
            cfg, cov = CONFIGS[name], self.covariates(name)
            start = time.time()
            model, info = train_one(
                cfg,
                seed,
                self.preset["epochs"],
                self.preset["stride"],
                self.y,
                cov,
                self.train_end,
                es_origins=self.es_origins,
                patience=self.preset["patience"],
            )
            preds = np.stack(
                [predict(model, self.y, cov, o, cfg) for o in self.eval_origins]
            )
            record = dict(
                config=name,
                seed=seed,
                preds=preds,
                P=count_params(model),
                E=info["epochs_run"],
                best_epoch=info["best_epoch"],
                history=info["history"],
                fit_seconds=time.time() - start,
                state=model.state_dict(),
            )
            torch.save(record, path)
        self.records[(name, seed)] = record
        return record

    def run_all(self, configs=None, seeds=None):
        for name in configs or self.preset["configs"]:
            for seed in seeds or self.preset["seeds"]:
                self.run(name, seed)

    def load_model(self, name: str, seed: int):
        model = build_model(CONFIGS[name])
        model.load_state_dict(self.run(name, seed)["state"])
        return model.eval()

    # ---------------------------------------------------------------------------------------
    def truth(self) -> np.ndarray:
        return np.stack(
            [self.y[o : o + PRED_LEN] for o in self.eval_origins]
        )  # [12, 168]

    def baseline_preds(self, method: str) -> np.ndarray:
        return np.stack([BASELINES[method](self.y[:o]) for o in self.eval_origins])

    def runs_table(self) -> pd.DataFrame:
        truth, rows = self.truth(), []
        for (name, seed), r in self.records.items():
            for b, o in enumerate(self.eval_origins):
                rows.append(
                    dict(
                        config=name,
                        seed=seed,
                        block=b,
                        origin=o,
                        rmse=rmse(truth[b], r["preds"][b]),
                        mae=mae(truth[b], r["preds"][b]),
                        smape=smape(truth[b], r["preds"][b]),
                        P=r["P"],
                        E=r["E"],
                        best_epoch=r["best_epoch"],
                    )
                )
        return pd.DataFrame(rows)

    def ablation_summary(self) -> pd.DataFrame:
        return summarise_ablation(self.runs_table(), self.y, self.eval_origins)

    def horizon_rmse(
        self, baselines=("damped persistence -> mean last 720",)
    ) -> pd.DataFrame:
        """RMSE at each of the 168 forecast steps, pooled over weeks and seeds."""
        truth, curves = self.truth(), {}
        for name in dict.fromkeys(n for n, _ in self.records):
            preds = np.stack(
                [r["preds"] for (n, _), r in self.records.items() if n == name]
            )
            curves[f"config {name}"] = np.sqrt(((preds - truth) ** 2).mean(axis=(0, 1)))
        for method in baselines:
            curves[f"[baseline] {method}"] = np.sqrt(
                ((self.baseline_preds(method) - truth) ** 2).mean(axis=0)
            )
        return pd.DataFrame(curves, index=pd.RangeIndex(1, PRED_LEN + 1, name="step"))

    def _chosen_seeds(self, choice: dict) -> list:

        name = choice.get("config")
        seeds = choice.get("seeds") or (
            [choice["seed"]] if choice.get("seed") is not None else None
        )
        if name is None or not seeds:
            raise ValueError(
                "Fill in `choice` (config and seed/seeds) from Outputs 5.1-5.5 first."
            )
        for seed in seeds:
            if (name, seed) not in self.records:
                raise ValueError(
                    f"({name}, seed {seed}) was not trained in this study."
                )
        return list(seeds)

    def _declared_cost(self, name: str, seeds) -> tuple:

        recs = [self.records[(name, s)] for s in seeds]
        return sum(r["P"] for r in recs), sum(r["E"] + r["best_epoch"] for r in recs)

    def candidate_preds(
        self, name: str, seeds, anchored: bool, phi: float = 0.966
    ) -> np.ndarray:
        """12-week validation forecasts of a seed ensemble (mean over seeds), optionally anchored."""
        preds = np.mean([self.run(name, s)["preds"] for s in seeds], axis=0)
        if anchored:
            preds = np.stack(
                [
                    anchor(preds[b], self.y[:o], phi)
                    for b, o in enumerate(self.eval_origins)
                ]
            )
        return preds

    def _candidate_row(self, name: str, group, anchored: bool, phi: float) -> dict:
        truth = self.truth()
        block_rmse = lambda p: np.array(
            [rmse(truth[b], p[b]) for b in range(len(truth))]
        )
        trained = sorted(s for n, s in self.records if n == name)
        single = np.mean(
            [block_rmse(self.candidate_preds(name, [s], False)) for s in trained],
            axis=0,
        )
        p = self.candidate_preds(name, group, anchored, phi)
        r = block_rmse(p)
        P, E = self._declared_cost(name, group)
        return dict(
            config=name,
            seeds=",".join(map(str, group)),
            anchored=anchored,
            rmse_mean=r.mean(),
            rmse_std_blocks=r.std(ddof=1),
            mae_mean=np.mean([mae(truth[b], p[b]) for b in range(len(truth))]),
            smape_mean=np.mean([smape(truth[b], p[b]) for b in range(len(truth))]),
            diff_vs_single_mean=(r - single).mean(),
            diff_vs_single_std=(r - single).std(ddof=1),
            P=P,
            declared_E=E,
        )

    def final_candidates(
        self, name: str, seeds=None, phi: float = 0.966
    ) -> pd.DataFrame:

        seeds = list(seeds or self.preset["seeds"])
        groups = [[s] for s in seeds] + ([seeds] if len(seeds) > 1 else [])
        return pd.DataFrame(
            [
                self._candidate_row(name, g, anchored, phi)
                for g in groups
                for anchored in (False, True)
            ]
        ).round(2)

    def selection_summary(self, choice: dict, phi: float = 0.966) -> pd.DataFrame:

        name, seeds = choice["config"], self._chosen_seeds(choice)
        phi = choice.get("phi", phi)
        row = self._candidate_row(name, seeds, bool(choice.get("anchor", False)), phi)
        if choice.get("refit", True):
            row["refit_epochs"] = ",".join(
                str(self.records[(name, s)]["best_epoch"]) for s in seeds
            )
        else:
            row["declared_E"] = sum(self.records[(name, s)]["E"] for s in seeds)
            row["refit_epochs"] = "none (validated models)"
        return pd.DataFrame([row]).round(2)

    # ---------------------------------------------------------------------------------------
    def refit_one(self, name: str, seed: int):

        r = self.run(name, seed)
        epochs = r["best_epoch"]
        cfg, cov = CONFIGS[name], self.covariates(name, train_end=self.n)
        path = CHECKPOINT_DIR / (
            f"{self.preset_name}-refit-{name}-seed{seed}-"
            f"{self._identity(name, seed, refit_epochs=epochs, schedule='matched')}.pt"
        )
        model = build_model(cfg)
        if path.exists():
            saved = torch.load(path, weights_only=False)
            model.load_state_dict(saved["state"])
            refit_epochs = saved["epochs_run"]
            print(f"loaded {path.name}")
        else:
            print(
                f"refitting config {name}, seed {seed} on all {self.n} observations "
                f"for {epochs} epochs"
            )
            model, info = train_one(
                cfg,
                seed,
                epochs,
                self.preset["stride"],
                self.y,
                cov,
                train_end=self.n,
                schedule_epochs=self.preset["epochs"],
            )
            refit_epochs = info["epochs_run"]
            torch.save(dict(state=model.state_dict(), epochs_run=refit_epochs), path)
        forecast = predict(model, self.y, cov, origin=self.n, cfg=cfg)
        return forecast, count_params(model), r["E"] + refit_epochs

    def anneal_refit_one(self, name: str, seed: int):

        r = self.run(name, seed)
        epochs = r["best_epoch"]
        cfg, cov = CONFIGS[name], self.covariates(name, train_end=self.n)
        path = CHECKPOINT_DIR / (
            f"{self.preset_name}-refit-{name}-seed{seed}-"
            f"{self._identity(name, seed, refit_epochs=epochs)}.pt"
        )
        model = build_model(cfg)
        if path.exists():
            saved = torch.load(path, weights_only=False)
            model.load_state_dict(saved["state"])
            refit_epochs = saved["epochs_run"]
            print(f"loaded {path.name}")
        else:
            print(
                f"anneal refit: config {name}, seed {seed} on all {self.n} observations "
                f"for {epochs} epochs"
            )
            model, info = train_one(
                cfg, seed, epochs, self.preset["stride"], self.y, cov, train_end=self.n
            )
            refit_epochs = info["epochs_run"]
            torch.save(dict(state=model.state_dict(), epochs_run=refit_epochs), path)
        forecast = predict(model, self.y, cov, origin=self.n, cfg=cfg)
        return forecast, count_params(model), r["E"] + refit_epochs

    def validated_forecast(self, name: str, seed: int):

        r = self.run(name, seed)
        forecast = predict(
            self.load_model(name, seed),
            self.y,
            self.covariates(name),
            origin=self.n,
            cfg=CONFIGS[name],
        )
        return forecast, r["P"], r["E"]

    RECIPES = {
        "snapshot": ("validated",),
        "anneal": ("anneal",),
        "blend": ("validated", "anneal"),
        "matched": ("matched",),
    }

    def refit(self, choice: dict, phi: float = 0.966):

        name, seeds = choice["config"], self._chosen_seeds(choice)
        recipe = choice.get("recipe") or (
            "matched" if choice.get("refit", True) else "snapshot"
        )
        member_fn = dict(
            validated=self.validated_forecast,
            anneal=self.anneal_refit_one,
            matched=self.refit_one,
        )
        forecasts, P, E = [], 0, 0
        for s in seeds:
            val_epochs = self.run(name, s)["E"]
            E += val_epochs
            for kind in self.RECIPES[recipe]:
                f, p, e = member_fn[kind](name, s)
                forecasts.append(f)
                P, E = P + p, E + (
                    e - val_epochs
                )  # e includes val_epochs; count them once
        forecast = np.mean(forecasts, axis=0)
        if choice.get("anchor", False):
            forecast = anchor(forecast, self.y, choice.get("phi", phi))
        return forecast, P, E
