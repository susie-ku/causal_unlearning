"""Extended experiment suite for NeurIPS submission.

Runs:
  1. Multi-seed sweep (seeds 0, 42, 123) for baseline, oracle, and all λ values
  2. Competing baselines: GRL and intervened fine-tuning
  3. Extended λ sweep  (λ ∈ {0.0, 0.1, 0.2, 0.5, 1.0, 2.0})
  4. β (locality) ablation  (β ∈ {0.0, 1e-4, 1e-3, 1e-2})
  5. Epochs ablation  (T ∈ {1, 2, 5} at λ=0.5)
  6. ρ (spurious correlation) sweep  (ρ ∈ {0.7, 0.8, 0.9, 0.95})

Outputs are written to:
  artifacts/multi_seed/
  artifacts/baselines/
  artifacts/ablation_lambda/
  artifacts/ablation_beta/
  artifacts/ablation_epochs/
  artifacts/ablation_rho/

Each directory contains a summary.json conforming to the same schema as
artifacts/default/summary.json so downstream plotting code can consume it.
"""
from __future__ import annotations

import sys
import time
from copy import deepcopy
from pathlib import Path

# Allow running as `python experiments/run_extended.py` from project root
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from causal_unlearning.baselines import train_grl_unlearning, train_intervened_finetuning
from causal_unlearning.config import DataConfig, RunConfig, TrainConfig, UnlearningConfig
from causal_unlearning.datasets import build_dataloaders
from causal_unlearning.experiments import run_full_pipeline
from causal_unlearning.metrics import evaluate_model
from causal_unlearning.models import build_model, SmallCNN
from causal_unlearning.training import (
    load_checkpoint,
    resolve_device,
    save_checkpoint,
    train_supervised,
    train_unlearning,
)
from causal_unlearning.utils import ensure_dir, save_json, set_seed


# ── Helpers ────────────────────────────────────────────────────────────────

def _make_run_config(
    output_dir: str,
    seed: int = 42,
    rho: float = 0.9,
    lambda_values: tuple[float, ...] = (0.0, 0.1, 0.5, 1.0),
    unlearn_epochs: int = 2,
    beta: float = 1e-3,
) -> RunConfig:
    return RunConfig(
        output_dir=output_dir,
        data=DataConfig(seed=seed, observational_correlation=rho, num_workers=0),
        baseline=TrainConfig(epochs=5, lr=1e-3, weight_decay=1e-4),
        oracle=TrainConfig(epochs=5, lr=1e-3, weight_decay=1e-4),
        unlearning=UnlearningConfig(epochs=unlearn_epochs, lr=5e-4, weight_decay=0.0, lambda_locality=beta),
        lambda_ce_values=lambda_values,
    )


def _load_model(path: str | Path, device: str = "auto"):
    payload = load_checkpoint(path, device=resolve_device(device))
    model = build_model(payload["model_name"])
    model.load_state_dict(payload["state_dict"])
    model.to(resolve_device(device))
    return model


def _run_and_time(label: str, fn, *args, **kwargs):
    t0 = time.perf_counter()
    print(f"\n{'='*60}\n[START] {label}\n{'='*60}", flush=True)
    result = fn(*args, **kwargs)
    elapsed = time.perf_counter() - t0
    print(f"[DONE]  {label}  ({elapsed:.1f}s)", flush=True)
    return result


# ── 1. Multi-seed sweep ─────────────────────────────────────────────────────

def run_multi_seed(seeds=(0, 42, 123)):
    results = {}
    for seed in seeds:
        out = f"artifacts/multi_seed/seed_{seed}"
        cfg = _make_run_config(out, seed=seed)
        summary = _run_and_time(f"multi_seed seed={seed}", run_full_pipeline, cfg)
        results[seed] = summary

    # Aggregate: compute mean ± std per model across seeds
    aggregate = _aggregate_multi_seed(results, seeds)
    save_json("artifacts/multi_seed/aggregate.json", aggregate)
    print("\nMulti-seed aggregate saved to artifacts/multi_seed/aggregate.json", flush=True)
    return aggregate


def _aggregate_multi_seed(results: dict, seeds: tuple) -> dict:
    import statistics

    model_keys = ["baseline", "oracle"] + [
        f"unlearn_lambda_{_tag(lam)}"
        for lam in (0.0, 0.1, 0.5, 1.0)
    ]
    metric_keys = ["observational_accuracy", "intervened_accuracy", "causal_effect_proxy"]

    aggregate: dict = {"seeds": list(seeds), "models": {}}

    for model_key in model_keys:
        values: dict[str, list[float]] = {k: [] for k in metric_keys}
        for seed in seeds:
            seed_summary = results[seed]
            if model_key in ("baseline", "oracle"):
                metrics = seed_summary[model_key]["metrics"]
            else:
                run = next(
                    r for r in seed_summary["unlearning_runs"] if r["name"] == model_key
                )
                metrics = run["metrics"]
            for k in metric_keys:
                if k in metrics:
                    values[k].append(metrics[k])

        agg_entry: dict[str, dict] = {}
        for k, vals in values.items():
            if vals:
                agg_entry[k] = {
                    "mean": statistics.mean(vals),
                    "std": statistics.stdev(vals) if len(vals) > 1 else 0.0,
                    "values": vals,
                }
        aggregate["models"][model_key] = agg_entry

    return aggregate


def _tag(lam: float) -> str:
    return str(lam).replace(".", "p").replace("p0", "0") if lam != 0.0 else "0"


# ── 2. Competing baselines ──────────────────────────────────────────────────

def run_baselines(seed: int = 42):
    out_dir = Path("artifacts/baselines")
    ensure_dir(out_dir)

    data_cfg = DataConfig(seed=seed, num_workers=0)
    loaders = build_dataloaders(data_cfg)
    device_str = "auto"
    device = resolve_device(device_str)

    # Load existing baseline and oracle checkpoints (already trained)
    baseline_path = "artifacts/default/checkpoints/baseline.pt"
    oracle_path = "artifacts/default/checkpoints/oracle.pt"
    baseline_model = _load_model(baseline_path, device_str)
    oracle_model = _load_model(oracle_path, device_str)

    unlearn_cfg = UnlearningConfig(epochs=2, lr=5e-4, weight_decay=0.0, lambda_ce=1.0, lambda_locality=1e-3)

    results = {}

    # --- GRL baseline ---
    print("\n[GRL baseline]", flush=True)
    grl_model, grl_history = train_grl_unlearning(
        deepcopy(baseline_model),
        loaders.observational_train,
        loaders,
        unlearn_cfg,
        oracle_model=oracle_model,
        grl_alpha=1.0,
    )
    # Wrap for evaluate_model
    import torch.nn as nn
    class _W(nn.Module):
        def __init__(self, m): super().__init__(); self.m = m
        def forward(self, x): return self.m(x)
    grl_wrapped = _W(grl_model).to(device)
    grl_metrics = evaluate_model(grl_wrapped, loaders.as_dict(), device, oracle_model=oracle_model)
    results["grl"] = {"history": grl_history, "metrics": grl_metrics, "name": "grl"}
    save_json(out_dir / "grl.json", results["grl"])
    print(f"  GRL metrics: {grl_metrics}", flush=True)

    # --- Naive intervened fine-tuning baseline ---
    print("\n[Intervened FT baseline]", flush=True)
    ift_model, ift_history = train_intervened_finetuning(
        deepcopy(baseline_model),
        loaders.intervened_train,
        loaders,
        unlearn_cfg,
        oracle_model=oracle_model,
    )
    ift_metrics = evaluate_model(ift_model, loaders.as_dict(), device, oracle_model=oracle_model)
    results["intervened_ft"] = {"history": ift_history, "metrics": ift_metrics, "name": "intervened_ft"}
    save_json(out_dir / "intervened_ft.json", results["intervened_ft"])
    print(f"  Intervened FT metrics: {ift_metrics}", flush=True)

    save_json(out_dir / "summary.json", results)
    return results


# ── 3. Extended λ sweep ─────────────────────────────────────────────────────

def run_lambda_ablation(seed: int = 42):
    lambda_values = (0.0, 0.1, 0.2, 0.5, 1.0, 2.0)
    cfg = _make_run_config(
        "artifacts/ablation_lambda",
        seed=seed,
        lambda_values=lambda_values,
    )
    return _run_and_time("extended lambda sweep", run_full_pipeline, cfg)


# ── 4. β (locality) ablation ────────────────────────────────────────────────

def run_beta_ablation(seed: int = 42):
    beta_values = [0.0, 1e-4, 1e-3, 1e-2]
    out_dir = Path("artifacts/ablation_beta")
    ensure_dir(out_dir)

    data_cfg = DataConfig(seed=seed, num_workers=0)
    loaders = build_dataloaders(data_cfg)
    device = resolve_device("auto")

    baseline_model = _load_model("artifacts/default/checkpoints/baseline.pt")
    oracle_model = _load_model("artifacts/default/checkpoints/oracle.pt")

    results = []
    for beta in beta_values:
        print(f"\n[beta ablation] beta={beta}", flush=True)
        cfg = UnlearningConfig(epochs=2, lr=5e-4, weight_decay=0.0, lambda_ce=0.5, lambda_locality=beta)
        model = build_model()
        model.load_state_dict(deepcopy(baseline_model.state_dict()))
        model, history = train_unlearning(model, loaders.observational_train, loaders, cfg, oracle_model=oracle_model)
        metrics = evaluate_model(model, loaders.as_dict(), device, oracle_model=oracle_model)
        tag = str(beta).replace(".", "p").replace("-", "n")
        entry = {"beta": beta, "history": history, "metrics": metrics}
        results.append(entry)
        save_json(out_dir / f"beta_{tag}.json", entry)
        print(f"  metrics: {metrics}", flush=True)

    save_json(out_dir / "summary.json", {"beta_values": beta_values, "runs": results})
    return results


# ── 5. Epochs ablation ──────────────────────────────────────────────────────

def run_epochs_ablation(seed: int = 42):
    epoch_values = [1, 2, 5, 10]
    out_dir = Path("artifacts/ablation_epochs")
    ensure_dir(out_dir)

    data_cfg = DataConfig(seed=seed, num_workers=0)
    loaders = build_dataloaders(data_cfg)
    device = resolve_device("auto")

    baseline_model = _load_model("artifacts/default/checkpoints/baseline.pt")
    oracle_model = _load_model("artifacts/default/checkpoints/oracle.pt")

    results = []
    for n_epochs in epoch_values:
        print(f"\n[epochs ablation] epochs={n_epochs}", flush=True)
        cfg = UnlearningConfig(epochs=n_epochs, lr=5e-4, weight_decay=0.0, lambda_ce=0.5, lambda_locality=1e-3)
        model = build_model()
        model.load_state_dict(deepcopy(baseline_model.state_dict()))
        model, history = train_unlearning(model, loaders.observational_train, loaders, cfg, oracle_model=oracle_model)
        metrics = evaluate_model(model, loaders.as_dict(), device, oracle_model=oracle_model)
        entry = {"epochs": n_epochs, "history": history, "metrics": metrics}
        results.append(entry)
        save_json(out_dir / f"epochs_{n_epochs}.json", entry)
        print(f"  metrics: {metrics}", flush=True)

    save_json(out_dir / "summary.json", {"epoch_values": epoch_values, "runs": results})
    return results


# ── 6. ρ sweep ──────────────────────────────────────────────────────────────

def run_rho_sweep(seed: int = 42):
    rho_values = [0.7, 0.8, 0.9, 0.95]
    results = {}
    for rho in rho_values:
        tag = str(rho).replace(".", "p")
        cfg = _make_run_config(
            f"artifacts/ablation_rho/rho_{tag}",
            seed=seed,
            rho=rho,
            lambda_values=(0.0, 0.5, 1.0),
        )
        summary = _run_and_time(f"rho sweep rho={rho}", run_full_pipeline, cfg)
        results[rho] = summary

    save_json("artifacts/ablation_rho/summary.json", {
        "rho_values": rho_values,
        "runs": {str(k): v for k, v in results.items()},
    })
    return results


# ── Main ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Extended NeurIPS experiment suite")
    parser.add_argument(
        "--experiments",
        nargs="+",
        default=["baselines", "lambda", "beta", "epochs"],
        choices=["multi_seed", "baselines", "lambda", "beta", "epochs", "rho", "all"],
        help="Which experiment groups to run",
    )
    args = parser.parse_args()

    exps = set(args.experiments)
    if "all" in exps:
        exps = {"multi_seed", "baselines", "lambda", "beta", "epochs", "rho"}

    t_total = time.perf_counter()

    if "baselines" in exps:
        _run_and_time("Competing baselines", run_baselines)

    if "lambda" in exps:
        _run_and_time("Extended lambda sweep", run_lambda_ablation)

    if "beta" in exps:
        _run_and_time("Beta ablation", run_beta_ablation)

    if "epochs" in exps:
        _run_and_time("Epochs ablation", run_epochs_ablation)

    if "rho" in exps:
        _run_and_time("Rho sweep", run_rho_sweep)

    if "multi_seed" in exps:
        _run_and_time("Multi-seed sweep", run_multi_seed)

    print(f"\n{'='*60}\nAll requested experiments done in {time.perf_counter()-t_total:.1f}s\n{'='*60}", flush=True)
