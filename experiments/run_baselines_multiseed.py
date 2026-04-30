"""
Multi-seed CMNIST baselines (Naive FT, Intervened FT, GRL, oracle distillation).

Loads per-seed baseline/oracle checkpoints from artifacts/multi_seed/seed_<S>/
and runs each baseline for 2 fine-tune epochs at the standard config.
Outputs to artifacts/baselines_multiseed/.
"""
from __future__ import annotations
import sys, json
from copy import deepcopy
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict

from causal_unlearning.config import DataConfig, UnlearningConfig
from causal_unlearning.datasets import build_dataloaders
from causal_unlearning.metrics import evaluate_model, symmetric_kl
from causal_unlearning.models import build_model, SmallCNN
from causal_unlearning.training import (load_checkpoint, _locality_penalty,
                                         resolve_device)
from causal_unlearning.baselines import (train_grl_unlearning,
                                          train_intervened_finetuning)
from causal_unlearning.utils import save_json, ensure_dir

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEEDS  = [0, 42, 123]
OUT    = ROOT / "artifacts" / "baselines_multiseed"
ensure_dir(OUT)


def _ckpt(seed, name):
    return ROOT / "artifacts" / "multi_seed" / f"seed_{seed}" / "checkpoints" / f"{name}.pt"


def _load(path):
    payload = load_checkpoint(path, device=DEVICE)
    m = build_model(payload.get("model_name", "small_cnn"))
    m.load_state_dict(payload["state_dict"])
    m.to(DEVICE).eval()
    return m


def train_naive_ft(baseline, loaders, n_epochs=2, lr=5e-4, lambda_loc=1e-3):
    model = SmallCNN().to(DEVICE)
    model.load_state_dict(deepcopy(baseline.state_dict()))
    ref = {n: p.detach().clone() for n, p in model.named_parameters()}
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    for ep in range(n_epochs):
        model.train()
        for batch in loaders.observational_train:
            x = batch["image"].to(DEVICE); y = batch["label"].to(DEVICE)
            opt.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model(x), y) + lambda_loc * _locality_penalty(model, ref)
            loss.backward(); opt.step()
    model.eval()
    return model


def train_oracle_distillation(baseline, oracle, loaders, n_epochs=2, lr=5e-4,
                               lambda_loc=1e-3, T=2.0):
    """KL-match oracle outputs on the intervened distribution."""
    model = SmallCNN().to(DEVICE)
    model.load_state_dict(deepcopy(baseline.state_dict()))
    ref = {n: p.detach().clone() for n, p in model.named_parameters()}
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    for ep in range(n_epochs):
        model.train()
        for batch in loaders.intervened_train:
            x = batch["image"].to(DEVICE); y = batch["label"].to(DEVICE)
            opt.zero_grad(set_to_none=True)
            with torch.no_grad():
                ologit = oracle(x)
            slogit = model(x)
            kl = F.kl_div(F.log_softmax(slogit/T, 1),
                          F.softmax(ologit/T, 1), reduction="batchmean") * T**2
            ce = F.cross_entropy(slogit, y)
            loss = ce + kl + lambda_loc * _locality_penalty(model, ref)
            loss.backward(); opt.step()
    model.eval()
    return model


def main():
    cfg = UnlearningConfig(epochs=2, lr=5e-4, weight_decay=0.0,
                            lambda_ce=1.0, lambda_locality=1e-3)
    all_runs = defaultdict(list)
    for seed in SEEDS:
        print(f"\n{'='*50}\nSeed {seed}\n{'='*50}", flush=True)
        if not (_ckpt(seed,"baseline").exists() and _ckpt(seed,"oracle").exists()):
            print(f"  SKIP: missing ckpt", flush=True)
            continue
        baseline = _load(_ckpt(seed, "baseline"))
        oracle   = _load(_ckpt(seed, "oracle"))
        data_cfg = DataConfig(seed=seed, num_workers=0,
                                root=str(ROOT / "data"))
        loaders  = build_dataloaders(data_cfg)
        device   = resolve_device("auto")

        # Naive FT
        print("[naive_ft]", flush=True)
        m = train_naive_ft(baseline, loaders, n_epochs=cfg.epochs, lr=cfg.lr)
        mets = evaluate_model(m, loaders.as_dict(), device, oracle_model=oracle)
        all_runs["naive_ft"].append({"seed": seed, "metrics": mets})
        print(f"  {mets}", flush=True)

        # Intervened FT
        print("[intervened_ft]", flush=True)
        m, _ = train_intervened_finetuning(
            deepcopy(baseline), loaders.intervened_train, loaders, cfg,
            oracle_model=oracle)
        mets = evaluate_model(m, loaders.as_dict(), device, oracle_model=oracle)
        all_runs["intervened_ft"].append({"seed": seed, "metrics": mets})
        print(f"  {mets}", flush=True)

        # GRL
        print("[grl]", flush=True)
        grl_m, _ = train_grl_unlearning(
            deepcopy(baseline), loaders.observational_train, loaders, cfg,
            oracle_model=oracle, grl_alpha=1.0)
        # The trained model is wrapped; apply through its forward
        class _W(nn.Module):
            def __init__(self, m): super().__init__(); self.m = m
            def forward(self, x): return self.m(x)
        grl_w = _W(grl_m).to(device)
        mets = evaluate_model(grl_w, loaders.as_dict(), device, oracle_model=oracle)
        all_runs["grl"].append({"seed": seed, "metrics": mets})
        print(f"  {mets}", flush=True)

        # Oracle distillation
        print("[distillation]", flush=True)
        m = train_oracle_distillation(baseline, oracle, loaders, n_epochs=cfg.epochs, lr=cfg.lr)
        mets = evaluate_model(m, loaders.as_dict(), device, oracle_model=oracle)
        all_runs["distillation"].append({"seed": seed, "metrics": mets})
        print(f"  {mets}", flush=True)

    save_json(OUT / "runs.json", dict(all_runs))

    # Aggregate
    summary = {}
    for method, runs in all_runs.items():
        get = lambda k: [r["metrics"][k] for r in runs]
        summary[method] = {
            "obs_acc_mean":  float(np.mean(get("observational_accuracy"))),
            "obs_acc_std":   float(np.std(get("observational_accuracy"))),
            "int_acc_mean":  float(np.mean(get("intervened_accuracy"))),
            "int_acc_std":   float(np.std(get("intervened_accuracy"))),
            "ce_mean":       float(np.mean(get("causal_effect_proxy"))),
            "ce_std":        float(np.std(get("causal_effect_proxy"))),
        }
        if any("fidelity_to_oracle_kl" in r["metrics"] for r in runs):
            ds = [r["metrics"].get("fidelity_to_oracle_kl") for r in runs
                  if r["metrics"].get("fidelity_to_oracle_kl") is not None]
            if ds:
                summary[method]["fid_mean"] = float(np.mean(ds))
                summary[method]["fid_std"]  = float(np.std(ds))
    save_json(OUT / "summary.json", summary)

    print("\n=== Aggregate ===")
    for m, s in summary.items():
        print(f"{m:<18} obs={s['obs_acc_mean']:.3f}±{s['obs_acc_std']:.3f}  "
              f"int={s['int_acc_mean']:.3f}±{s['int_acc_std']:.3f}  "
              f"ce={s['ce_mean']:.3f}±{s['ce_std']:.3f}")


if __name__ == "__main__":
    main()
