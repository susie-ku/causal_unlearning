"""
Multi-seed analysis on CMNIST: probing, data efficiency, noisy CFs, distance-to-oracle
at seeds {0, 42, 123}.  Outputs to artifacts/analysis_multiseed/.
"""
from __future__ import annotations
import os, sys, json
from pathlib import Path
from collections import defaultdict
from copy import deepcopy

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Subset

from causal_unlearning.config import DataConfig, UnlearningConfig
from causal_unlearning.datasets import build_dataloaders, ColoredMNISTDataset
from causal_unlearning.metrics import evaluate_model, symmetric_kl
from causal_unlearning.models import build_model, SmallCNN
from causal_unlearning.training import load_checkpoint, train_unlearning, _locality_penalty
from causal_unlearning.utils import save_json, ensure_dir

DEVICE  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEEDS   = [0, 42, 123]
OUT_DIR = ROOT / "artifacts" / "analysis_multiseed"
ensure_dir(OUT_DIR)


def _ckpt(seed, name):
    return ROOT / "artifacts" / "multi_seed" / f"seed_{seed}" / "checkpoints" / f"{name}.pt"


def _load(path):
    payload = load_checkpoint(path, device=DEVICE)
    m = build_model(payload.get("model_name", "small_cnn"))
    m.load_state_dict(payload["state_dict"])
    m.to(DEVICE).eval()
    return m


def _loaders(seed=42, batch_size=256):
    cfg = DataConfig(seed=seed, num_workers=4, batch_size=batch_size,
                     root=str(ROOT / "data"))
    return build_dataloaders(cfg)


@torch.inference_mode()
def _extract(model, loader):
    feats, colors, labels = [], [], []
    for batch in loader:
        x = batch["image"].to(DEVICE); y = batch["label"]; c = batch["color"]
        f = model.features(x).flatten(1).cpu()
        feats.append(f); colors.append(c); labels.append(y)
    return (torch.cat(feats).numpy(),
            torch.cat(colors).numpy(),
            torch.cat(labels).numpy())


# ─────────────────────────────────────────────────────────────────────────────
# 1. Multi-seed representation probing
# ─────────────────────────────────────────────────────────────────────────────
def run_probing_multiseed():
    print("\n[1] Multi-seed representation probing", flush=True)
    results = defaultdict(list)
    for seed in SEEDS:
        loaders = _loaders(seed)
        for tag, fname in [("baseline",   "baseline"),
                           ("unlearn_l05","unlearn_lambda_0p5"),
                           ("oracle",     "oracle")]:
            p = _ckpt(seed, fname)
            if not p.exists():
                print(f"  SKIP seed={seed} {tag}: missing {p}", flush=True)
                continue
            model = _load(p)
            f_obs, c_obs, y_obs = _extract(model, loaders.observational_eval)
            f_int, c_int, y_int = _extract(model, loaders.intervened_eval)
            scaler = StandardScaler().fit(f_obs)
            X_obs = scaler.transform(f_obs); X_int = scaler.transform(f_int)
            cp = LogisticRegression(max_iter=1000, C=1.0).fit(X_obs, c_obs)
            lp = LogisticRegression(max_iter=1000, C=1.0).fit(X_obs, y_obs)
            results[tag].append({
                "seed": seed,
                "color_probe_obs": float(cp.score(X_obs, c_obs)),
                "color_probe_int": float(cp.score(X_int, c_int)),
                "label_probe_int": float(lp.score(X_int, y_int)),
            })
            print(f"  seed={seed} {tag}: c_obs={cp.score(X_obs,c_obs):.3f} "
                  f"c_int={cp.score(X_int,c_int):.3f} l_int={lp.score(X_int,y_int):.3f}",
                  flush=True)
    save_json(OUT_DIR / "probing_multiseed.json", dict(results))
    summary = {}
    for tag, runs in results.items():
        summary[tag] = {}
        for k in ["color_probe_obs", "color_probe_int", "label_probe_int"]:
            vals = [r[k] for r in runs]
            summary[tag][k+"_mean"] = float(np.mean(vals))
            summary[tag][k+"_std"]  = float(np.std(vals))
    save_json(OUT_DIR / "probing_summary.json", summary)
    return summary


# ─────────────────────────────────────────────────────────────────────────────
# 2. Multi-seed data efficiency
# ─────────────────────────────────────────────────────────────────────────────
def _do_unlearning_run(baseline, oracle, train_loader, eval_loaders,
                        n_epochs=2, lambda_ce=0.5, lambda_loc=1e-3, lr=5e-4):
    """Run composite-loss fine-tuning manually."""
    model = SmallCNN().to(DEVICE)
    model.load_state_dict(deepcopy(baseline.state_dict()))
    ref = {n: p.detach().clone() for n, p in model.named_parameters()}
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    for ep in range(n_epochs):
        model.train()
        for batch in train_loader:
            x  = batch["image"].to(DEVICE)
            cf = batch["counterfactual"].to(DEVICE)
            y  = batch["label"].to(DEVICE)
            opt.zero_grad(set_to_none=True)
            logits  = model(x); cf_logits = model(cf)
            ret = F.cross_entropy(logits, y)
            ce  = symmetric_kl(logits, cf_logits).mean()
            loc = _locality_penalty(model, ref)
            loss = ret + lambda_ce * ce + lambda_loc * loc
            loss.backward(); opt.step()
    model.eval()
    return model


@torch.inference_mode()
def _eval_full(model, eval_loaders, oracle):
    """Compute obs/int acc, CE, D_fid."""
    obs_corr, obs_n = 0, 0
    for batch in eval_loaders.observational_eval:
        x = batch["image"].to(DEVICE); y = batch["label"].to(DEVICE)
        obs_corr += int((model(x).argmax(1) == y).sum()); obs_n += y.shape[0]
    int_corr, int_n = 0, 0
    ce_list, dfid_list = [], []
    for batch in eval_loaders.intervened_eval:
        x  = batch["image"].to(DEVICE); y  = batch["label"].to(DEVICE)
        logits = model(x)
        int_corr += int((logits.argmax(1) == y).sum()); int_n += y.shape[0]
        if "counterfactual" in batch:
            cf = batch["counterfactual"].to(DEVICE)
            ce_list.append(symmetric_kl(logits, model(cf)).cpu())
        with torch.inference_mode():
            o = oracle(x)
        dfid_list.append(F.kl_div(F.log_softmax(logits,1),
                                   F.softmax(o,1), reduction="none").sum(1).cpu())
    return {
        "obs_acc": obs_corr/obs_n, "int_acc": int_corr/int_n,
        "ce": float(torch.cat(ce_list).mean()) if ce_list else float("nan"),
        "dfid": float(torch.cat(dfid_list).mean()) if dfid_list else float("nan"),
    }


def run_data_efficiency_multiseed(fractions=(0.05, 0.1, 0.25, 0.5, 1.0),
                                   lambda_ce=0.5, n_epochs=2):
    print("\n[2] Multi-seed data efficiency", flush=True)
    runs = []
    for seed in SEEDS:
        if not (_ckpt(seed,"baseline").exists() and _ckpt(seed,"oracle").exists()):
            continue
        baseline = _load(_ckpt(seed, "baseline"))
        oracle   = _load(_ckpt(seed, "oracle"))
        loaders  = _loaders(seed, batch_size=128)
        full = loaders.observational_train.dataset
        n = len(full)
        for frac in fractions:
            n_use = int(frac * n)
            torch.manual_seed(seed); np.random.seed(seed)
            idx = np.random.permutation(n)[:n_use]
            sub = Subset(full, idx)
            sub_loader = DataLoader(sub, batch_size=128, shuffle=True, num_workers=0)
            model = _do_unlearning_run(baseline, oracle, sub_loader, loaders,
                                         n_epochs=n_epochs, lambda_ce=lambda_ce)
            mets = _eval_full(model, loaders, oracle)
            runs.append({"seed": seed, "frac": frac, "n_pairs": n_use, **mets})
            print(f"  seed={seed} frac={frac}: ce={mets['ce']:.3f} int={mets['int_acc']:.3f}", flush=True)
    save_json(OUT_DIR / "data_efficiency_multiseed.json", {"runs": runs})
    by_frac = defaultdict(list)
    for r in runs: by_frac[r["frac"]].append(r)
    summary = []
    for frac, rs in sorted(by_frac.items()):
        summary.append({
            "frac": frac,
            "ce_mean":  float(np.mean([r["ce"] for r in rs])),
            "ce_std":   float(np.std([r["ce"] for r in rs])),
            "int_mean": float(np.mean([r["int_acc"] for r in rs])),
            "int_std":  float(np.std([r["int_acc"] for r in rs])),
        })
    save_json(OUT_DIR / "data_efficiency_summary.json", summary)
    return summary


# ─────────────────────────────────────────────────────────────────────────────
# 3. Multi-seed noisy CFs
# ─────────────────────────────────────────────────────────────────────────────
class _NoisyCFLoader:
    def __init__(self, base, sigma):
        self.base = base; self.sigma = sigma
    def __iter__(self):
        for batch in self.base:
            if "counterfactual" in batch and self.sigma > 0:
                cf = batch["counterfactual"]
                cf = (cf + self.sigma * torch.randn_like(cf)).clamp(0, 1)
                batch = {**batch, "counterfactual": cf}
            yield batch
    def __len__(self):
        return len(self.base)


def run_noisy_cfs_multiseed(noise_levels=(0.0, 0.05, 0.1, 0.2, 0.5),
                              lambda_ce=0.5, n_epochs=2):
    print("\n[3] Multi-seed noisy CFs", flush=True)
    runs = []
    for seed in SEEDS:
        if not (_ckpt(seed,"baseline").exists() and _ckpt(seed,"oracle").exists()):
            continue
        baseline = _load(_ckpt(seed, "baseline"))
        oracle   = _load(_ckpt(seed, "oracle"))
        loaders  = _loaders(seed, batch_size=128)
        for sigma in noise_levels:
            noisy_train = _NoisyCFLoader(loaders.observational_train, sigma)
            model = _do_unlearning_run(baseline, oracle, noisy_train, loaders,
                                         n_epochs=n_epochs, lambda_ce=lambda_ce)
            mets = _eval_full(model, loaders, oracle)
            runs.append({"seed": seed, "sigma": sigma, **mets})
            print(f"  seed={seed} sigma={sigma}: ce={mets['ce']:.3f} int={mets['int_acc']:.3f}", flush=True)
    save_json(OUT_DIR / "noisy_cfs_multiseed.json", {"runs": runs})
    by_sigma = defaultdict(list)
    for r in runs: by_sigma[r["sigma"]].append(r)
    summary = []
    for s, rs in sorted(by_sigma.items()):
        summary.append({
            "sigma": s,
            "ce_mean":  float(np.mean([r["ce"] for r in rs])),
            "ce_std":   float(np.std([r["ce"] for r in rs])),
            "int_mean": float(np.mean([r["int_acc"] for r in rs])),
            "int_std":  float(np.std([r["int_acc"] for r in rs])),
        })
    save_json(OUT_DIR / "noisy_cfs_summary.json", summary)
    return summary


# ─────────────────────────────────────────────────────────────────────────────
# 4. Distance-to-oracle
# ─────────────────────────────────────────────────────────────────────────────
def run_distance_to_oracle():
    print("\n[4] Distance-to-oracle (CE vs D_fid scatter)", flush=True)
    points = []
    for seed in SEEDS:
        if not _ckpt(seed,"oracle").exists(): continue
        oracle  = _load(_ckpt(seed, "oracle"))
        loaders = _loaders(seed, batch_size=256)
        for tag, fname in [("baseline","baseline"),
                            ("ours_l00","unlearn_lambda_0"),
                            ("ours_l01","unlearn_lambda_0p1"),
                            ("ours_l05","unlearn_lambda_0p5"),
                            ("ours_l10","unlearn_lambda_1")]:
            p = _ckpt(seed, fname)
            if not p.exists(): continue
            m = _load(p)
            mets = _eval_full(m, loaders, oracle)
            points.append({"seed": seed, "tag": tag, **mets})
            print(f"  seed={seed} {tag}: ce={mets['ce']:.3f} dfid={mets['dfid']:.3f}", flush=True)
    save_json(OUT_DIR / "distance_to_oracle.json", {"points": points})
    return points


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="all",
                    choices=["all","probing","data_efficiency","noisy_cfs","dist"])
    args = ap.parse_args()
    if args.task in ("all","probing"):           run_probing_multiseed()
    if args.task in ("all","dist"):              run_distance_to_oracle()
    if args.task in ("all","data_efficiency"):   run_data_efficiency_multiseed()
    if args.task in ("all","noisy_cfs"):         run_noisy_cfs_multiseed()
    print("\nDone.")
