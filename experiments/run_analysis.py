"""
High-value analysis experiments for NeurIPS submission.

Addresses the 5 rejection risks:
  1. Representation probing  — does model forget C at feature level?
  2. Proper GRL baseline     — scheduled α, hyperparameter swept
  3. Data efficiency         — performance vs counterfactual pair fraction
  4. Noisy counterfactuals   — robustness to imperfect intervention
  5. Normalized CE reduction — controls for cross-seed baseline variance

All experiments use GPU (CUDA_VISIBLE_DEVICES=4 recommended).
Outputs: artifacts/analysis/
"""
from __future__ import annotations

import json
import sys
import time
from copy import deepcopy
from pathlib import Path

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
from causal_unlearning.training import (
    _locality_penalty,
    _epoch_summary,
    load_checkpoint,
    resolve_device,
    save_checkpoint,
    train_unlearning,
)
from causal_unlearning.utils import ensure_dir, save_json, set_seed

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}", flush=True)

OUT_DIR = ROOT / "artifacts" / "analysis"
ensure_dir(OUT_DIR)

BASELINE_CKPT = ROOT / "artifacts" / "default" / "checkpoints" / "baseline.pt"
ORACLE_CKPT   = ROOT / "artifacts" / "default" / "checkpoints" / "oracle.pt"
UNLEARN_CKPT  = ROOT / "artifacts" / "default" / "checkpoints" / "unlearn_lambda_0p5.pt"


def _load(path, device=DEVICE):
    payload = load_checkpoint(path, device=device)
    m = build_model(payload["model_name"])
    m.load_state_dict(payload["state_dict"])
    m.to(device)
    m.eval()
    return m


def _loaders(seed=42, batch_size=256):
    cfg = DataConfig(seed=seed, num_workers=4, batch_size=batch_size)
    return build_dataloaders(cfg)


# ─────────────────────────────────────────────────────────────────────────────
# 1. REPRESENTATION PROBING
# ─────────────────────────────────────────────────────────────────────────────

@torch.inference_mode()
def _extract_features(model: SmallCNN, loader: DataLoader, device=DEVICE):
    """Extract post-conv (128-dim) features and labels+colors."""
    model.eval()
    feats, colors, labels = [], [], []
    for batch in loader:
        imgs = batch["image"].to(device)
        f = model.features(imgs).flatten(1)   # (B, 128)
        feats.append(f.cpu().numpy())
        colors.append(batch["color"].numpy())
        labels.append(batch["label"].numpy())
    return (np.concatenate(feats),
            np.concatenate(colors),
            np.concatenate(labels))


def run_representation_probing():
    """
    Train linear probes to predict color C from features before/after unlearning.
    Metric: probe accuracy.  Drop → representation-level forgetting.
    Also test label probe (should stay high → useful features retained).
    """
    print("\n" + "="*60, flush=True)
    print("[PROBING] Representation leakage analysis", flush=True)

    loaders = _loaders()
    baseline  = _load(BASELINE_CKPT)
    unlearned = _load(UNLEARN_CKPT)
    oracle    = _load(ORACLE_CKPT)

    results = {}
    for name, model in [("baseline", baseline), ("unlearned_l05", unlearned), ("oracle", oracle)]:
        # Use observational eval split for probing (has spurious colors)
        feats_obs, colors_obs, labels_obs = _extract_features(
            model, loaders.observational_eval)
        feats_int, colors_int, labels_int = _extract_features(
            model, loaders.intervened_eval)

        scaler = StandardScaler()
        Xtr = scaler.fit_transform(feats_obs)
        Xte = scaler.transform(feats_int)

        # Color probe (on observational: spurious signal present)
        Xtr_obs = scaler.fit_transform(feats_obs)
        color_probe = LogisticRegression(max_iter=1000, C=1.0)
        color_probe.fit(Xtr_obs, colors_obs)
        color_probe_acc_obs = color_probe.score(Xtr_obs, colors_obs)
        # On intervened: color random → random-chance probe should be ~50%
        Xte_scaled = scaler.transform(feats_int)
        color_probe_acc_int = color_probe.score(Xte_scaled, colors_int)

        # Label probe
        label_probe = LogisticRegression(max_iter=1000, C=1.0)
        label_probe.fit(Xtr_obs, labels_obs)
        label_probe_acc = label_probe.score(Xte_scaled, labels_int)

        results[name] = {
            "color_probe_acc_obs": float(color_probe_acc_obs),
            "color_probe_acc_int": float(color_probe_acc_int),
            "label_probe_acc_int": float(label_probe_acc),
        }
        print(f"  {name}: color_probe_obs={color_probe_acc_obs:.3f}  "
              f"color_probe_int={color_probe_acc_int:.3f}  "
              f"label_probe_int={label_probe_acc:.3f}", flush=True)

    save_json(OUT_DIR / "representation_probing.json", results)
    return results


# ─────────────────────────────────────────────────────────────────────────────
# 2. PROPERLY TUNED GRL BASELINE
# ─────────────────────────────────────────────────────────────────────────────

class GradientReversalFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.save_for_backward(torch.as_tensor(float(alpha)))
        return x.clone()
    @staticmethod
    def backward(ctx, grad):
        alpha = ctx.saved_tensors[0].item()
        return -alpha * grad, None


class ProperGRLModel(nn.Module):
    def __init__(self, base: SmallCNN):
        super().__init__()
        self.features   = base.features
        self.classifier = base.classifier
        self.disc = nn.Sequential(
            nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.1), nn.Linear(64, 2))

    def forward(self, x):
        return self.classifier(self.features(x).flatten(1))

    def forward_full(self, x, alpha):
        f = self.features(x).flatten(1)
        return self.classifier(f), self.disc(GradientReversalFn.apply(f, alpha))


def run_grl_tuned(seed=42):
    """GRL with progressive α schedule (Ganin et al. Section 3.2)."""
    print("\n" + "="*60, flush=True)
    print("[GRL] Properly tuned GRL baseline", flush=True)

    set_seed(seed)
    loaders = _loaders(seed)
    baseline = _load(BASELINE_CKPT)
    oracle   = _load(ORACLE_CKPT)

    results = {}
    # Sweep peak_alpha values; progressive schedule: alpha(p) = 2/(1+exp(-10p))-1
    for peak_alpha in [0.3, 1.0, 3.0]:
        model = ProperGRLModel(deepcopy(baseline)).to(DEVICE)
        ref_params = {n: p.detach().clone() for n, p in model.named_parameters()
                      if "disc" not in n}
        opt = torch.optim.AdamW(model.parameters(), lr=5e-4)
        history = []

        n_epochs = 2
        steps_per_epoch = len(loaders.observational_train)
        total_steps = n_epochs * steps_per_epoch
        step = 0

        for epoch in range(1, n_epochs + 1):
            model.train()
            tot_loss = tot_ret = tot_adv = tot_loc = tot_correct = tot_n = 0
            for batch in loaders.observational_train:
                imgs   = batch["image"].to(DEVICE)
                labels = batch["label"].to(DEVICE)
                colors = batch["color"].to(DEVICE)

                # Progressive alpha schedule
                p = step / total_steps
                alpha = peak_alpha * (2.0 / (1.0 + np.exp(-10.0 * p)) - 1.0)
                step += 1

                opt.zero_grad(set_to_none=True)
                lab_logits, disc_logits = model.forward_full(imgs, alpha)
                ret = F.cross_entropy(lab_logits, labels)
                adv = F.cross_entropy(disc_logits, colors)
                # Locality on non-discriminator params
                loc_p = {n: p for n, p in model.named_parameters() if "disc" not in n}
                loc = sum(F.mse_loss(p, ref_params[n], reduction="mean")
                          for n, p in loc_p.items()) / len(loc_p)
                loss = ret + 1e-3 * loc + adv   # lambda_ce=1 for adv
                loss.backward(); opt.step()

                bs = labels.shape[0]
                tot_loss += float(loss)*bs; tot_ret += float(ret)*bs
                tot_adv  += float(adv)*bs;  tot_loc += float(loc)*bs
                tot_correct += int((lab_logits.argmax(1) == labels).sum()); tot_n += bs

            # Eval
            class _W(nn.Module):
                def __init__(self, m): super().__init__(); self.m = m
                def forward(self, x): return self.m(x)
            metrics = evaluate_model(_W(model).to(DEVICE), loaders.as_dict(), DEVICE, oracle_model=oracle)
            history.append({"epoch": epoch,
                             "alpha": alpha,
                             "retain_loss": tot_ret/tot_n,
                             "adv_loss": tot_adv/tot_n, **metrics})
            print(f"  alpha_peak={peak_alpha} ep={epoch}: "
                  f"int={metrics['intervened_accuracy']:.3f} CE={metrics['causal_effect_proxy']:.3f}", flush=True)

        final_metrics = history[-1]
        results[f"peak_alpha_{peak_alpha}"] = {"history": history, "metrics": {
            k: final_metrics[k] for k in ["observational_accuracy","intervened_accuracy",
                                           "causal_effect_proxy","fidelity_to_oracle_kl"]}}

    save_json(OUT_DIR / "grl_tuned.json", results)
    return results


# ─────────────────────────────────────────────────────────────────────────────
# 3. DATA EFFICIENCY (counterfactual pair fraction)
# ─────────────────────────────────────────────────────────────────────────────

def run_data_efficiency(seed=42):
    """Performance vs fraction of counterfactual pairs used."""
    print("\n" + "="*60, flush=True)
    print("[DATA EFF] Counterfactual data efficiency", flush=True)

    fractions = [0.05, 0.1, 0.25, 0.5, 1.0]
    results = []

    cfg = DataConfig(seed=seed, num_workers=4)
    loaders = _loaders(seed)
    baseline = _load(BASELINE_CKPT)
    oracle   = _load(ORACLE_CKPT)

    for frac in fractions:
        set_seed(seed)
        # Subset the observational training set
        full_ds = loaders.observational_train.dataset
        n_sub = max(1, int(len(full_ds) * frac))
        rng = np.random.RandomState(seed)
        idx = rng.choice(len(full_ds), n_sub, replace=False)
        sub_loader = DataLoader(
            Subset(full_ds, idx), batch_size=256, shuffle=True, num_workers=4)

        model = build_model().to(DEVICE)
        model.load_state_dict(deepcopy(baseline.state_dict()))
        ucfg = UnlearningConfig(epochs=2, lr=5e-4, lambda_ce=0.5, lambda_locality=1e-3)

        # Inline unlearning to use sub_loader
        ref_params = {n: p.detach().clone() for n, p in model.named_parameters()}
        opt = torch.optim.AdamW(model.parameters(), lr=ucfg.lr)
        for epoch in range(ucfg.epochs):
            model.train()
            for batch in sub_loader:
                factual = batch["image"].to(DEVICE)
                cf      = batch["counterfactual"].to(DEVICE)
                labels  = batch["label"].to(DEVICE)
                opt.zero_grad(set_to_none=True)
                fl = model(factual); cl = model(cf)
                ret  = F.cross_entropy(fl, labels)
                ce   = symmetric_kl(fl, cl).mean()
                loc  = _locality_penalty(model, ref_params)
                loss = ret + 0.5*ce + 1e-3*loc
                loss.backward(); opt.step()

        metrics = evaluate_model(model, loaders.as_dict(), DEVICE, oracle_model=oracle)
        entry = {"frac": frac, "n_pairs": n_sub, "metrics": metrics}
        results.append(entry)
        print(f"  frac={frac:.0%} (n={n_sub}): int={metrics['intervened_accuracy']:.3f} "
              f"CE={metrics['causal_effect_proxy']:.3f}", flush=True)

    save_json(OUT_DIR / "data_efficiency.json", {"fractions": fractions, "runs": results})
    return results


# ─────────────────────────────────────────────────────────────────────────────
# 4. NOISY COUNTERFACTUALS (stress test)
# ─────────────────────────────────────────────────────────────────────────────

def run_noisy_counterfactuals(seed=42):
    """Stress test: add Gaussian noise to counterfactual images."""
    print("\n" + "="*60, flush=True)
    print("[STRESS] Noisy counterfactuals", flush=True)

    noise_levels = [0.0, 0.05, 0.10, 0.20, 0.50]
    results = []

    loaders = _loaders(seed)
    baseline = _load(BASELINE_CKPT)
    oracle   = _load(ORACLE_CKPT)

    for sigma in noise_levels:
        set_seed(seed)
        model = build_model().to(DEVICE)
        model.load_state_dict(deepcopy(baseline.state_dict()))
        ref_params = {n: p.detach().clone() for n, p in model.named_parameters()}
        opt = torch.optim.AdamW(model.parameters(), lr=5e-4)

        for epoch in range(2):
            model.train()
            for batch in loaders.observational_train:
                factual = batch["image"].to(DEVICE)
                cf_clean = batch["counterfactual"].to(DEVICE)
                labels   = batch["label"].to(DEVICE)
                # Add noise to counterfactual
                cf = (cf_clean + sigma * torch.randn_like(cf_clean)).clamp(0, 1)
                opt.zero_grad(set_to_none=True)
                fl = model(factual); cl = model(cf)
                ret  = F.cross_entropy(fl, labels)
                ce   = symmetric_kl(fl, cl).mean()
                loc  = _locality_penalty(model, ref_params)
                loss = ret + 0.5*ce + 1e-3*loc
                loss.backward(); opt.step()

        metrics = evaluate_model(model, loaders.as_dict(), DEVICE, oracle_model=oracle)
        entry = {"sigma": sigma, "metrics": metrics}
        results.append(entry)
        print(f"  sigma={sigma:.2f}: int={metrics['intervened_accuracy']:.3f} "
              f"CE={metrics['causal_effect_proxy']:.3f}", flush=True)

    save_json(OUT_DIR / "noisy_counterfactuals.json", {"noise_levels": noise_levels, "runs": results})
    return results


# ─────────────────────────────────────────────────────────────────────────────
# 5. NORMALIZED CE REDUCTION (controls for cross-seed variance)
# ─────────────────────────────────────────────────────────────────────────────

def compute_normalized_metrics():
    """
    For each seed, compute:
    - CE reduction ratio: (CE_base - CE_ours) / CE_base
    - Int-acc gain relative to oracle gap: (int_ours - int_base) / (int_oracle - int_base)
    These normalize away baseline strength variation.
    """
    print("\n" + "="*60, flush=True)
    print("[NORMALIZED] Computing normalized metrics across seeds", flush=True)

    results = {}
    for seed in [0, 42, 123]:
        path = ROOT / "artifacts" / "multi_seed" / f"seed_{seed}" / "summary.json"
        with open(path) as f:
            s = json.load(f)
        base_ce   = s["baseline"]["metrics"]["causal_effect_proxy"]
        base_int  = s["baseline"]["metrics"]["intervened_accuracy"]
        oracle_int = s["oracle"]["metrics"]["intervened_accuracy"]

        seed_res = {}
        for run in s["unlearning_runs"]:
            lam = run["lambda_ce"]
            m   = run["metrics"]
            ce_ours  = m["causal_effect_proxy"]
            int_ours = m["intervened_accuracy"]
            # CE reduction ratio (1.0 = perfect, 0.0 = no improvement)
            ce_reduction = (base_ce - ce_ours) / base_ce if base_ce > 0 else 0.0
            # Oracle gap closure ratio (1.0 = matches oracle, 0.0 = no improvement)
            oracle_gap = oracle_int - base_int
            gap_closure = (int_ours - base_int) / oracle_gap if oracle_gap > 0 else 0.0
            seed_res[f"lambda_{lam}"] = {
                "ce_reduction_ratio": ce_reduction,
                "oracle_gap_closure": gap_closure,
                "raw_int_acc": int_ours,
                "raw_ce": ce_ours,
            }
        results[f"seed_{seed}"] = seed_res
        print(f"  seed={seed}: λ=0.5 CE_red={seed_res['lambda_0.5']['ce_reduction_ratio']:.3f} "
              f"gap_close={seed_res['lambda_0.5']['oracle_gap_closure']:.3f}", flush=True)

    # Summary stats
    for lam_key in ["lambda_0.0", "lambda_0.1", "lambda_0.5", "lambda_1.0"]:
        ce_reds = [results[f"seed_{s}"][lam_key]["ce_reduction_ratio"] for s in [0,42,123]]
        gc = [results[f"seed_{s}"][lam_key]["oracle_gap_closure"] for s in [0,42,123]]
        import statistics
        print(f"  {lam_key}: CE_red={statistics.mean(ce_reds):.3f}±{statistics.stdev(ce_reds):.3f} "
              f"gap_close={statistics.mean(gc):.3f}±{statistics.stdev(gc):.3f}", flush=True)

    save_json(OUT_DIR / "normalized_metrics.json", results)
    return results


# ─────────────────────────────────────────────────────────────────────────────
# 6. ORACLE DISTILLATION BASELINE
# ─────────────────────────────────────────────────────────────────────────────

def run_oracle_distillation(seed=42):
    """
    Distill oracle θ* into θ₀ via KL matching on intervened data.
    This is the 'upper bound' baseline — requires oracle access at unlearning time.
    """
    print("\n" + "="*60, flush=True)
    print("[DISTIL] Oracle distillation baseline", flush=True)

    set_seed(seed)
    loaders = _loaders(seed)
    baseline = _load(BASELINE_CKPT)
    oracle   = _load(ORACLE_CKPT)

    model = build_model().to(DEVICE)
    model.load_state_dict(deepcopy(baseline.state_dict()))
    ref_params = {n: p.detach().clone() for n, p in model.named_parameters()}
    opt = torch.optim.AdamW(model.parameters(), lr=5e-4)
    history = []

    print("  [distil] Distilling oracle knowledge with KL loss on intervened data", flush=True)
    for epoch in range(2):
        model.train()
        tot_ce = tot_kl = tot_loc = tot_n = 0
        for batch in loaders.intervened_train:
            imgs   = batch["image"].to(DEVICE)
            labels = batch["label"].to(DEVICE)
            opt.zero_grad(set_to_none=True)
            with torch.no_grad():
                oracle_probs = F.softmax(oracle(imgs), dim=1)
            student_log = F.log_softmax(model(imgs), dim=1)
            # KL distillation loss
            kl_loss = F.kl_div(student_log, oracle_probs, reduction="batchmean")
            ce_loss = F.cross_entropy(model(imgs), labels)
            loc = _locality_penalty(model, ref_params)
            loss = ce_loss + 2.0 * kl_loss + 1e-3 * loc
            loss.backward(); opt.step()
            tot_ce += float(ce_loss)*imgs.shape[0]
            tot_kl += float(kl_loss)*imgs.shape[0]
            tot_loc += float(loc)*imgs.shape[0]
            tot_n += imgs.shape[0]

        metrics = evaluate_model(model, loaders.as_dict(), DEVICE, oracle_model=oracle)
        history.append({"epoch": epoch+1, "kl_loss": tot_kl/tot_n, **metrics})
        print(f"  ep={epoch+1}: int={metrics['intervened_accuracy']:.3f} "
              f"CE={metrics['causal_effect_proxy']:.3f} Dfid={metrics.get('fidelity_to_oracle_kl',0):.3f}",
              flush=True)

    result = {"history": history, "metrics": history[-1]}
    save_json(OUT_DIR / "oracle_distillation.json", result)
    return result


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiments", nargs="+",
                        default=["probe", "grl", "data_eff", "noise", "norm", "distil"],
                        choices=["probe", "grl", "data_eff", "noise", "norm", "distil", "all"])
    args = parser.parse_args()
    exps = set(args.experiments)
    if "all" in exps:
        exps = {"probe", "grl", "data_eff", "noise", "norm", "distil"}

    t0 = time.perf_counter()
    if "norm" in exps:
        compute_normalized_metrics()
    if "probe" in exps:
        run_representation_probing()
    if "grl" in exps:
        run_grl_tuned()
    if "data_eff" in exps:
        run_data_efficiency()
    if "noise" in exps:
        run_noisy_counterfactuals()
    if "distil" in exps:
        run_oracle_distillation()

    print(f"\nAll done in {time.perf_counter()-t0:.1f}s", flush=True)
