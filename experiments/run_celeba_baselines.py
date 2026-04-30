"""
CelebA full baseline matrix (mirror of Waterbirds): naive FT, IFT, GRL,
concept erasure, oracle distillation, DFR.  Loads cached baseline + oracle
from artifacts/celeba/seed_<S>/{baseline,oracle}.pt.
"""
from __future__ import annotations
import os, sys, json, time, random, argparse
from copy import deepcopy
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler

from run_celeba import (build_loaders, build_model, evaluate, locality_penalty,
                          ResNet50Model, _load_metadata, IMG_DIR, DATA_ROOT,
                          TRAIN_TRANSFORM, EVAL_TRANSFORM, CelebAGroupDataset,
                          DEVICE, ARTIFACT)
from causal_unlearning.metrics import symmetric_kl
from causal_unlearning.baselines import GradientReversalFunction
from causal_unlearning.utils import save_json, ensure_dir


def train_naive_ft(baseline, loaders, n_epochs=3, lr=5e-4, lambda_loc=1e-3):
    ref = {n: p.detach().clone() for n, p in baseline.named_parameters()}
    model = build_model(pretrained=False)
    model.load_state_dict(deepcopy(baseline.state_dict()))
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    best_val = -1.0; best_state = None; history = []
    for ep in range(1, n_epochs+1):
        model.train(); tot_loss = tot_n = 0
        for batch in loaders["train"]:
            x = batch["image"].to(DEVICE); y = batch["label"].to(DEVICE)
            opt.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model(x), y) + lambda_loc * locality_penalty(model, ref)
            loss.backward(); opt.step()
            tot_loss += float(loss)*x.shape[0]; tot_n += x.shape[0]
        val_m = evaluate(model, loaders["val"])
        history.append({"epoch": ep, "loss": tot_loss/tot_n, "val_wg": val_m["worst_group_acc"]})
        print(f"  [naive_ft] ep={ep}: loss={tot_loss/tot_n:.4f} val_wg={val_m['worst_group_acc']:.3f}", flush=True)
        if val_m["worst_group_acc"] > best_val:
            best_val = val_m["worst_group_acc"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    if best_state: model.load_state_dict(best_state)
    return model, history


def train_intervened_ft(baseline, loaders, n_epochs=3, lr=5e-4, lambda_loc=1e-3):
    ref = {n: p.detach().clone() for n, p in baseline.named_parameters()}
    model = build_model(pretrained=False)
    model.load_state_dict(deepcopy(baseline.state_dict()))
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    best_val = -1.0; best_state = None; history = []
    for ep in range(1, n_epochs+1):
        model.train(); tot_loss = tot_n = 0
        for batch in loaders["balanced_train"]:
            x = batch["image"].to(DEVICE); y = batch["label"].to(DEVICE)
            opt.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model(x), y) + lambda_loc * locality_penalty(model, ref)
            loss.backward(); opt.step()
            tot_loss += float(loss)*x.shape[0]; tot_n += x.shape[0]
        val_m = evaluate(model, loaders["val"])
        history.append({"epoch": ep, "loss": tot_loss/tot_n, "val_wg": val_m["worst_group_acc"]})
        print(f"  [ift] ep={ep}: loss={tot_loss/tot_n:.4f} val_wg={val_m['worst_group_acc']:.3f}", flush=True)
        if val_m["worst_group_acc"] > best_val:
            best_val = val_m["worst_group_acc"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    if best_state: model.load_state_dict(best_state)
    return model, history


def train_grl(baseline, loaders, n_epochs=3, lr=5e-4, peak_alpha=1.0):
    """Adversarial debiasing via GRL on background discriminator."""
    model = build_model(pretrained=False)
    model.load_state_dict(deepcopy(baseline.state_dict()))
    discriminator = nn.Sequential(nn.Linear(2048, 64), nn.ReLU(), nn.Linear(64, 2)).to(DEVICE)
    opt_main = torch.optim.AdamW(model.parameters(), lr=lr)
    opt_disc = torch.optim.AdamW(discriminator.parameters(), lr=lr)
    best_val = -1.0; best_state = None; history = []
    total_steps = n_epochs * len(loaders["train"]); step = 0
    for ep in range(1, n_epochs+1):
        model.train(); discriminator.train()
        tot_main = tot_n = 0
        for batch in loaders["train"]:
            x = batch["image"].to(DEVICE); y = batch["label"].to(DEVICE)
            c = batch["color"].to(DEVICE)
            p = step / max(total_steps, 1); step += 1
            alpha = peak_alpha * (2.0 / (1.0 + np.exp(-10.0 * p)) - 1.0)
            feat = model.get_features(x); logits = model.fc(feat)
            main_loss = F.cross_entropy(logits, y)
            adv_loss = F.cross_entropy(discriminator(GradientReversalFunction.apply(feat, alpha)), c)
            loss = main_loss + adv_loss
            opt_main.zero_grad(set_to_none=True); opt_disc.zero_grad(set_to_none=True)
            loss.backward(); opt_main.step(); opt_disc.step()
            tot_main += float(main_loss)*x.shape[0]; tot_n += x.shape[0]
        val_m = evaluate(model, loaders["val"])
        history.append({"epoch": ep, "main": tot_main/tot_n, "val_wg": val_m["worst_group_acc"]})
        print(f"  [grl] ep={ep}: main={tot_main/tot_n:.3f} val_wg={val_m['worst_group_acc']:.3f}", flush=True)
        if val_m["worst_group_acc"] > best_val:
            best_val = val_m["worst_group_acc"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    if best_state: model.load_state_dict(best_state)
    return model, history


def train_distillation(baseline, oracle, loaders, n_epochs=3, lr=5e-4, lambda_loc=1e-3, T=2.0):
    ref = {n: p.detach().clone() for n, p in baseline.named_parameters()}
    model = build_model(pretrained=False)
    model.load_state_dict(deepcopy(baseline.state_dict()))
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    best_val = -1.0; best_state = None; history = []
    for ep in range(1, n_epochs+1):
        model.train(); tot_loss = tot_n = 0
        for batch in loaders["balanced_train"]:
            x = batch["image"].to(DEVICE); y = batch["label"].to(DEVICE)
            opt.zero_grad(set_to_none=True)
            with torch.no_grad():
                ologit = oracle(x)
            slogit = model(x)
            kl = F.kl_div(F.log_softmax(slogit/T, 1), F.softmax(ologit/T, 1),
                           reduction="batchmean") * T**2
            loss = F.cross_entropy(slogit, y) + kl + lambda_loc * locality_penalty(model, ref)
            loss.backward(); opt.step()
            tot_loss += float(loss)*x.shape[0]; tot_n += x.shape[0]
        val_m = evaluate(model, loaders["val"])
        history.append({"epoch": ep, "loss": tot_loss/tot_n, "val_wg": val_m["worst_group_acc"]})
        print(f"  [distill] ep={ep}: loss={tot_loss/tot_n:.4f} val_wg={val_m['worst_group_acc']:.3f}", flush=True)
        if val_m["worst_group_acc"] > best_val:
            best_val = val_m["worst_group_acc"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    if best_state: model.load_state_dict(best_state)
    return model, history


def train_concept_erasure(baseline, loaders, n_epochs=3, lr=1e-3):
    """LEACE-style: project features orthogonal to LR-of-C direction, retrain head."""
    from sklearn.linear_model import LogisticRegression
    baseline.eval()
    feats_l, cs_l = [], []
    with torch.inference_mode():
        for batch in loaders["train"]:
            f = baseline.get_features(batch["image"].to(DEVICE)).cpu()
            feats_l.append(f); cs_l.append(batch["color"])
    F_train = torch.cat(feats_l).numpy()
    C_train = torch.cat(cs_l).numpy()
    lr_model = LogisticRegression(max_iter=200).fit(F_train, C_train)
    w = torch.from_numpy(lr_model.coef_[0]).float().to(DEVICE)
    w = w / (w.norm() + 1e-8)
    d = w.shape[0]
    P = torch.eye(d, device=DEVICE) - torch.outer(w, w)

    class ProjectedHead(nn.Module):
        def __init__(self, backbone, P, num_classes=2):
            super().__init__()
            self.backbone = backbone
            self.register_buffer("P", P.detach())
            self.head = nn.Linear(P.shape[0], num_classes).to(P.device)
        def get_features(self, x):
            return self.backbone.get_features(x) @ self.P.t()
        def forward(self, x):
            return self.head(self.get_features(x))

    erased = ProjectedHead(baseline, P).to(DEVICE)
    for p in erased.backbone.parameters(): p.requires_grad_(False)
    opt = torch.optim.AdamW(erased.head.parameters(), lr=lr, weight_decay=1e-4)
    best_val = -1.0; best_state = None; history = []
    for ep in range(1, n_epochs+1):
        erased.train(); tot_loss = tot_n = 0
        for batch in loaders["train"]:
            x = batch["image"].to(DEVICE); y = batch["label"].to(DEVICE)
            opt.zero_grad(set_to_none=True)
            loss = F.cross_entropy(erased(x), y); loss.backward(); opt.step()
            tot_loss += float(loss)*x.shape[0]; tot_n += x.shape[0]
        val_m = evaluate(erased, loaders["val"])
        history.append({"epoch": ep, "loss": tot_loss/tot_n, "val_wg": val_m["worst_group_acc"]})
        print(f"  [erase] ep={ep}: loss={tot_loss/tot_n:.4f} val_wg={val_m['worst_group_acc']:.3f}", flush=True)
        if val_m["worst_group_acc"] > best_val:
            best_val = val_m["worst_group_acc"]
            best_state = {k: v.detach().cpu().clone() for k, v in erased.state_dict().items()}
    if best_state: erased.load_state_dict(best_state)
    return erased, history


def train_dfr(baseline, val_meta, n_epochs=10, lr=1e-3, wd=1e-3, batch_size=64):
    """Deep Feature Reweighting on group-balanced VAL subset; freeze backbone, retrain head."""
    val_ds = CelebAGroupDataset(val_meta, TRAIN_TRANSFORM)
    groups = val_meta.apply(lambda r: int(r.y)*2 + int(r.c), axis=1).values
    counts = np.bincount(groups, minlength=4).astype(np.float32)
    counts = np.maximum(counts, 1)
    weights = 1.0 / counts[groups]
    sampler = WeightedRandomSampler(
        torch.from_numpy(weights).float(), num_samples=len(val_ds),
        replacement=True, generator=torch.Generator().manual_seed(0))
    bal_val_loader = DataLoader(val_ds, sampler=sampler, batch_size=batch_size,
                                 num_workers=4, pin_memory=True)

    model = build_model(pretrained=False)
    model.load_state_dict(deepcopy(baseline.state_dict()))
    for p in model.features.parameters(): p.requires_grad_(False)
    model.fc = nn.Linear(2048, 2).to(DEVICE)
    opt = torch.optim.AdamW(model.fc.parameters(), lr=lr, weight_decay=wd)
    best_val = -1.0; best_state = None; history = []
    for ep in range(1, n_epochs+1):
        model.train(); tot_loss = tot_n = 0
        for batch in bal_val_loader:
            x = batch["image"].to(DEVICE); y = batch["label"].to(DEVICE)
            opt.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model(x), y); loss.backward(); opt.step()
            tot_loss += float(loss)*x.shape[0]; tot_n += x.shape[0]
        val_m = evaluate(model, bal_val_loader)  # eval on balanced val
        history.append({"epoch": ep, "loss": tot_loss/tot_n, "val_wg": val_m["worst_group_acc"]})
        print(f"  [dfr] ep={ep}: loss={tot_loss/tot_n:.4f} val_wg={val_m['worst_group_acc']:.3f}", flush=True)
        if val_m["worst_group_acc"] > best_val:
            best_val = val_m["worst_group_acc"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    if best_state: model.load_state_dict(best_state)
    return model, history


def run_seed(seed, n_unlearn=3, train_subset=20000, eval_subset=5000):
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
    OUT = ARTIFACT / f"seed_{seed}"
    base_path = OUT / "baseline.pt"
    oracle_path = OUT / "oracle.pt"
    if not (base_path.exists() and oracle_path.exists()):
        print(f"SKIP seed {seed}: missing baseline/oracle (run run_celeba.py first)", flush=True)
        return
    print(f"\n{'='*50}\nCelebA baselines seed={seed}\n{'='*50}", flush=True)

    bckpt = torch.load(base_path, map_location=DEVICE, weights_only=False)
    baseline = build_model(pretrained=False); baseline.load_state_dict(bckpt["state_dict"])
    ockpt = torch.load(oracle_path, map_location=DEVICE, weights_only=False)
    oracle = build_model(pretrained=False); oracle.load_state_dict(ockpt["state_dict"])

    loaders = build_loaders(batch_size=64, seed=seed,
                              train_subset=train_subset, eval_subset=eval_subset)

    summary = {}

    def run_method(name, fn, ckpt_name):
        path = OUT / ckpt_name
        if path.exists():
            ckpt = torch.load(path, map_location=DEVICE, weights_only=False)
            test_m = ckpt["metrics"]
            print(f"  loaded {name}: avg={test_m['avg_acc']:.3f} wg={test_m['worst_group_acc']:.3f}", flush=True)
        else:
            print(f"[{name}]", flush=True)
            m, hist = fn()
            test_m = evaluate(m, loaders["test_cf"], oracle_model=oracle)
            try:
                torch.save({"state_dict": m.state_dict(), "history": hist,
                            "metrics": test_m}, path)
            except Exception:
                save_json(path.with_suffix(".json"), {"history": hist, "metrics": test_m})
            print(f"  {name}: avg={test_m['avg_acc']:.3f} wg={test_m['worst_group_acc']:.3f} ce={test_m.get('ce_proxy',float('nan')):.3f}", flush=True)
        summary[name] = {"metrics": test_m}

    run_method("naive_ft",      lambda: train_naive_ft(baseline, loaders, n_epochs=n_unlearn), "naive_ft.pt")
    run_method("intervened_ft", lambda: train_intervened_ft(baseline, loaders, n_epochs=n_unlearn), "intervened_ft.pt")
    run_method("grl",           lambda: train_grl(baseline, loaders, n_epochs=n_unlearn), "grl.pt")
    run_method("concept_erasure", lambda: train_concept_erasure(baseline, loaders, n_epochs=n_unlearn), "concept_erasure.pt")
    run_method("distillation",  lambda: train_distillation(baseline, oracle, loaders, n_epochs=n_unlearn), "distillation.pt")

    # DFR uses the val set
    df = _load_metadata()
    val_meta = df[df.split == 1].reset_index(drop=True)
    rng = np.random.default_rng(seed)
    if len(val_meta) > eval_subset * 4:
        idx = rng.choice(len(val_meta), size=eval_subset * 4, replace=False)
        val_meta = val_meta.iloc[idx].reset_index(drop=True)
    run_method("dfr", lambda: train_dfr(baseline, val_meta, n_epochs=10), "dfr.pt")

    save_json(OUT / "baselines_summary.json", summary)
    return summary


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n-unlearn", type=int, default=3)
    p.add_argument("--train-subset", type=int, default=20000)
    p.add_argument("--eval-subset", type=int, default=5000)
    args = p.parse_args()
    t0 = time.perf_counter()
    run_seed(args.seed, n_unlearn=args.n_unlearn,
              train_subset=args.train_subset, eval_subset=args.eval_subset)
    print(f"\nSeed {args.seed} baselines done in {time.perf_counter()-t0:.1f}s", flush=True)
