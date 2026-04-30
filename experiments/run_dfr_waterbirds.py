"""
DFR (Deep Feature Reweighting; Kirichenko et al. 2023) on Waterbirds.

Method: load the pre-trained baseline ERM ResNet-18, FREEZE the backbone,
and re-train ONLY the final linear head on a group-balanced reweighting set.
Following Kirichenko et al., we use the val split as the reweighting set.

3 seeds {0, 42, 123}.  Outputs to artifacts/waterbirds/seed_<S>/dfr.pt and
updates aggregate.json on next aggregator run.
"""
from __future__ import annotations
import os, sys, json, argparse, random
from copy import deepcopy
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler

# Reuse the v2 dataset and model definitions
sys.path.insert(0, str(ROOT / "experiments"))
from run_waterbirds_v2 import (WaterbirdsDataset, build_loaders, build_model,
                                 evaluate, EVAL_TRANSFORM, TRAIN_TRANSFORM,
                                 DATA_ROOT, DEVICE)
from causal_unlearning.utils import save_json, ensure_dir

ARTIFACT = ROOT / "artifacts" / "waterbirds"
SEEDS    = [0, 42, 123]


def train_dfr(baseline, loaders, n_epochs=10, lr=1e-3, wd=1e-3):
    """Freeze backbone, retrain head on group-balanced VAL data."""
    # Build group-balanced sampler over the val set
    val_meta = loaders["train_meta"]  # repurpose; we want val
    # build val_ds + balanced_val_loader from val split
    meta = pd.read_csv(DATA_ROOT / "metadata.csv")
    val_meta = meta[meta.split == 1].reset_index(drop=True)
    val_ds   = WaterbirdsDataset(val_meta, DATA_ROOT, TRAIN_TRANSFORM)
    groups   = val_meta.apply(lambda r: int(r.y)*2 + int(r.place), axis=1).values
    counts   = np.bincount(groups, minlength=4)
    weights  = 1.0 / np.maximum(counts[groups], 1)
    sampler  = WeightedRandomSampler(
        torch.from_numpy(weights).float(), num_samples=len(val_ds),
        replacement=True, generator=torch.Generator().manual_seed(0))
    bal_val_loader = DataLoader(val_ds, sampler=sampler, batch_size=64,
                                 num_workers=4, pin_memory=True)

    model = build_model(pretrained=False)
    model.load_state_dict(deepcopy(baseline.state_dict()))
    # Freeze backbone, train head only
    for p in model.features.parameters():
        p.requires_grad_(False)
    # Re-init head
    model.fc = nn.Linear(512, 2).to(DEVICE)
    opt = torch.optim.AdamW(model.fc.parameters(), lr=lr, weight_decay=wd)

    best_val = -1.0; best_state = None
    history = []
    for ep in range(1, n_epochs+1):
        model.train()
        tot_loss = tot_n = 0
        for batch in bal_val_loader:
            x = batch["image"].to(DEVICE); y = batch["label"].to(DEVICE)
            opt.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model(x), y)
            loss.backward(); opt.step()
            tot_loss += float(loss)*x.shape[0]; tot_n += x.shape[0]
        val_m = evaluate(model, loaders["val"])
        history.append({"epoch": ep, "loss": tot_loss/tot_n,
                        "val_avg": val_m["avg_acc"], "val_wg": val_m["worst_group_acc"]})
        if val_m["worst_group_acc"] > best_val:
            best_val = val_m["worst_group_acc"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        print(f"  [DFR] ep={ep:2d}: loss={tot_loss/tot_n:.4f} "
              f"val_wg={val_m['worst_group_acc']:.3f}", flush=True)
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, history


def run_seed(seed, n_epochs=10):
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
    OUT = ARTIFACT / f"seed_{seed}"
    base_path = OUT / "baseline.pt"
    oracle_path = OUT / "oracle.pt"
    if not base_path.exists():
        print(f"  Seed {seed}: missing baseline at {base_path}; skipping", flush=True)
        return None
    print(f"\n{'='*50}\nDFR Waterbirds seed={seed}\n{'='*50}", flush=True)

    ckpt = torch.load(base_path, map_location=DEVICE)
    baseline = build_model(pretrained=False)
    baseline.load_state_dict(ckpt["state_dict"])

    if oracle_path.exists():
        ckpt_o = torch.load(oracle_path, map_location=DEVICE)
        oracle = build_model(pretrained=False)
        oracle.load_state_dict(ckpt_o["state_dict"])
    else:
        oracle = None

    loaders = build_loaders(batch_size=64, seed=seed)
    dfr_model, hist = train_dfr(baseline, loaders, n_epochs=n_epochs)
    test_m = evaluate(dfr_model, loaders["test_cf"], oracle_model=oracle)
    print(f"  DFR test: avg={test_m['avg_acc']:.3f} wg={test_m['worst_group_acc']:.3f} "
          f"ce={test_m.get('ce_proxy', float('nan')):.3f}", flush=True)

    torch.save({"state_dict": dfr_model.state_dict(), "history": hist,
                "metrics": test_m}, OUT / "dfr.pt")
    return test_m


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n-epochs", type=int, default=10)
    args = p.parse_args()
    run_seed(args.seed, n_epochs=args.n_epochs)
