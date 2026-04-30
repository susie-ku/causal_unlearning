"""
Waterbirds experiment with ResNet-18.

Dataset: waterbird_complete95_forest2water2
  - Y ∈ {0=landbird, 1=waterbird}  (binary classification)
  - C ∈ {0=land bg, 1=water bg}    (spurious attribute, 95% correlated at training)
  - 4 groups: (Y=0,C=0), (Y=0,C=1), (Y=1,C=0), (Y=1,C=1)

Counterfactual construction: for each (x, c, y) training example, we sample
a random image from the SAME label but OPPOSITE background (same-class,
different-concept partner). This is an approximation of the true counterfactual
(same bird identity in different background), standard for Waterbirds.

Models:
  - Baseline θ₀: ResNet-18 ERM on spurious training set
  - Oracle θ*:  ResNet-18 trained on group-balanced set (simulates intervened world)
  - Unlearned θ⁻: composite loss fine-tuning from θ₀
  - Baselines: LfF (upsampling minority), plain FT

Metrics (all standard for Waterbirds):
  - Average accuracy
  - Worst-group accuracy (min over 4 groups)
  - CE proxy (background invariance)
  - Fidelity to oracle
"""
from __future__ import annotations

import os, sys, json, time, random
from copy import deepcopy
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms
from torchvision.models import resnet18, ResNet18_Weights

from causal_unlearning.metrics import symmetric_kl
from causal_unlearning.training import _locality_penalty, resolve_device
from causal_unlearning.utils import ensure_dir, save_json

DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_ROOT   = ROOT / "data" / "waterbird_complete95_forest2water2"
OUT_DIR     = ROOT / "artifacts" / "waterbirds"
ensure_dir(OUT_DIR)

print(f"Using device: {DEVICE}", flush=True)

TRAIN_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])
EVAL_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


# ─────────────────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────────────────

class WaterbirdsDataset(Dataset):
    """Standard Waterbirds dataset with counterfactual partner lookup."""

    def __init__(self, metadata: pd.DataFrame, data_root: Path, transform,
                 with_counterfactual: bool = False):
        self.meta = metadata.reset_index(drop=True)
        self.root = data_root
        self.transform = transform
        self.with_cf = with_counterfactual

        # Build same-label, opposite-background index for counterfactuals
        if with_counterfactual:
            self._cf_idx: dict[int, list[int]] = {}
            for i, row in self.meta.iterrows():
                y_i, c_i = int(row.y), int(row.place)
                # Same label, opposite background
                partners = self.meta[
                    (self.meta.y == y_i) & (self.meta.place == (1 - c_i))
                ].index.tolist()
                self._cf_idx[i] = partners

    def __len__(self):
        return len(self.meta)

    def __getitem__(self, idx):
        row = self.meta.iloc[idx]
        img_path = self.root / row.img_filename
        img = Image.open(img_path).convert("RGB")
        img_t = self.transform(img)
        item = {
            "image": img_t,
            "label": torch.tensor(int(row.y), dtype=torch.long),
            "color": torch.tensor(int(row.place), dtype=torch.long),  # background
            "group": torch.tensor(int(row.y) * 2 + int(row.place), dtype=torch.long),
        }
        if self.with_cf:
            partners = self._cf_idx.get(idx, [])
            if partners:
                cf_idx = random.choice(partners)
            else:
                cf_idx = idx   # fallback: same image (degenerate but rare)
            cf_row = self.meta.iloc[cf_idx]
            cf_img = Image.open(self.root / cf_row.img_filename).convert("RGB")
            item["counterfactual"] = self.transform(cf_img)
        return item


def build_waterbirds_loaders(batch_size: int = 64, seed: int = 42):
    """Build train/val/test loaders with optional group-balanced train."""
    meta = pd.read_csv(DATA_ROOT / "metadata.csv")
    # splits: 0=train, 1=val, 2=test
    train_meta = meta[meta.split == 0].reset_index(drop=True)
    val_meta   = meta[meta.split == 1].reset_index(drop=True)
    test_meta  = meta[meta.split == 2].reset_index(drop=True)

    train_ds    = WaterbirdsDataset(train_meta, DATA_ROOT, TRAIN_TRANSFORM, with_counterfactual=True)
    val_ds      = WaterbirdsDataset(val_meta,   DATA_ROOT, EVAL_TRANSFORM)
    test_ds     = WaterbirdsDataset(test_meta,  DATA_ROOT, EVAL_TRANSFORM)
    test_cf_ds  = WaterbirdsDataset(test_meta,  DATA_ROOT, EVAL_TRANSFORM, with_counterfactual=True)

    kw = dict(batch_size=batch_size, num_workers=4, pin_memory=True)
    g = torch.Generator(); g.manual_seed(seed)
    train_loader   = DataLoader(train_ds,   shuffle=True, generator=g, **kw)
    val_loader     = DataLoader(val_ds,     shuffle=False, **kw)
    test_loader    = DataLoader(test_ds,    shuffle=False, **kw)
    test_cf_loader = DataLoader(test_cf_ds, shuffle=False, **kw)

    # Group-balanced sampler for oracle training (simulates intervened world)
    groups = train_meta.apply(lambda r: int(r.y)*2 + int(r.place), axis=1).values
    group_counts = np.bincount(groups)
    weights = 1.0 / group_counts[groups]
    sampler = WeightedRandomSampler(
        torch.from_numpy(weights).float(), num_samples=len(train_ds),
        replacement=True, generator=torch.Generator().manual_seed(seed))
    balanced_loader = DataLoader(train_ds, sampler=sampler, **kw)

    return {
        "train": train_loader,
        "val": val_loader,
        "test": test_loader,
        "test_cf": test_cf_loader,
        "balanced_train": balanced_loader,
        "train_meta": train_meta,
        "test_meta": test_meta,
    }


def build_model_wb(pretrained: bool = True) -> nn.Module:
    m = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
    m.fc = nn.Linear(512, 2)
    return m.to(DEVICE)


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation
# ─────────────────────────────────────────────────────────────────────────────

@torch.inference_mode()
def evaluate_wb(model: nn.Module, loader: DataLoader, oracle_model=None):
    model.eval()
    preds, labels_list, groups_list = [], [], []
    cf_kl_list = []
    oracle_kl_list = []

    for batch in loader:
        imgs   = batch["image"].to(DEVICE)
        labels = batch["label"].to(DEVICE)
        groups = batch["group"]
        logits = model(imgs)
        preds.append(logits.argmax(1).cpu())
        labels_list.append(labels.cpu())
        groups_list.append(groups)

        # CE proxy: only computed if counterfactual available in batch
        if "counterfactual" in batch:
            cf = batch["counterfactual"].to(DEVICE)
            cf_logits = model(cf)
            kl = symmetric_kl(logits, cf_logits)
            cf_kl_list.append(kl.cpu())

        if oracle_model is not None:
            with torch.no_grad():
                oracle_logits = oracle_model(imgs)
            kl_o = F.kl_div(F.log_softmax(logits, 1),
                             F.softmax(oracle_logits, 1), reduction="none").sum(1)
            oracle_kl_list.append(kl_o.cpu())

    all_preds  = torch.cat(preds)
    all_labels = torch.cat(labels_list)
    all_groups = torch.cat(groups_list)

    avg_acc = float((all_preds == all_labels).float().mean())
    group_accs = {}
    for g in range(4):
        mask = all_groups == g
        if mask.sum() > 0:
            group_accs[g] = float((all_preds[mask] == all_labels[mask]).float().mean())
    worst_group_acc = min(group_accs.values()) if group_accs else 0.0

    metrics = {
        "avg_acc": avg_acc,
        "worst_group_acc": worst_group_acc,
        "group_accs": {str(k): v for k, v in group_accs.items()},
    }
    if cf_kl_list:
        metrics["ce_proxy"] = float(torch.cat(cf_kl_list).mean())
    if oracle_kl_list:
        metrics["fidelity_kl"] = float(torch.cat(oracle_kl_list).mean())
    return metrics


# ─────────────────────────────────────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────────────────────────────────────

def train_wb_supervised(model, loader, val_loader, n_epochs=5, lr=1e-4):
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_epochs)
    history = []
    for epoch in range(1, n_epochs+1):
        model.train()
        tot_loss = tot_correct = tot_n = 0
        for batch in loader:
            imgs, labels = batch["image"].to(DEVICE), batch["label"].to(DEVICE)
            opt.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model(imgs), labels)
            loss.backward(); opt.step()
            tot_loss += float(loss)*imgs.shape[0]
            tot_correct += int((model(imgs).argmax(1)==labels).sum()); tot_n += imgs.shape[0]
        sched.step()
        val_m = evaluate_wb(model, val_loader)
        history.append({"epoch": epoch, "train_loss": tot_loss/tot_n,
                        "val_avg_acc": val_m["avg_acc"],
                        "val_worst_acc": val_m["worst_group_acc"]})
        print(f"  ep={epoch}: loss={tot_loss/tot_n:.4f} val_avg={val_m['avg_acc']:.3f} "
              f"val_wg={val_m['worst_group_acc']:.3f}", flush=True)
    return model, history


def train_wb_unlearning(model, baseline_model, loaders, n_epochs=3,
                        lr=5e-4, lambda_ce=0.5, lambda_loc=1e-3, oracle_model=None):
    ref_params = {n: p.detach().clone() for n, p in model.named_parameters()}
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    history = []

    for epoch in range(1, n_epochs+1):
        model.train()
        tot_ret = tot_ce = tot_loc = tot_correct = tot_n = 0
        for batch in loaders["train"]:
            factual = batch["image"].to(DEVICE)
            labels  = batch["label"].to(DEVICE)
            cf = batch.get("counterfactual")
            if cf is None:
                continue
            cf = cf.to(DEVICE)

            opt.zero_grad(set_to_none=True)
            fl = model(factual)
            cl = model(cf)
            ret = F.cross_entropy(fl, labels)
            ce  = symmetric_kl(fl, cl).mean()
            loc = _locality_penalty(model, ref_params)
            loss = ret + lambda_ce * ce + lambda_loc * loc
            loss.backward(); opt.step()

            tot_ret += float(ret)*factual.shape[0]
            tot_ce  += float(ce)*factual.shape[0]
            tot_loc += float(loc)*factual.shape[0]
            tot_correct += int((fl.argmax(1)==labels).sum())
            tot_n += factual.shape[0]

        val_m = evaluate_wb(model, loaders["val"], oracle_model)
        history.append({"epoch": epoch,
                        "retain_loss": tot_ret/tot_n,
                        "ce_loss": tot_ce/tot_n, **val_m})
        print(f"  [unlearn] ep={epoch}: val_avg={val_m['avg_acc']:.3f} "
              f"val_wg={val_m['worst_group_acc']:.3f} "
              f"ce_proxy={val_m.get('ce_proxy',float('nan')):.3f}", flush=True)
    return model, history


# ─────────────────────────────────────────────────────────────────────────────
# Main experiment
# ─────────────────────────────────────────────────────────────────────────────

def run_waterbirds_experiment(seed: int = 42):
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)

    print(f"\n{'='*60}\nWaterbirds Experiment (seed={seed})\n{'='*60}", flush=True)
    loaders = build_waterbirds_loaders(batch_size=64, seed=seed)

    # ── Baseline θ₀: ERM on spurious training set ─────────────────────────
    baseline_ckpt = OUT_DIR / "baseline.pt"
    if baseline_ckpt.exists():
        print("\n[1/4] Loading cached baseline...", flush=True)
        ckpt = torch.load(baseline_ckpt, map_location=DEVICE)
        baseline = build_model_wb(pretrained=False)
        baseline.load_state_dict(ckpt["state_dict"])
        base_hist = ckpt["history"]
        base_test = ckpt["metrics"]
    else:
        print("\n[1/4] Training baseline (ERM on spurious data)...", flush=True)
        baseline = build_model_wb(pretrained=True)
        baseline, base_hist = train_wb_supervised(
            baseline, loaders["train"], loaders["val"], n_epochs=5, lr=1e-4)
        base_test = evaluate_wb(baseline, loaders["test"])
        torch.save({"state_dict": baseline.state_dict(), "history": base_hist,
                    "metrics": base_test}, baseline_ckpt)
    print(f"  Baseline test: avg={base_test['avg_acc']:.3f} "
          f"wg={base_test['worst_group_acc']:.3f}", flush=True)

    # ── Oracle θ*: group-balanced training (approximates intervened world) ─
    oracle_ckpt = OUT_DIR / "oracle.pt"
    if oracle_ckpt.exists():
        print("\n[2/4] Loading cached oracle...", flush=True)
        ckpt = torch.load(oracle_ckpt, map_location=DEVICE)
        oracle = build_model_wb(pretrained=False)
        oracle.load_state_dict(ckpt["state_dict"])
        oracle_hist = ckpt["history"]
        oracle_test = ckpt["metrics"]
    else:
        print("\n[2/4] Training oracle (group-balanced)...", flush=True)
        oracle = build_model_wb(pretrained=True)
        oracle, oracle_hist = train_wb_supervised(
            oracle, loaders["balanced_train"], loaders["val"], n_epochs=5, lr=1e-4)
        oracle_test = evaluate_wb(oracle, loaders["test"])
        torch.save({"state_dict": oracle.state_dict(), "history": oracle_hist,
                    "metrics": oracle_test}, oracle_ckpt)
    print(f"  Oracle test: avg={oracle_test['avg_acc']:.3f} "
          f"wg={oracle_test['worst_group_acc']:.3f}", flush=True)

    # ── Compute CE proxy on baseline (using CF-enabled test loader) ───────
    base_ce_metrics = evaluate_wb(baseline, loaders["test_cf"], oracle_model=oracle)
    base_ce = base_ce_metrics.get('ce_proxy')
    print(f"  Baseline CE proxy (test): {base_ce:.3f}" if base_ce is not None
          else "  Baseline CE proxy (test): N/A", flush=True)

    # ── Unlearning θ⁻ ─────────────────────────────────────────────────────
    print("\n[3/4] Post-hoc unlearning (λ=0.5, 3 epochs)...", flush=True)
    unlearn_model = build_model_wb(pretrained=False)
    unlearn_model.load_state_dict(deepcopy(baseline.state_dict()))
    unlearn_model, ul_hist = train_wb_unlearning(
        unlearn_model, baseline, loaders, n_epochs=3,
        lr=5e-4, lambda_ce=0.5, lambda_loc=1e-3, oracle_model=oracle)
    ul_test = evaluate_wb(unlearn_model, loaders["test_cf"], oracle_model=oracle)
    ul_ce = ul_test.get('ce_proxy')
    print(f"  Unlearned test: avg={ul_test['avg_acc']:.3f} "
          f"wg={ul_test['worst_group_acc']:.3f} "
          f"ce={ul_ce:.3f}" if ul_ce is not None else
          f"  Unlearned test: avg={ul_test['avg_acc']:.3f} "
          f"wg={ul_test['worst_group_acc']:.3f} ce=N/A", flush=True)
    torch.save({"state_dict": unlearn_model.state_dict(), "history": ul_hist,
                "metrics": ul_test},
               OUT_DIR / "unlearn_l05.pt")

    # ── Lambda sweep λ ∈ {0.0, 0.1, 0.5, 1.0} ────────────────────────────
    print("\n[4/4] Lambda sweep...", flush=True)
    sweep_results = []
    for lam in [0.0, 0.1, 0.5, 1.0]:
        m = build_model_wb(pretrained=False)
        m.load_state_dict(deepcopy(baseline.state_dict()))
        m, h = train_wb_unlearning(m, baseline, loaders, n_epochs=3,
                                   lr=5e-4, lambda_ce=lam, lambda_loc=1e-3,
                                   oracle_model=oracle)
        metrics = evaluate_wb(m, loaders["test_cf"], oracle_model=oracle)
        sweep_results.append({"lambda_ce": lam, "history": h, "metrics": metrics})
        lam_ce = metrics.get('ce_proxy')
        print(f"  λ={lam}: avg={metrics['avg_acc']:.3f} "
              f"wg={metrics['worst_group_acc']:.3f} "
              f"ce={lam_ce:.3f}" if lam_ce is not None else
              f"  λ={lam}: avg={metrics['avg_acc']:.3f} "
              f"wg={metrics['worst_group_acc']:.3f} ce=N/A", flush=True)

    summary = {
        "seed": seed,
        "baseline": {"history": base_hist, "metrics": base_test,
                     "ce_metrics": base_ce_metrics},
        "oracle": {"history": oracle_hist, "metrics": oracle_test},
        "unlearning_runs": sweep_results,
    }
    save_json(OUT_DIR / "summary.json", summary)
    return summary


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--gpu", type=int, default=4)
    args = p.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    t0 = time.perf_counter()
    summary = run_waterbirds_experiment(seed=args.seed)
    print(f"\nWaterbirds done in {time.perf_counter()-t0:.1f}s", flush=True)
