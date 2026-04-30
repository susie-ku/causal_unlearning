"""
CelebA blond/gender experiment with all baselines (mirrors Waterbirds protocol).

Y = Blond_Hair, C = Male.  4 groups: (Y, C) ∈ {(0,0), (0,1), (1,0), (1,1)}.
Counterfactual partner: same-Y, opposite-C sample.

Self-contained loader (bypasses torchvision.CelebA's gdrive dependency).
Expects:  data/celeba/img_align_celeba/{*.jpg}, list_attr_celeba.txt, list_eval_partition.txt.
"""
from __future__ import annotations
import os, sys, json, time, random, argparse
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
from torchvision.models import resnet50, ResNet50_Weights

from causal_unlearning.metrics import symmetric_kl
from causal_unlearning.utils import ensure_dir, save_json

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_ROOT = ROOT / "data" / "celeba"
IMG_DIR   = DATA_ROOT / "img_align_celeba"
ARTIFACT  = ROOT / "artifacts" / "celeba"

ATTR_Y = "Blond_Hair"
ATTR_C = "Male"


def _load_metadata():
    """Return DataFrame with columns: filename, split, y, c, group."""
    # Parse list_attr_celeba.txt:
    #   line 0: count, line 1: header (40 attribute names), then each line: filename val ... val
    attr_path = DATA_ROOT / "list_attr_celeba.txt"
    with open(attr_path) as f:
        n = int(f.readline().strip())
        header = f.readline().strip().split()
        rows = []
        for line in f:
            parts = line.strip().split()
            if not parts: continue
            rows.append(parts)
    attr_df = pd.DataFrame(rows, columns=["filename"] + header)
    # Convert -1/1 -> 0/1
    for h in header:
        attr_df[h] = (attr_df[h].astype(int) == 1).astype(int)

    # Parse list_eval_partition.txt: filename split
    part_path = DATA_ROOT / "list_eval_partition.txt"
    part_rows = []
    with open(part_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 2:
                part_rows.append((parts[0], int(parts[1])))
    part_df = pd.DataFrame(part_rows, columns=["filename", "split"])

    df = attr_df.merge(part_df, on="filename")
    df["y"] = df[ATTR_Y].astype(int)
    df["c"] = df[ATTR_C].astype(int)
    df["group"] = df["y"] * 2 + df["c"]
    return df


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


class CelebAGroupDataset(Dataset):
    def __init__(self, meta_df, transform, with_counterfactual=False, subset_n=None, seed=42):
        df = meta_df.reset_index(drop=True)
        if subset_n is not None and subset_n < len(df):
            rng = np.random.default_rng(seed)
            idx = rng.choice(len(df), size=subset_n, replace=False)
            df = df.iloc[idx].reset_index(drop=True)
        self.df = df
        self.transform = transform
        self.with_cf = with_counterfactual
        if with_counterfactual:
            self.idx_by_yc = {}
            for i, row in self.df.iterrows():
                key = (int(row.y), int(row.c))
                self.idx_by_yc.setdefault(key, []).append(i)

    def __len__(self):
        return len(self.df)

    def _load(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(IMG_DIR / row.filename).convert("RGB")
        return self.transform(img)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        y = int(row.y); c = int(row.c)
        item = {
            "image": self._load(idx),
            "label": torch.tensor(y, dtype=torch.long),
            "color": torch.tensor(c, dtype=torch.long),
            "group": torch.tensor(y * 2 + c, dtype=torch.long),
        }
        if self.with_cf:
            partners = self.idx_by_yc.get((y, 1 - c), [])
            cf_idx = random.choice(partners) if partners else idx
            item["counterfactual"] = self._load(cf_idx)
        return item


def build_loaders(batch_size=64, seed=42, train_subset=20000, eval_subset=5000):
    """Subset the train (and val/test) splits to keep wall-time tractable.
    Default: 20k train, 5k val, 5k test (still has all 4 groups)."""
    df = _load_metadata()
    train_df = df[df.split == 0].copy()
    val_df   = df[df.split == 1].copy()
    test_df  = df[df.split == 2].copy()
    print(f"  Full split sizes: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}", flush=True)

    train_ds = CelebAGroupDataset(train_df, TRAIN_TRANSFORM, with_counterfactual=True,
                                    subset_n=train_subset, seed=seed)
    val_ds   = CelebAGroupDataset(val_df,   EVAL_TRANSFORM, subset_n=eval_subset, seed=seed)
    test_ds  = CelebAGroupDataset(test_df,  EVAL_TRANSFORM, subset_n=eval_subset, seed=seed+1)
    test_cf  = CelebAGroupDataset(test_df,  EVAL_TRANSFORM, with_counterfactual=True,
                                    subset_n=eval_subset, seed=seed+1)

    print(f"  Subset sizes: train={len(train_ds)}, val={len(val_ds)}, test={len(test_ds)}", flush=True)
    g_counts = train_ds.df.group.value_counts().sort_index().tolist()
    print(f"  Train group counts: {g_counts}", flush=True)

    kw = dict(batch_size=batch_size, num_workers=4, pin_memory=True)
    g = torch.Generator(); g.manual_seed(seed)
    train_loader   = DataLoader(train_ds, shuffle=True, generator=g, **kw)
    val_loader     = DataLoader(val_ds,   shuffle=False, **kw)
    test_loader    = DataLoader(test_ds,  shuffle=False, **kw)
    test_cf_loader = DataLoader(test_cf,  shuffle=False, **kw)

    groups = train_ds.df.group.values
    counts = np.bincount(groups, minlength=4).astype(np.float32)
    counts = np.maximum(counts, 1)
    weights = 1.0 / counts[groups]
    sampler = WeightedRandomSampler(
        torch.from_numpy(weights).float(), num_samples=len(train_ds),
        replacement=True, generator=torch.Generator().manual_seed(seed))
    balanced_loader = DataLoader(train_ds, sampler=sampler, **kw)

    return {"train": train_loader, "val": val_loader, "test": test_loader,
            "test_cf": test_cf_loader, "balanced_train": balanced_loader,
            "train_ds": train_ds}


class ResNet50Model(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        backbone = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2 if pretrained else None)
        self.features = nn.Sequential(*list(backbone.children())[:-1])
        self.fc = nn.Linear(2048, 2)

    def get_features(self, x):
        return self.features(x).flatten(1)

    def forward(self, x):
        return self.fc(self.get_features(x))


def build_model(pretrained=True):
    return ResNet50Model(pretrained=pretrained).to(DEVICE)


@torch.inference_mode()
def evaluate(model, loader, oracle_model=None):
    model.eval()
    preds, labels_list, groups_list = [], [], []
    cf_kl_list, oracle_kl_list = [], []
    for batch in loader:
        imgs = batch["image"].to(DEVICE)
        labels = batch["label"].to(DEVICE)
        groups = batch["group"]
        logits = model(imgs)
        preds.append(logits.argmax(1).cpu()); labels_list.append(labels.cpu())
        groups_list.append(groups)
        if "counterfactual" in batch:
            cf_logits = model(batch["counterfactual"].to(DEVICE))
            cf_kl_list.append(symmetric_kl(logits, cf_logits).cpu())
        if oracle_model is not None:
            with torch.no_grad():
                ologit = oracle_model(imgs)
            kl_o = F.kl_div(F.log_softmax(logits, 1),
                             F.softmax(ologit, 1), reduction="none").sum(1)
            oracle_kl_list.append(kl_o.cpu())
    p = torch.cat(preds); l = torch.cat(labels_list); g = torch.cat(groups_list)
    avg = float((p == l).float().mean())
    group_accs = {}
    for gi in range(4):
        mask = g == gi
        if mask.sum() > 0:
            group_accs[str(gi)] = float((p[mask] == l[mask]).float().mean())
    wg = min(group_accs.values()) if group_accs else 0.0
    out = {"avg_acc": avg, "worst_group_acc": wg, "group_accs": group_accs}
    if cf_kl_list:
        out["ce_proxy"] = float(torch.cat(cf_kl_list).mean())
    if oracle_kl_list:
        out["fidelity_kl"] = float(torch.cat(oracle_kl_list).mean())
    return out


def locality_penalty(model, ref):
    pen = 0.0; n = 0
    for name, p in model.named_parameters():
        if name in ref:
            pen = pen + ((p - ref[name]) ** 2).mean(); n += 1
    return pen / max(n, 1)


def train_supervised(model, loader, val_loader, n_epochs=10, lr=1e-4):
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_epochs)
    history = []; best_val = -1.0; best_state = None
    for ep in range(1, n_epochs+1):
        model.train()
        tot_loss = tot_n = 0
        for batch in loader:
            x = batch["image"].to(DEVICE); y = batch["label"].to(DEVICE)
            opt.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model(x), y); loss.backward(); opt.step()
            tot_loss += float(loss)*x.shape[0]; tot_n += x.shape[0]
        sched.step()
        val_m = evaluate(model, val_loader)
        history.append({"epoch": ep, "loss": tot_loss/tot_n,
                         "val_avg": val_m["avg_acc"], "val_wg": val_m["worst_group_acc"]})
        print(f"  ep={ep:2d}: loss={tot_loss/tot_n:.4f} val_avg={val_m['avg_acc']:.3f} "
              f"val_wg={val_m['worst_group_acc']:.3f}", flush=True)
        if val_m["worst_group_acc"] > best_val:
            best_val = val_m["worst_group_acc"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, history


def train_unlearning(model, loaders, n_epochs=3, lr=5e-4,
                      lambda_ce=0.5, lambda_loc=1e-3):
    ref = {n: p.detach().clone() for n, p in model.named_parameters()}
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    history = []; best_val = -1.0; best_state = None
    for ep in range(1, n_epochs+1):
        model.train()
        tot_n = tot_ce = 0
        for batch in loaders["train"]:
            x = batch["image"].to(DEVICE); cf = batch["counterfactual"].to(DEVICE)
            y = batch["label"].to(DEVICE)
            opt.zero_grad(set_to_none=True)
            logits = model(x); cf_logits = model(cf)
            ce_term = symmetric_kl(logits, cf_logits).mean()
            loss = F.cross_entropy(logits, y) + lambda_ce * ce_term \
                    + lambda_loc * locality_penalty(model, ref)
            loss.backward(); opt.step()
            tot_ce += float(ce_term) * x.shape[0]; tot_n += x.shape[0]
        val_m = evaluate(model, loaders["val"])
        history.append({"epoch": ep, "ce": tot_ce/tot_n,
                         "val_avg": val_m["avg_acc"], "val_wg": val_m["worst_group_acc"]})
        print(f"  [unlearn λ={lambda_ce}] ep={ep}: ce={tot_ce/tot_n:.4f} "
              f"val_wg={val_m['worst_group_acc']:.3f}", flush=True)
        if val_m["worst_group_acc"] > best_val:
            best_val = val_m["worst_group_acc"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, history


def run_seed(seed, n_train=8, n_unlearn=3, train_subset=20000, eval_subset=5000):
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
    OUT = ARTIFACT / f"seed_{seed}"; ensure_dir(OUT)
    print(f"\n{'='*50}\nCelebA seed={seed}\n{'='*50}", flush=True)
    loaders = build_loaders(batch_size=64, seed=seed,
                              train_subset=train_subset, eval_subset=eval_subset)

    base_path = OUT / "baseline.pt"
    if base_path.exists():
        ckpt = torch.load(base_path, map_location=DEVICE, weights_only=False)
        baseline = build_model(pretrained=False)
        baseline.load_state_dict(ckpt["state_dict"])
        base_test = ckpt["metrics"]
        print(f"  loaded baseline test: avg={base_test['avg_acc']:.3f} wg={base_test['worst_group_acc']:.3f}", flush=True)
    else:
        print("[1] Training baseline (ERM)...", flush=True)
        baseline = build_model(pretrained=True)
        baseline, hist = train_supervised(baseline, loaders["train"], loaders["val"], n_epochs=n_train)
        base_test = evaluate(baseline, loaders["test"])
        torch.save({"state_dict": baseline.state_dict(), "history": hist,
                    "metrics": base_test}, base_path)
        print(f"  Baseline test: avg={base_test['avg_acc']:.3f} wg={base_test['worst_group_acc']:.3f}", flush=True)

    oracle_path = OUT / "oracle.pt"
    if oracle_path.exists():
        ckpt = torch.load(oracle_path, map_location=DEVICE, weights_only=False)
        oracle = build_model(pretrained=False)
        oracle.load_state_dict(ckpt["state_dict"])
        oracle_test = ckpt["metrics"]
        print(f"  loaded oracle test: avg={oracle_test['avg_acc']:.3f} wg={oracle_test['worst_group_acc']:.3f}", flush=True)
    else:
        print("[2] Training oracle (group-balanced)...", flush=True)
        oracle = build_model(pretrained=True)
        oracle, hist = train_supervised(oracle, loaders["balanced_train"], loaders["val"], n_epochs=n_train)
        oracle_test = evaluate(oracle, loaders["test"])
        torch.save({"state_dict": oracle.state_dict(), "history": hist,
                    "metrics": oracle_test}, oracle_path)
        print(f"  Oracle test: avg={oracle_test['avg_acc']:.3f} wg={oracle_test['worst_group_acc']:.3f}", flush=True)

    base_full = evaluate(baseline, loaders["test_cf"], oracle_model=oracle)

    results = {"seed": seed,
                "baseline": {"metrics": base_test, "ce_metrics": base_full},
                "oracle":   {"metrics": oracle_test}}

    for lam in [0.1, 0.5, 1.0]:
        path = OUT / f"ours_l{lam}.pt"
        if path.exists():
            ckpt = torch.load(path, map_location=DEVICE, weights_only=False)
            test_m = ckpt["metrics"]
        else:
            print(f"[ours λ={lam}]", flush=True)
            m = build_model(pretrained=False)
            m.load_state_dict(deepcopy(baseline.state_dict()))
            m, hist = train_unlearning(m, loaders, n_epochs=n_unlearn,
                                         lr=5e-4, lambda_ce=lam, lambda_loc=1e-3)
            test_m = evaluate(m, loaders["test_cf"], oracle_model=oracle)
            torch.save({"state_dict": m.state_dict(), "history": hist,
                        "metrics": test_m}, path)
        print(f"  ours λ={lam} test: avg={test_m['avg_acc']:.3f} wg={test_m['worst_group_acc']:.3f} "
              f"ce={test_m.get('ce_proxy', float('nan')):.3f}", flush=True)
        results[f"ours_l{lam}"] = {"metrics": test_m}

    save_json(OUT / "summary.json", results)
    return results


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n-train", type=int, default=8)
    p.add_argument("--n-unlearn", type=int, default=3)
    p.add_argument("--train-subset", type=int, default=20000)
    p.add_argument("--eval-subset", type=int, default=5000)
    args = p.parse_args()
    t0 = time.perf_counter()
    run_seed(args.seed, n_train=args.n_train, n_unlearn=args.n_unlearn,
              train_subset=args.train_subset, eval_subset=args.eval_subset)
    print(f"\nSeed {args.seed} done in {time.perf_counter()-t0:.1f}s", flush=True)
