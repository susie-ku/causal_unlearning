"""
Waterbirds v2 — proper multi-seed evaluation with all baselines.

Protocol:
  - ResNet-18, ImageNet pretrained, fc replaced with Linear(512,2)
  - 20 epochs max, AdamW lr=1e-4, cosine schedule, weight decay 1e-4
  - Early stopping on validation worst-group accuracy
  - 3 seeds {0, 42, 123}, identical seed for all methods
  - Counterfactual partner: same Y, opposite place (background) — approximation

Methods:
  - Baseline (ERM)
  - Oracle (group-balanced sampling)
  - Naive FT (lambda=0)
  - Intervened FT (group-balanced FT for 3 epochs)
  - GRL (progressive alpha schedule)
  - Concept Erasure (linear projection on penultimate features, then re-train head)
  - Oracle Distillation (KL match oracle on intervened batch)
  - Ours, lambda in {0.1, 0.5, 1.0}

Outputs: artifacts/waterbirds/seed_<S>/{baseline,oracle,...}.pt and summary.json
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
from torchvision.models import resnet18, ResNet18_Weights

from causal_unlearning.metrics import symmetric_kl
from causal_unlearning.utils import ensure_dir, save_json

DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_ROOT   = ROOT / "data" / "waterbird_complete95_forest2water2"
ARTIFACT_BASE = ROOT / "artifacts" / "waterbirds"

print(f"Using device: {DEVICE}", flush=True)


# ─────────────────────────────────────────────────────────────────────────────
# Transforms
# ─────────────────────────────────────────────────────────────────────────────
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
    def __init__(self, metadata, data_root, transform, with_counterfactual=False):
        self.meta = metadata.reset_index(drop=True)
        self.root = data_root
        self.transform = transform
        self.with_cf = with_counterfactual
        if with_counterfactual:
            self._cf_idx = {}
            for i, row in self.meta.iterrows():
                y_i, c_i = int(row.y), int(row.place)
                partners = self.meta[
                    (self.meta.y == y_i) & (self.meta.place == (1 - c_i))
                ].index.tolist()
                self._cf_idx[i] = partners

    def __len__(self):
        return len(self.meta)

    def __getitem__(self, idx):
        row = self.meta.iloc[idx]
        img = Image.open(self.root / row.img_filename).convert("RGB")
        item = {
            "image": self.transform(img),
            "label": torch.tensor(int(row.y), dtype=torch.long),
            "color": torch.tensor(int(row.place), dtype=torch.long),
            "group": torch.tensor(int(row.y) * 2 + int(row.place), dtype=torch.long),
        }
        if self.with_cf:
            partners = self._cf_idx.get(idx, [])
            cf_idx = random.choice(partners) if partners else idx
            cf_row = self.meta.iloc[cf_idx]
            cf_img = Image.open(self.root / cf_row.img_filename).convert("RGB")
            item["counterfactual"] = self.transform(cf_img)
        return item


def build_loaders(batch_size=64, seed=42):
    meta = pd.read_csv(DATA_ROOT / "metadata.csv")
    train_meta = meta[meta.split == 0].reset_index(drop=True)
    val_meta   = meta[meta.split == 1].reset_index(drop=True)
    test_meta  = meta[meta.split == 2].reset_index(drop=True)

    train_ds   = WaterbirdsDataset(train_meta, DATA_ROOT, TRAIN_TRANSFORM, with_counterfactual=True)
    val_ds     = WaterbirdsDataset(val_meta,   DATA_ROOT, EVAL_TRANSFORM)
    test_ds    = WaterbirdsDataset(test_meta,  DATA_ROOT, EVAL_TRANSFORM)
    test_cf_ds = WaterbirdsDataset(test_meta,  DATA_ROOT, EVAL_TRANSFORM, with_counterfactual=True)

    kw = dict(batch_size=batch_size, num_workers=4, pin_memory=True)
    g = torch.Generator(); g.manual_seed(seed)
    train_loader   = DataLoader(train_ds,   shuffle=True, generator=g, **kw)
    val_loader     = DataLoader(val_ds,     shuffle=False, **kw)
    test_loader    = DataLoader(test_ds,    shuffle=False, **kw)
    test_cf_loader = DataLoader(test_cf_ds, shuffle=False, **kw)

    groups = train_meta.apply(lambda r: int(r.y)*2 + int(r.place), axis=1).values
    group_counts = np.bincount(groups)
    weights = 1.0 / group_counts[groups]
    sampler = WeightedRandomSampler(
        torch.from_numpy(weights).float(), num_samples=len(train_ds),
        replacement=True, generator=torch.Generator().manual_seed(seed))
    balanced_loader = DataLoader(train_ds, sampler=sampler, **kw)

    return {"train": train_loader, "val": val_loader, "test": test_loader,
            "test_cf": test_cf_loader, "balanced_train": balanced_loader,
            "train_meta": train_meta}


# ─────────────────────────────────────────────────────────────────────────────
# Model with feature hook
# ─────────────────────────────────────────────────────────────────────────────
class ResNet18WB(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        backbone = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
        self.features = nn.Sequential(*list(backbone.children())[:-1])  # up to avgpool
        self.fc = nn.Linear(512, 2)

    def get_features(self, x):
        return self.features(x).flatten(1)

    def forward(self, x):
        return self.fc(self.get_features(x))


def build_model(pretrained=True):
    return ResNet18WB(pretrained=pretrained).to(DEVICE)


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation
# ─────────────────────────────────────────────────────────────────────────────
@torch.inference_mode()
def evaluate(model, loader, oracle_model=None):
    model.eval()
    preds, labels_list, groups_list = [], [], []
    cf_kl_list, oracle_kl_list = [], []

    for batch in loader:
        imgs   = batch["image"].to(DEVICE)
        labels = batch["label"].to(DEVICE)
        groups = batch["group"]
        logits = model(imgs)
        preds.append(logits.argmax(1).cpu())
        labels_list.append(labels.cpu())
        groups_list.append(groups)

        if "counterfactual" in batch:
            cf = batch["counterfactual"].to(DEVICE)
            cf_logits = model(cf)
            cf_kl_list.append(symmetric_kl(logits, cf_logits).cpu())

        if oracle_model is not None:
            with torch.no_grad():
                oracle_logits = oracle_model(imgs)
            kl_o = F.kl_div(F.log_softmax(logits, 1),
                             F.softmax(oracle_logits, 1), reduction="none").sum(1)
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


# ─────────────────────────────────────────────────────────────────────────────
# Locality penalty (for fine-tune methods)
# ─────────────────────────────────────────────────────────────────────────────
def locality_penalty(model, ref_params):
    pen = 0.0; n = 0
    for name, p in model.named_parameters():
        if name in ref_params:
            pen = pen + ((p - ref_params[name]) ** 2).mean()
            n += 1
    return pen / max(n, 1)


# ─────────────────────────────────────────────────────────────────────────────
# Training routines
# ─────────────────────────────────────────────────────────────────────────────
def train_supervised(model, loader, val_loader, n_epochs=20, lr=1e-4,
                     early_stop_metric="worst_group_acc", verbose=True):
    """Train from-scratch ERM/oracle, save best ckpt by val worst-group acc."""
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_epochs)
    history = []
    best_val = -1.0
    best_state = None
    for ep in range(1, n_epochs+1):
        model.train()
        tot_loss = tot_n = 0
        for batch in loader:
            imgs = batch["image"].to(DEVICE); labels = batch["label"].to(DEVICE)
            opt.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model(imgs), labels)
            loss.backward(); opt.step()
            tot_loss += float(loss) * imgs.shape[0]; tot_n += imgs.shape[0]
        sched.step()
        val_m = evaluate(model, val_loader)
        history.append({"epoch": ep, "train_loss": tot_loss/tot_n,
                        "val_avg": val_m["avg_acc"], "val_wg": val_m["worst_group_acc"]})
        if verbose:
            print(f"  ep={ep:2d}: loss={tot_loss/tot_n:.4f} val_avg={val_m['avg_acc']:.3f} "
                  f"val_wg={val_m['worst_group_acc']:.3f}", flush=True)
        if val_m[early_stop_metric] > best_val:
            best_val = val_m[early_stop_metric]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, history, best_val


def train_unlearning(model, baseline_model, loaders, n_epochs=5, lr=5e-4,
                     lambda_ce=0.5, lambda_loc=1e-3, oracle_model=None, verbose=True):
    """Composite loss fine-tune."""
    ref_params = {n: p.detach().clone() for n, p in model.named_parameters()}
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    history = []; best_val = -1.0; best_state = None
    for ep in range(1, n_epochs+1):
        model.train()
        tot_ret = tot_ce = tot_loc = tot_n = 0
        for batch in loaders["train"]:
            x  = batch["image"].to(DEVICE); cf = batch["counterfactual"].to(DEVICE)
            y  = batch["label"].to(DEVICE)
            opt.zero_grad(set_to_none=True)
            logits = model(x); cf_logits = model(cf)
            ret = F.cross_entropy(logits, y)
            ce  = symmetric_kl(logits, cf_logits).mean()
            loc = locality_penalty(model, ref_params)
            loss = ret + lambda_ce * ce + lambda_loc * loc
            loss.backward(); opt.step()
            tot_ret += float(ret)*x.shape[0]; tot_ce += float(ce)*x.shape[0]
            tot_loc += float(loc)*x.shape[0]; tot_n += x.shape[0]
        val_m = evaluate(model, loaders["val"])
        history.append({"epoch": ep, "ret": tot_ret/tot_n, "ce": tot_ce/tot_n,
                        "loc": tot_loc/tot_n, "val_avg": val_m["avg_acc"],
                        "val_wg": val_m["worst_group_acc"]})
        if verbose:
            print(f"  [unlearn λ={lambda_ce}] ep={ep}: ret={tot_ret/tot_n:.3f} "
                  f"ce={tot_ce/tot_n:.4f} val_wg={val_m['worst_group_acc']:.3f}", flush=True)
        if val_m["worst_group_acc"] > best_val:
            best_val = val_m["worst_group_acc"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, history


def train_intervened_ft(model, loaders, n_epochs=5, lr=5e-4, lambda_loc=1e-3, verbose=True):
    """Fine-tune on group-balanced data with same locality penalty."""
    ref_params = {n: p.detach().clone() for n, p in model.named_parameters()}
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    history = []; best_val = -1.0; best_state = None
    for ep in range(1, n_epochs+1):
        model.train()
        tot_loss = tot_n = 0
        for batch in loaders["balanced_train"]:
            x = batch["image"].to(DEVICE); y = batch["label"].to(DEVICE)
            opt.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model(x), y) + lambda_loc * locality_penalty(model, ref_params)
            loss.backward(); opt.step()
            tot_loss += float(loss)*x.shape[0]; tot_n += x.shape[0]
        val_m = evaluate(model, loaders["val"])
        history.append({"epoch": ep, "loss": tot_loss/tot_n,
                        "val_avg": val_m["avg_acc"], "val_wg": val_m["worst_group_acc"]})
        if verbose:
            print(f"  [IFT] ep={ep}: loss={tot_loss/tot_n:.3f} val_wg={val_m['worst_group_acc']:.3f}", flush=True)
        if val_m["worst_group_acc"] > best_val:
            best_val = val_m["worst_group_acc"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, history


def train_grl(model, loaders, n_epochs=5, lr=5e-4, peak_alpha=1.0, verbose=True):
    """Adversarial debiasing via gradient reversal on background discriminator."""
    from causal_unlearning.baselines import GradientReversalFunction
    discriminator = nn.Sequential(nn.Linear(512, 64), nn.ReLU(),
                                   nn.Linear(64, 2)).to(DEVICE)
    opt_main = torch.optim.AdamW(model.parameters(), lr=lr)
    opt_disc = torch.optim.AdamW(discriminator.parameters(), lr=lr)
    history = []; best_val = -1.0; best_state = None
    total_steps = n_epochs * len(loaders["train"]); step = 0
    for ep in range(1, n_epochs+1):
        model.train(); discriminator.train()
        tot_main = tot_adv = tot_n = 0
        for batch in loaders["train"]:
            x = batch["image"].to(DEVICE); y = batch["label"].to(DEVICE)
            c = batch["color"].to(DEVICE)
            p = step / max(total_steps, 1); step += 1
            alpha = peak_alpha * (2.0 / (1.0 + np.exp(-10.0 * p)) - 1.0)
            feat = model.get_features(x)
            logits = model.fc(feat)
            main_loss = F.cross_entropy(logits, y)
            rev_feat = GradientReversalFunction.apply(feat, alpha)
            adv_loss = F.cross_entropy(discriminator(rev_feat), c)
            loss = main_loss + adv_loss
            opt_main.zero_grad(set_to_none=True); opt_disc.zero_grad(set_to_none=True)
            loss.backward(); opt_main.step(); opt_disc.step()
            tot_main += float(main_loss)*x.shape[0]
            tot_adv += float(adv_loss)*x.shape[0]; tot_n += x.shape[0]
        val_m = evaluate(model, loaders["val"])
        history.append({"epoch": ep, "main": tot_main/tot_n, "adv": tot_adv/tot_n,
                        "val_avg": val_m["avg_acc"], "val_wg": val_m["worst_group_acc"]})
        if verbose:
            print(f"  [GRL] ep={ep}: main={tot_main/tot_n:.3f} adv={tot_adv/tot_n:.3f} "
                  f"val_wg={val_m['worst_group_acc']:.3f}", flush=True)
        if val_m["worst_group_acc"] > best_val:
            best_val = val_m["worst_group_acc"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, history


def train_distillation(model, oracle_model, loaders, n_epochs=5, lr=5e-4,
                        lambda_loc=1e-3, T=2.0, verbose=True):
    """Match oracle outputs on intervened (balanced) data."""
    ref_params = {n: p.detach().clone() for n, p in model.named_parameters()}
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    history = []; best_val = -1.0; best_state = None
    for ep in range(1, n_epochs+1):
        model.train()
        tot_loss = tot_n = 0
        for batch in loaders["balanced_train"]:
            x = batch["image"].to(DEVICE); y = batch["label"].to(DEVICE)
            opt.zero_grad(set_to_none=True)
            with torch.no_grad():
                oracle_logits = oracle_model(x)
            student_logits = model(x)
            kl = F.kl_div(F.log_softmax(student_logits/T, 1),
                          F.softmax(oracle_logits/T, 1), reduction="batchmean") * T**2
            ce = F.cross_entropy(student_logits, y)
            loc = locality_penalty(model, ref_params)
            loss = ce + kl + lambda_loc * loc
            loss.backward(); opt.step()
            tot_loss += float(loss)*x.shape[0]; tot_n += x.shape[0]
        val_m = evaluate(model, loaders["val"])
        history.append({"epoch": ep, "loss": tot_loss/tot_n,
                        "val_avg": val_m["avg_acc"], "val_wg": val_m["worst_group_acc"]})
        if verbose:
            print(f"  [Distill] ep={ep}: loss={tot_loss/tot_n:.3f} val_wg={val_m['worst_group_acc']:.3f}", flush=True)
        if val_m["worst_group_acc"] > best_val:
            best_val = val_m["worst_group_acc"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, history


# ─────────────────────────────────────────────────────────────────────────────
# Concept Erasure (linear projection on penultimate features)
# ─────────────────────────────────────────────────────────────────────────────
class ProjectedHead(nn.Module):
    """Wraps a backbone with a projection P that removes a concept direction,
    followed by a (re-trained) linear classifier."""
    def __init__(self, backbone, P, num_classes=2):
        super().__init__()
        self.backbone = backbone   # ResNet18WB (features + fc); only features used
        self.register_buffer("P", P.detach())
        self.head = nn.Linear(P.shape[0], num_classes).to(P.device)

    def get_features(self, x):
        feat = self.backbone.get_features(x)
        return feat @ self.P.t()

    def forward(self, x):
        return self.head(self.get_features(x))


def train_concept_erasure(baseline, loaders, n_epochs_head=3, lr=1e-3, verbose=True):
    """LEACE-style baseline: project features orthogonal to concept direction, retrain head."""
    from sklearn.linear_model import LogisticRegression

    # 1. Extract features and concept labels on train set
    baseline.eval()
    feats, cs = [], []
    with torch.inference_mode():
        for batch in loaders["train"]:
            f = baseline.get_features(batch["image"].to(DEVICE)).cpu()
            feats.append(f); cs.append(batch["color"])
    F_train = torch.cat(feats).numpy()
    C_train = torch.cat(cs).numpy()

    # 2. Fit logistic regression to predict concept
    lr_model = LogisticRegression(max_iter=200).fit(F_train, C_train)
    w = torch.from_numpy(lr_model.coef_[0]).float().to(DEVICE)
    w = w / (w.norm() + 1e-8)

    # 3. Build orthogonal projection P = I - w w^T (rank d-1)
    d = w.shape[0]
    P = torch.eye(d, device=DEVICE) - torch.outer(w, w)

    # 4. Wrap and train new head on projected features
    erased = ProjectedHead(baseline, P, num_classes=2).to(DEVICE)
    for p in erased.backbone.parameters():
        p.requires_grad_(False)
    opt = torch.optim.AdamW(erased.head.parameters(), lr=lr, weight_decay=1e-4)
    history = []; best_val = -1.0; best_state = None
    for ep in range(1, n_epochs_head+1):
        erased.train()
        tot_loss = tot_n = 0
        for batch in loaders["train"]:
            x = batch["image"].to(DEVICE); y = batch["label"].to(DEVICE)
            opt.zero_grad(set_to_none=True)
            loss = F.cross_entropy(erased(x), y)
            loss.backward(); opt.step()
            tot_loss += float(loss)*x.shape[0]; tot_n += x.shape[0]
        val_m = evaluate(erased, loaders["val"])
        history.append({"epoch": ep, "loss": tot_loss/tot_n,
                        "val_avg": val_m["avg_acc"], "val_wg": val_m["worst_group_acc"]})
        if verbose:
            print(f"  [Erase] ep={ep}: loss={tot_loss/tot_n:.3f} val_wg={val_m['worst_group_acc']:.3f}", flush=True)
        if val_m["worst_group_acc"] > best_val:
            best_val = val_m["worst_group_acc"]
            best_state = {k: v.detach().cpu().clone() for k, v in erased.state_dict().items()}
    if best_state is not None:
        erased.load_state_dict(best_state)
    return erased, history


# ─────────────────────────────────────────────────────────────────────────────
# Main pipeline
# ─────────────────────────────────────────────────────────────────────────────
def run_seed(seed, n_train_epochs=20, n_unlearn_epochs=5, force=False):
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
    OUT = ARTIFACT_BASE / f"seed_{seed}"; ensure_dir(OUT)

    print(f"\n{'='*60}\nWaterbirds seed={seed}\n{'='*60}", flush=True)
    loaders = build_loaders(batch_size=64, seed=seed)

    # ── Baseline ERM ──────────────────────────────────────────────────────
    base_path = OUT / "baseline.pt"
    if base_path.exists() and not force:
        print("[1] Loading cached baseline...", flush=True)
        ckpt = torch.load(base_path, map_location=DEVICE)
        baseline = build_model(pretrained=False)
        baseline.load_state_dict(ckpt["state_dict"])
        base_test = ckpt["metrics"]
    else:
        print("[1] Training baseline (ERM)...", flush=True)
        baseline = build_model(pretrained=True)
        baseline, base_hist, _ = train_supervised(
            baseline, loaders["train"], loaders["val"], n_epochs=n_train_epochs)
        base_test = evaluate(baseline, loaders["test"])
        torch.save({"state_dict": baseline.state_dict(), "history": base_hist,
                    "metrics": base_test}, base_path)
    print(f"  Baseline test: avg={base_test['avg_acc']:.3f} wg={base_test['worst_group_acc']:.3f}", flush=True)

    # ── Oracle (group-balanced) ───────────────────────────────────────────
    oracle_path = OUT / "oracle.pt"
    if oracle_path.exists() and not force:
        print("[2] Loading cached oracle...", flush=True)
        ckpt = torch.load(oracle_path, map_location=DEVICE)
        oracle = build_model(pretrained=False)
        oracle.load_state_dict(ckpt["state_dict"])
        oracle_test = ckpt["metrics"]
    else:
        print("[2] Training oracle (group-balanced)...", flush=True)
        oracle = build_model(pretrained=True)
        oracle, oracle_hist, _ = train_supervised(
            oracle, loaders["balanced_train"], loaders["val"], n_epochs=n_train_epochs)
        oracle_test = evaluate(oracle, loaders["test"])
        torch.save({"state_dict": oracle.state_dict(), "history": oracle_hist,
                    "metrics": oracle_test}, oracle_path)
    print(f"  Oracle test: avg={oracle_test['avg_acc']:.3f} wg={oracle_test['worst_group_acc']:.3f}", flush=True)

    # CE proxy + fidelity for baseline
    base_full = evaluate(baseline, loaders["test_cf"], oracle_model=oracle)

    results = {
        "seed": seed,
        "baseline": {"metrics": base_test, "ce_metrics": base_full},
        "oracle":   {"metrics": oracle_test},
    }

    def run_method(name, model_fn, ckpt_name):
        path = OUT / ckpt_name
        if path.exists() and not force:
            print(f"[{name}] Loading cached...", flush=True)
            ckpt = torch.load(path, map_location=DEVICE)
            m = build_model(pretrained=False)
            try:
                m.load_state_dict(ckpt["state_dict"])
            except RuntimeError:
                # ProjectedHead has different state dict; skip caching
                m = None
            test_m = ckpt["metrics"]
        else:
            print(f"[{name}] Training...", flush=True)
            m, hist = model_fn()
            test_m = evaluate(m, loaders["test_cf"], oracle_model=oracle)
            try:
                state_dict = m.state_dict()
                torch.save({"state_dict": state_dict, "history": hist,
                            "metrics": test_m}, path)
            except Exception:
                save_json(path.with_suffix(".json"),
                          {"history": hist, "metrics": test_m})
        print(f"  {name} test: avg={test_m['avg_acc']:.3f} wg={test_m['worst_group_acc']:.3f} "
              f"ce={test_m.get('ce_proxy', float('nan')):.3f}", flush=True)
        results[name] = {"metrics": test_m}
        return test_m

    # ── Naive FT (lambda=0) ───────────────────────────────────────────────
    def run_naive():
        m = build_model(pretrained=False); m.load_state_dict(deepcopy(baseline.state_dict()))
        return train_unlearning(m, baseline, loaders, n_epochs=n_unlearn_epochs,
                                  lr=5e-4, lambda_ce=0.0)
    run_method("naive_ft", run_naive, "naive_ft.pt")

    # ── Intervened FT ─────────────────────────────────────────────────────
    def run_ift():
        m = build_model(pretrained=False); m.load_state_dict(deepcopy(baseline.state_dict()))
        return train_intervened_ft(m, loaders, n_epochs=n_unlearn_epochs)
    run_method("intervened_ft", run_ift, "intervened_ft.pt")

    # ── GRL (progressive alpha) ───────────────────────────────────────────
    def run_grl_method():
        m = build_model(pretrained=False); m.load_state_dict(deepcopy(baseline.state_dict()))
        return train_grl(m, loaders, n_epochs=n_unlearn_epochs, peak_alpha=1.0)
    run_method("grl", run_grl_method, "grl.pt")

    # ── Concept Erasure ───────────────────────────────────────────────────
    print("[concept_erasure] Training...", flush=True)
    erased, erased_hist = train_concept_erasure(baseline, loaders, n_epochs_head=n_unlearn_epochs)
    erased_test = evaluate(erased, loaders["test_cf"], oracle_model=oracle)
    save_json(OUT / "concept_erasure.json", {"history": erased_hist, "metrics": erased_test})
    print(f"  concept_erasure test: avg={erased_test['avg_acc']:.3f} wg={erased_test['worst_group_acc']:.3f} "
          f"ce={erased_test.get('ce_proxy', float('nan')):.3f}", flush=True)
    results["concept_erasure"] = {"metrics": erased_test}

    # ── Oracle Distillation ───────────────────────────────────────────────
    def run_distill():
        m = build_model(pretrained=False); m.load_state_dict(deepcopy(baseline.state_dict()))
        return train_distillation(m, oracle, loaders, n_epochs=n_unlearn_epochs)
    run_method("distillation", run_distill, "distillation.pt")

    # ── Ours, lambda sweep ────────────────────────────────────────────────
    for lam in [0.1, 0.5, 1.0]:
        def run_ours(lam=lam):
            m = build_model(pretrained=False); m.load_state_dict(deepcopy(baseline.state_dict()))
            return train_unlearning(m, baseline, loaders, n_epochs=n_unlearn_epochs,
                                      lr=5e-4, lambda_ce=lam)
        run_method(f"ours_l{lam}", run_ours, f"ours_l{lam}.pt")

    save_json(OUT / "summary.json", results)
    return results


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--train-epochs", type=int, default=20)
    p.add_argument("--unlearn-epochs", type=int, default=5)
    p.add_argument("--force", action="store_true")
    args = p.parse_args()

    t0 = time.perf_counter()
    run_seed(args.seed, n_train_epochs=args.train_epochs,
             n_unlearn_epochs=args.unlearn_epochs, force=args.force)
    print(f"\nSeed {args.seed} done in {time.perf_counter()-t0:.1f}s", flush=True)
