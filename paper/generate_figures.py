"""
Generate publication-quality figures for the NeurIPS paper.
Run from the paper/ directory:
    python generate_figures.py

Requires: matplotlib, numpy (already in the conda env)
Outputs figures/ directory with PDF files.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ── Style ──────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.titlesize": 10,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.dpi": 300,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linestyle": "--",
})

PALETTE = {
    "baseline": "#E05C5C",   # red
    "oracle":   "#4CAF50",   # green
    "l0":       "#9E9E9E",   # grey
    "l01":      "#64B5F6",   # light blue
    "l05":      "#1976D2",   # medium blue
    "l10":      "#0D47A1",   # dark blue
}

FIGURES_DIR = Path(__file__).parent / "figures"
FIGURES_DIR.mkdir(exist_ok=True)

ROOT = Path(__file__).parent.parent

SUMMARY_PATH = (
    Path(__file__).parent.parent / "artifacts" / "default" / "summary.json"
)


def load_summary() -> dict:
    with open(SUMMARY_PATH) as f:
        return json.load(f)


# ── Figure 1: Model Comparison Bar Chart ──────────────────────────────────
def fig_model_comparison(summary: dict) -> None:
    """Grouped bar chart: obs vs int accuracy for all six models."""

    models = [
        ("Baseline $\\theta_0$",   summary["baseline"]["metrics"], PALETTE["baseline"]),
        ("Oracle $\\theta^*$",     summary["oracle"]["metrics"],   PALETTE["oracle"]),
    ]
    for run in summary["unlearning_runs"]:
        lam = run["lambda_ce"]
        label = f"Unlearn $\\lambda={lam}$"
        key = f"l{str(lam).replace('.','')}"
        models.append((label, run["metrics"], PALETTE.get(key, "#666")))

    names     = [m[0] for m in models]
    obs_accs  = [m[1]["observational_accuracy"] * 100 for m in models]
    int_accs  = [m[1]["intervened_accuracy"]    * 100 for m in models]

    x = np.arange(len(names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(7.0, 3.4))

    bars_obs = ax.bar(x - width / 2, obs_accs, width,
                      label="Observational accuracy", color=[m[2] for m in models],
                      alpha=0.85, edgecolor="white", linewidth=0.5)
    bars_int = ax.bar(x + width / 2, int_accs, width,
                      label="Intervened accuracy",    color=[m[2] for m in models],
                      alpha=0.45, edgecolor="white", linewidth=0.5,
                      hatch="//")

    # value labels
    for bar in bars_obs:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.6,
                f"{bar.get_height():.1f}", ha="center", va="bottom", fontsize=7)
    for bar in bars_int:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.6,
                f"{bar.get_height():.1f}", ha="center", va="bottom", fontsize=7)

    ax.set_ylabel("Accuracy (%)")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=18, ha="right")
    ax.set_ylim(0, 100)
    ax.set_yticks(range(0, 101, 20))

    # Legend
    solid_patch  = mpatches.Patch(facecolor="gray", alpha=0.85, label="Observational")
    hatch_patch  = mpatches.Patch(facecolor="gray", alpha=0.45, hatch="//",
                                  edgecolor="gray", label="Intervened")
    ax.legend(handles=[solid_patch, hatch_patch], loc="lower right")

    # Dashed reference line at oracle int acc
    oracle_int = summary["oracle"]["metrics"]["intervened_accuracy"] * 100
    ax.axhline(oracle_int, color=PALETTE["oracle"], linestyle="--", linewidth=1.0,
               label=f"Oracle int. ({oracle_int:.1f}%)")
    ax.text(len(names) - 0.5, oracle_int + 0.8, f"Oracle int. ({oracle_int:.1f}%)",
            color=PALETTE["oracle"], fontsize=7, ha="right")

    fig.tight_layout()
    out = FIGURES_DIR / "model_comparison.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


# ── Figure 2: Lambda Sweep (3 panels) ─────────────────────────────────────
def fig_lambda_sweep(summary: dict) -> None:
    """3-panel figure: CE proxy, intervened accuracy, fidelity vs lambda."""

    runs   = summary["unlearning_runs"]
    lambdas = [r["lambda_ce"]                              for r in runs]
    ce      = [r["metrics"]["causal_effect_proxy"]         for r in runs]
    int_acc = [r["metrics"]["intervened_accuracy"] * 100   for r in runs]
    fid     = [r["metrics"]["fidelity_to_oracle_kl"]       for r in runs]

    # Reference lines
    baseline_ce  = summary["baseline"]["metrics"]["causal_effect_proxy"]
    oracle_ce    = summary["oracle"]["metrics"]["causal_effect_proxy"]
    baseline_int = summary["baseline"]["metrics"]["intervened_accuracy"] * 100
    oracle_int   = summary["oracle"]["metrics"]["intervened_accuracy"]   * 100

    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.8))

    # ── Panel 1: CE Proxy ──
    ax = axes[0]
    ax.plot(lambdas, ce, marker="o", color="#1976D2", linewidth=1.8, markersize=6,
            zorder=3, label="Unlearned $\\theta^-$")
    ax.axhline(baseline_ce, color=PALETTE["baseline"], linestyle="--", linewidth=1.0,
               label=f"Baseline ({baseline_ce:.2f})")
    ax.axhline(oracle_ce,   color=PALETTE["oracle"],   linestyle="--", linewidth=1.0,
               label=f"Oracle ({oracle_ce:.3f})")
    ax.fill_between(lambdas, oracle_ce, ce, alpha=0.10, color="#1976D2")
    ax.set_xlabel("$\\lambda$")
    ax.set_ylabel("$\\mathrm{CE}_C(\\theta)$ (Sym. KL)")
    ax.set_title("(a) Causal Effect Proxy")
    ax.legend(fontsize=7, loc="upper right")
    ax.set_xticks(lambdas)

    # ── Panel 2: Intervened Accuracy ──
    ax = axes[1]
    ax.plot(lambdas, int_acc, marker="s", color="#1976D2", linewidth=1.8, markersize=6,
            zorder=3)
    ax.axhline(baseline_int, color=PALETTE["baseline"], linestyle="--", linewidth=1.0,
               label=f"Baseline ({baseline_int:.1f}%)")
    ax.axhline(oracle_int,   color=PALETTE["oracle"],   linestyle="--", linewidth=1.0,
               label=f"Oracle ({oracle_int:.1f}%)")
    ax.fill_between(lambdas, baseline_int, int_acc, alpha=0.12, color="#4CAF50")
    ax.set_xlabel("$\\lambda$")
    ax.set_ylabel("Intervened accuracy (%)")
    ax.set_title("(b) Intervened Accuracy")
    ax.set_ylim(40, 92)
    ax.legend(fontsize=7, loc="lower right")
    ax.set_xticks(lambdas)

    # ── Panel 3: Fidelity to Oracle ──
    ax = axes[2]
    ax.plot(lambdas, fid, marker="^", color="#E65100", linewidth=1.8, markersize=6,
            zorder=3)
    ax.set_xlabel("$\\lambda$")
    ax.set_ylabel("$\\mathcal{D}_{\\mathrm{fid}}$ (KL to oracle)")
    ax.set_title("(c) Fidelity to Oracle")
    ax.set_xticks(lambdas)
    # Annotate best lambda
    best_idx = int(np.argmin(fid))
    ax.annotate(f"$\\lambda={lambdas[best_idx]}$\n(best fid.)",
                xy=(lambdas[best_idx], fid[best_idx]),
                xytext=(lambdas[best_idx] + 0.15, fid[best_idx] - 0.04),
                fontsize=7, color="#E65100",
                arrowprops=dict(arrowstyle="->", color="#E65100", lw=0.8))

    fig.tight_layout()
    out = FIGURES_DIR / "lambda_sweep.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


# ── Figure 3: Training Curves ──────────────────────────────────────────────
def fig_training_curves(summary: dict) -> None:
    """Loss and CE proxy over training epochs for baseline and oracle."""

    b_hist = summary["baseline"]["history"]
    o_hist = summary["oracle"]["history"]

    epochs_b = [h["epoch"] for h in b_hist]
    epochs_o = [h["epoch"] for h in o_hist]

    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.6))

    # ── Training Loss ──
    ax = axes[0]
    ax.plot(epochs_b, [h["train_loss"]      for h in b_hist],
            marker="o", color=PALETTE["baseline"], label="Baseline $\\theta_0$",
            linewidth=1.5, markersize=5)
    ax.plot(epochs_o, [h["train_loss"]      for h in o_hist],
            marker="s", color=PALETTE["oracle"],   label="Oracle $\\theta^*$",
            linewidth=1.5, markersize=5)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Training loss")
    ax.set_title("(a) Training Loss")
    ax.legend(fontsize=8)
    ax.set_xticks(epochs_b)

    # ── Train Accuracy ──
    ax = axes[1]
    ax.plot(epochs_b, [h["train_accuracy"] * 100 for h in b_hist],
            marker="o", color=PALETTE["baseline"], linewidth=1.5, markersize=5)
    ax.plot(epochs_o, [h["train_accuracy"] * 100 for h in o_hist],
            marker="s", color=PALETTE["oracle"],   linewidth=1.5, markersize=5)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Train accuracy (%)")
    ax.set_title("(b) Train Accuracy")
    ax.set_ylim(0, 100)
    ax.set_xticks(epochs_b)

    # ── CE Proxy over training ──
    ax = axes[2]
    ax.plot(epochs_b, [h["causal_effect_proxy"] for h in b_hist],
            marker="o", color=PALETTE["baseline"], linewidth=1.5, markersize=5,
            label="Baseline $\\theta_0$")
    ax.plot(epochs_o, [h["causal_effect_proxy"] for h in o_hist],
            marker="s", color=PALETTE["oracle"],   linewidth=1.5, markersize=5,
            label="Oracle $\\theta^*$")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("$\\mathrm{CE}_C(\\theta)$")
    ax.set_title("(c) Causal Effect Proxy")
    ax.legend(fontsize=8)
    ax.set_xticks(epochs_b)

    fig.tight_layout()
    out = FIGURES_DIR / "training_curves.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


# ── Figure 4: Pareto Frontier ──────────────────────────────────────────────
def fig_pareto(summary: dict) -> None:
    """Scatter: CE proxy vs intervened accuracy (Pareto-style trade-off)."""

    fig, ax = plt.subplots(figsize=(4.8, 3.4))
    fig.subplots_adjust(left=0.18, right=0.97, bottom=0.15, top=0.92)

    # Baseline
    bm = summary["baseline"]["metrics"]
    ax.scatter(bm["causal_effect_proxy"], bm["intervened_accuracy"] * 100,
               marker="*", s=200, color=PALETTE["baseline"], zorder=5,
               label="Baseline $\\theta_0$")
    ax.annotate("$\\theta_0$",
                xy=(bm["causal_effect_proxy"], bm["intervened_accuracy"] * 100),
                xytext=(bm["causal_effect_proxy"] - 0.22, bm["intervened_accuracy"] * 100 + 2.0),
                fontsize=8.5, arrowprops=dict(arrowstyle="-", color="gray", lw=0.5))

    # Oracle
    om = summary["oracle"]["metrics"]
    ax.scatter(om["causal_effect_proxy"], om["intervened_accuracy"] * 100,
               marker="*", s=200, color=PALETTE["oracle"], zorder=5,
               label="Oracle $\\theta^*$")
    ax.annotate("$\\theta^*$",
                xy=(om["causal_effect_proxy"], om["intervened_accuracy"] * 100),
                xytext=(om["causal_effect_proxy"] + 0.12, om["intervened_accuracy"] * 100 - 3.5),
                fontsize=8.5, arrowprops=dict(arrowstyle="-", color="gray", lw=0.5))

    # Unlearning sweep — per-lambda offsets; λ=0.5 and λ=1.0 staggered (same y)
    colors_sweep = [PALETTE["l0"], PALETTE["l01"], PALETTE["l05"], PALETTE["l10"]]
    sweep_offsets = {
        0.0: (+0.06, -3.5),   # below-right
        0.1: (+0.07, +2.0),   # above-right
        0.5: (+0.07, +2.0),   # above-right
        1.0: (-0.06, -3.5),   # below-left  ← avoids λ=0.5 label
    }
    for run, c in zip(summary["unlearning_runs"], colors_sweep):
        m = run["metrics"]
        lam = run["lambda_ce"]
        ax.scatter(m["causal_effect_proxy"], m["intervened_accuracy"] * 100,
                   marker="D", s=80, color=c, zorder=5)
        dx, dy = sweep_offsets.get(lam, (+0.06, +1.2))
        ax.annotate(f"$\\lambda={lam}$",
                    xy=(m["causal_effect_proxy"], m["intervened_accuracy"] * 100),
                    xytext=(m["causal_effect_proxy"] + dx,
                            m["intervened_accuracy"] * 100 + dy),
                    fontsize=7.5, color=c,
                    arrowprops=dict(arrowstyle="-", color=c, lw=0.5))

    # Dashed line connecting sweep
    xs = [r["metrics"]["causal_effect_proxy"]       for r in summary["unlearning_runs"]]
    ys = [r["metrics"]["intervened_accuracy"] * 100 for r in summary["unlearning_runs"]]
    ax.plot(xs, ys, linestyle="--", color="gray", linewidth=1.0, alpha=0.6, zorder=2)

    ax.set_xlabel("$\\mathrm{CE}_C(\\theta)$  (lower $\\rightarrow$ more causal forgetting)",
                  fontsize=9)
    ax.set_ylabel("Intervened accuracy (%)", fontsize=9)
    ax.set_title("Causal Forgetting vs. Utility Trade-off", fontsize=10)

    # Ideal-direction arrow: upper-right empty region, points toward top-left
    ax.annotate("", xy=(0.55, 78), xytext=(1.85, 59),
                arrowprops=dict(arrowstyle="->", color="purple", lw=1.3))
    ax.text(1.88, 58.0, "Ideal\ndirection", fontsize=7.5, color="purple",
            ha="left", va="top", linespacing=1.3)

    ax.legend(fontsize=8.5, loc="lower left", framealpha=0.9)
    out = FIGURES_DIR / "pareto_tradeoff.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


# ── Dataset helpers ────────────────────────────────────────────────────────
REPO_ROOT  = Path(__file__).parent.parent
DATA_ROOT  = REPO_ROOT / "data"
CKPT_DIR   = REPO_ROOT / "artifacts" / "default" / "checkpoints"

def _load_project():
    """Import project modules (works when run from paper/ directory)."""
    import sys
    sys.path.insert(0, str(REPO_ROOT / "src"))
    from causal_unlearning.datasets import ColoredMNISTDataset
    from causal_unlearning.models   import build_model
    from causal_unlearning.training import load_checkpoint, resolve_device
    return ColoredMNISTDataset, build_model, load_checkpoint, resolve_device


def _load_model(ckpt_name: str, build_model, load_checkpoint, resolve_device):
    path = CKPT_DIR / ckpt_name
    payload = load_checkpoint(path, device="cpu")
    model = build_model(payload["model_name"])
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model


def _colorize(gray: "np.ndarray", color_id: int) -> "np.ndarray":
    """gray: (H,W) float32 [0,1]  → RGB (H,W,3)."""
    rgb = np.zeros((*gray.shape, 3), dtype=np.float32)
    rgb[..., color_id] = gray
    return rgb


# ── Figure: Dataset Examples ───────────────────────────────────────────────
def fig_dataset_examples() -> None:
    """
    Grid showing representative Colored MNIST samples.
    Row 0: observational world (correlated colors) – factual images
    Row 1: same digits – counterfactual images (flipped color)
    Row 2: intervened world (random colors)
    """
    import torch
    ColoredMNISTDataset, *_ = _load_project()

    # Target digit identities to showcase (mix of low and high)
    TARGET_LABELS = [1, 3, 4, 6, 8]
    N = len(TARGET_LABELS)

    # ── load observational dataset ──────────────────────────────────────
    obs_ds = ColoredMNISTDataset(
        root=str(DATA_ROOT), split="test", world="observational",
        size=5000, correlation=0.9, seed=43, download=False)
    # ── load intervened dataset ─────────────────────────────────────────
    int_ds = ColoredMNISTDataset(
        root=str(DATA_ROOT), split="test", world="intervened",
        size=5000, correlation=0.9, seed=43, download=False)

    def find_first(ds, label):
        for i in range(len(ds)):
            item = ds[i]
            if int(item["label"].item()) == label:
                return item
        return None

    obs_items = [find_first(obs_ds, lbl) for lbl in TARGET_LABELS]
    int_items = [find_first(int_ds, lbl) for lbl in TARGET_LABELS]

    fig, axes = plt.subplots(3, N, figsize=(N * 1.4, 4.6))
    row_labels = [
        "Obs. $w$\n(factual)",
        "Obs. $w$\n(counterfact.)",
        "Intervened\n$w^{\\tau}$",
    ]

    for col, (lbl, obs, intv) in enumerate(zip(TARGET_LABELS, obs_items, int_items)):
        imgs = [
            obs["image"].permute(1, 2, 0).numpy(),
            obs["counterfactual"].permute(1, 2, 0).numpy(),
            intv["image"].permute(1, 2, 0).numpy(),
        ]
        border_colors = [
            ("red" if obs["color"].item() == 0 else "green"),
            ("green" if obs["color"].item() == 0 else "red"),
            ("red" if intv["color"].item() == 0 else "green"),
        ]
        for row in range(3):
            ax = axes[row, col]
            ax.imshow(np.clip(imgs[row], 0, 1), interpolation="nearest")
            ax.set_xticks([]); ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_edgecolor(border_colors[row])
                spine.set_linewidth(2.5)
            if row == 0:
                ax.set_title(f"digit {lbl}", fontsize=8)

    fig.suptitle(
        "Border color = actual pixel color (red / green). "
        "Row 3: intervened world — colors are uniformly random.",
        fontsize=7, y=0.01, va="bottom",
    )
    # Tight layout with minimal left margin for row labels
    fig.tight_layout(rect=[0.08, 0.05, 1, 1])
    # Place row labels flush against the image grid using axes bboxes
    for row, label in enumerate(row_labels):
        ax0 = axes[row, 0]
        bbox = ax0.get_position()
        y_center = (bbox.y0 + bbox.y1) / 2
        # x = midpoint of [0, bbox.x0] so labels sit snug against images
        x_label = bbox.x0 / 2
        fig.text(x_label, y_center, label, ha="center", va="center",
                 rotation=90, fontsize=8)
    out = FIGURES_DIR / "dataset_examples.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


# ── Figure: Qualitative Prediction Stability ───────────────────────────────
def fig_qualitative_predictions() -> None:
    """
    For 4 selected examples: show factual + counterfactual images and the
    predicted class probabilities under θ₀, θ⁻ (λ=0.5), θ*.
    Highlights where θ₀ flips its prediction.
    """
    import torch
    import torch.nn.functional as F

    ColoredMNISTDataset, build_model, load_checkpoint, resolve_device = _load_project()

    device = resolve_device("cpu")
    theta0   = _load_model("baseline.pt",          build_model, load_checkpoint, resolve_device)
    theta_u  = _load_model("unlearn_lambda_0p5.pt", build_model, load_checkpoint, resolve_device)
    theta_st = _load_model("oracle.pt",             build_model, load_checkpoint, resolve_device)
    models   = [theta0, theta_u, theta_st]
    mlabels  = ["$\\theta_0$ (baseline)", "$\\theta^-$ (ours, $\\lambda{=}0.5$)", "$\\theta^*$ (oracle)"]
    mcolors  = [PALETTE["baseline"], PALETTE["l05"], PALETTE["oracle"]]

    # Load observational test set
    obs_ds = ColoredMNISTDataset(
        root=str(DATA_ROOT), split="test", world="observational",
        size=5000, correlation=0.9, seed=43, download=False)

    # Find examples where θ₀ *flips* top-1 prediction factual→CF
    flips = []
    for i in range(len(obs_ds)):
        item = obs_ds[i]
        fact = item["image"].unsqueeze(0)
        cf   = item["counterfactual"].unsqueeze(0)
        with torch.no_grad():
            p_f  = theta0(fact).argmax(1).item()
            p_cf = theta0(cf).argmax(1).item()
        if p_f != p_cf:
            flips.append(item)
        if len(flips) == 4:
            break

    if len(flips) < 4:
        # Pad with any items if not enough flips
        for i in range(len(obs_ds)):
            if len(flips) >= 4:
                break
            item = obs_ds[i]
            if item not in flips:
                flips.append(item)

    N_EXAMPLES = len(flips)
    fig = plt.figure(figsize=(7.0, N_EXAMPLES * 1.95))
    gs  = fig.add_gridspec(N_EXAMPLES, 5, wspace=0.35, hspace=0.55,
                           width_ratios=[1, 1, 2.2, 2.2, 2.2])

    digit_names = [str(d) for d in range(10)]

    for row, item in enumerate(flips):
        true_lbl = int(item["label"].item())
        fact_img = item["image"].permute(1, 2, 0).numpy()
        cf_img   = item["counterfactual"].permute(1, 2, 0).numpy()
        fact_t   = item["image"].unsqueeze(0)
        cf_t     = item["counterfactual"].unsqueeze(0)

        # ── factual image ──
        ax_f = fig.add_subplot(gs[row, 0])
        ax_f.imshow(np.clip(fact_img, 0, 1), interpolation="nearest")
        ax_f.set_xticks([]); ax_f.set_yticks([])
        c_border = "red" if item["color"].item() == 0 else "green"
        for sp in ax_f.spines.values():
            sp.set_edgecolor(c_border); sp.set_linewidth(2.5)
        if row == 0:
            ax_f.set_title("Factual", fontsize=8)
        ax_f.set_xlabel(f"true: {true_lbl}", fontsize=7)

        # ── counterfactual image ──
        ax_cf = fig.add_subplot(gs[row, 1])
        ax_cf.imshow(np.clip(cf_img, 0, 1), interpolation="nearest")
        ax_cf.set_xticks([]); ax_cf.set_yticks([])
        cf_border = "green" if item["color"].item() == 0 else "red"
        for sp in ax_cf.spines.values():
            sp.set_edgecolor(cf_border); sp.set_linewidth(2.5); sp.set_linestyle("--")
        if row == 0:
            ax_cf.set_title("Counterfactual", fontsize=8)

        # ── probability bar charts per model ──
        for col, (mdl, mlbl, mcol) in enumerate(zip(models, mlabels, mcolors)):
            ax = fig.add_subplot(gs[row, col + 2])
            with torch.no_grad():
                prob_f  = F.softmax(mdl(fact_t), dim=1).squeeze().numpy()
                prob_cf = F.softmax(mdl(cf_t),   dim=1).squeeze().numpy()

            top_f  = int(np.argmax(prob_f))
            top_cf = int(np.argmax(prob_cf))
            flipped = (top_f != top_cf)

            x = np.arange(10)
            ax.bar(x - 0.2, prob_f,  0.38, color=mcol, alpha=0.85,
                   label="factual")
            ax.bar(x + 0.2, prob_cf, 0.38, color=mcol, alpha=0.38,
                   hatch="//", edgecolor="white", label="CF")

            # Highlight true label
            ax.axvline(true_lbl, color="black", linestyle=":", linewidth=0.8)

            # Red/green frame for flip status
            frame_col = "#D32F2F" if flipped else "#388E3C"
            for sp in ax.spines.values():
                sp.set_edgecolor(frame_col); sp.set_linewidth(1.8)

            ax.set_xlim(-0.6, 9.6)
            ax.set_xticks(x)
            ax.set_xticklabels(digit_names, fontsize=6)
            ax.set_ylim(0, 1.05)
            ax.set_yticks([0, 0.5, 1.0])
            ax.tick_params(axis="y", labelsize=6)
            if row == 0:
                ax.set_title(mlbl, fontsize=7.5)
            if col == 0:
                ax.set_ylabel(f"ex. {row+1}", fontsize=7)

    # Shared legend
    solid = mpatches.Patch(facecolor="gray", alpha=0.85,  label="Factual")
    hatch = mpatches.Patch(facecolor="gray", alpha=0.38, hatch="//",
                           edgecolor="gray",              label="Counterfactual")
    red_f  = mpatches.Patch(facecolor="none", edgecolor="#D32F2F",
                            linewidth=1.5, label="Prediction flips (red frame)")
    grn_f  = mpatches.Patch(facecolor="none", edgecolor="#388E3C",
                            linewidth=1.5, label="Stable prediction (green frame)")
    fig.legend(handles=[solid, hatch, red_f, grn_f],
               loc="lower center", ncol=4, fontsize=7,
               bbox_to_anchor=(0.5, -0.02))
    out = FIGURES_DIR / "qualitative_predictions.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


# ── Figure: Concept Diagram ────────────────────────────────────────────────
def fig_concept_diagram() -> None:
    """
    NeurIPS-quality concept diagram.
    Vertical layout:
      • Panels A & B  (y = 1.72 … 6.00) — same height, side-by-side
      • Clear gap      (y = 1.28 … 1.72) — arrow labels sit here
      • Strip          (y = 0.06 … 1.22) — θ⁻ box + metrics
    """
    from matplotlib.patches import FancyBboxPatch, Circle

    # ── Palette ────────────────────────────────────────────────────────────
    PA_BG = "#F6F6F6";  PA_BD = "#C0C0C0"
    PB_BG = "#F2FAF5";  PB_BD = "#7FC89A"
    ST_BG = "#EDF5FC";  ST_BD = "#80B4D8"
    LB_BG = "#F0FAF3";  LB_BD = "#7FC89A"
    RED = "#B03A2E";  GRN = "#1A7A45"
    BLU = "#1A5F8A";  TXT = "#1C1C1C";  GRY = "#5C5C5C"

    FW, FH = 12.5, 6.2
    fig = plt.figure(figsize=(FW, FH))
    ax  = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, FW);  ax.set_ylim(0, FH)
    ax.axis("off");  fig.patch.set_facecolor("white")

    # ── Helpers ─────────────────────────────────────────────────────────────
    def rbox(x0, y0, w, h, fc, ec, lw=0.85, pad=0.09, zo=1):
        ax.add_patch(FancyBboxPatch(
            (x0, y0), w, h, boxstyle=f"round,pad={pad}",
            facecolor=fc, edgecolor=ec, linewidth=lw, zorder=zo))

    def node(cx, cy, lbl, fc, r=0.28, lw=1.0, ls="-", zo=4):
        ax.add_patch(Circle((cx, cy), r, facecolor=fc, edgecolor=TXT,
                            linewidth=lw, linestyle=ls, zorder=zo))
        ax.text(cx, cy, lbl, ha="center", va="center",
                fontsize=11, fontweight="bold", color=TXT, zorder=zo + 1)

    def mbox(cx, cy, lbl, fc, ec, w=2.55, h=0.48, zo=4):
        rbox(cx - w/2, cy - h/2, w, h, fc, ec, lw=1.15, pad=0.06, zo=zo)
        ax.text(cx, cy, lbl, ha="center", va="center",
                fontsize=9.5, color=TXT, zorder=zo + 1)

    def arr(x0, y0, x1, y1, col, lw=1.15, ls="-", cs="arc3,rad=0.0",
            mut=11, zo=5):
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                    arrowprops=dict(arrowstyle="-|>", color=col, lw=lw,
                                   linestyle=ls, connectionstyle=cs,
                                   mutation_scale=mut), zorder=zo)

    # ── Key y-coordinates ────────────────────────────────────────────────────
    # panels span y = P_BOT … P_TOP ; strip y = S_BOT … S_TOP
    # gap between strip top and panel bottom = P_BOT - S_TOP = 0.50  (label space)
    P_BOT = 1.72;  P_TOP = 6.00   # panel bottom / top
    S_BOT = 0.06;  S_TOP = 1.22   # strip bottom / top
    GAP_Y = (P_BOT + S_TOP) / 2   # = 1.47  — centre of the clear gap

    # panel centres
    CX_A = 2.55;  CX_B = 9.95;  CX_C = 6.25   # A, B, loss-box

    # ── Background panels ────────────────────────────────────────────────────
    rbox(0.12,  P_BOT, 4.86, P_TOP - P_BOT, PA_BG, PA_BD, lw=0.9, pad=0.10, zo=0)
    rbox(7.52,  P_BOT, 4.86, P_TOP - P_BOT, PB_BG, PB_BD, lw=0.9, pad=0.10, zo=0)
    rbox(3.20,  S_BOT, 6.10, S_TOP - S_BOT, ST_BG, ST_BD, lw=0.8, pad=0.08, zo=0)

    # ════════════════════════════════════════════════════════════════════
    # PANEL A — World w  (Observational)
    # ════════════════════════════════════════════════════════════════════
    ax.text(CX_A, 5.80, r"(a)  World $w$ — Observational",
            ha="center", va="center", fontsize=10.5, fontweight="bold",
            color=TXT, zorder=6)

    # SCM nodes
    node(CX_A, 5.05, r"$Y$", "#D0E8F8")
    node(1.12,  3.88, r"$X$", "#E2E2E2")
    node(3.98,  3.88, r"$C$", "#FAE0C8")

    # Y→X  (causal, solid)
    arr(2.30, 4.80, 1.36, 4.14, TXT, lw=1.1)
    ax.text(1.60, 4.52, "shape", ha="center", fontsize=7.5,
            color=GRY, rotation=43, zorder=6)
    # Y→C  (spurious, dashed red)
    arr(2.80, 4.80, 3.74, 4.14, RED, lw=1.1, ls="--")
    ax.text(3.52, 4.52, r"$\rho{=}0.9$", ha="center", fontsize=7.5,
            color=RED, rotation=-43, zorder=6)

    # θ₀ model box
    Y_M0 = 2.90
    mbox(CX_A, Y_M0, r"$\theta_0(x,c)$", "#FDEAEA", RED)
    arr(1.12, 3.60, 1.72, 3.14, TXT, lw=0.95)
    arr(3.98, 3.60, 3.38, 3.14, RED, lw=0.95, ls="--")

    # Baseline metrics — well clear of panel bottom
    ax.text(CX_A, P_BOT + 0.30,
            r"obs: $80.4\%$  $|$  int: $49.1\%$  $|$  $\mathrm{CE}_C{=}2.46$",
            ha="center", va="center", fontsize=8.5, color=TXT, zorder=6)

    # ════════════════════════════════════════════════════════════════════
    # CENTRE  — do() operator + fine-tune loss box
    # ════════════════════════════════════════════════════════════════════
    # do() arrow
    ax.annotate("", xy=(7.32, 5.05), xytext=(5.18, 5.05),
                arrowprops=dict(arrowstyle="-|>", color=BLU, lw=2.3,
                                mutation_scale=16), zorder=6)
    ax.text(CX_C, 5.30, r"$\mathrm{do}(C \sim \mathrm{Bern}(1/2))$",
            ha="center", va="bottom", fontsize=9.0,
            color=BLU, fontweight="bold", zorder=7)

    # Fine-tune loss box: spans from P_BOT+0.28 up to ~4.60
    rbox(4.98, P_BOT + 0.28, 2.54, 2.68, LB_BG, LB_BD, lw=0.9, pad=0.12, zo=3)
    ax.text(CX_C, 4.64, "Fine-tune objective:",
            ha="center", va="center", fontsize=8.5,
            fontweight="bold", color=GRN, zorder=6)
    ax.text(CX_C, 4.24, r"$-\!\log p_\theta(y|x)$ [retain]",
            ha="center", va="center", fontsize=8.0, color=TXT, zorder=6)
    ax.text(CX_C, 3.84, r"$+\;\lambda\;\mathrm{CE}_C(\theta)$ [forget]",
            ha="center", va="center", fontsize=8.0, color=TXT, zorder=6)
    ax.text(CX_C, 3.44, r"$+\;\beta\;\|\theta{-}\theta_0\|^2$ [locality]",
            ha="center", va="center", fontsize=8.0, color=TXT, zorder=6)
    ax.text(CX_C, 2.62,
            r"$\lambda{=}0.5$,  $\beta{=}0.001$,  2 epochs",
            ha="center", va="center", fontsize=7.5,
            color=GRY, style="italic", zorder=6)

    # ════════════════════════════════════════════════════════════════════
    # PANEL B — World w^τ  (Intervened)
    # ════════════════════════════════════════════════════════════════════
    ax.text(CX_B, 5.80, r"(b)  World $w^{\tau}$ — Intervened",
            ha="center", va="center", fontsize=10.5, fontweight="bold",
            color=GRN, zorder=6)

    node(CX_B, 5.05, r"$Y$", "#D0E8F8")
    node(8.52,  3.88, r"$X$", "#E2E2E2")
    node(11.38, 3.88, r"$C$", "#EAF5FB", lw=1.2, ls="dashed")

    # Y→X only
    arr(9.70, 4.80, 8.76, 4.14, TXT, lw=1.1)
    ax.text(8.98, 4.52, "shape", ha="center", fontsize=7.5,
            color=GRY, rotation=43, zorder=6)

    # C⊥Y badge
    ax.text(11.38, 4.38, r"$C \perp Y$", ha="center", va="bottom",
            fontsize=8.0, color=BLU, zorder=6)
    arr(11.38, 4.36, 11.38, 4.20, BLU, lw=0.9, mut=8)

    # θ* model box
    Y_MS = 2.90
    mbox(CX_B, Y_MS, r"$\theta^*(x,c)$", "#E4FAF0", GRN)
    arr(8.52, 3.60, 9.12, 3.14, TXT, lw=0.95)
    arr(11.38, 3.60, 10.78, 3.14, TXT, lw=0.95)

    # Oracle metrics
    ax.text(CX_B, P_BOT + 0.30,
            r"obs: $86.2\%$  $|$  int: $86.0\%$  $|$  $\mathrm{CE}_C{=}0.006$",
            ha="center", va="center", fontsize=8.5, color=TXT, zorder=6)

    # ════════════════════════════════════════════════════════════════════
    # BOTTOM STRIP — Unlearned model θ⁻
    # ════════════════════════════════════════════════════════════════════
    # italic descriptor at top of strip
    ax.text(CX_C, S_TOP - 0.10,
            r"Post-hoc unlearned  $\theta^-(x,c)$"
            r"  ($\lambda{=}0.5$,  $\beta{=}0.001$,  2 epochs)",
            ha="center", va="top", fontsize=8.0,
            color=BLU, style="italic", zorder=6)
    # θ⁻ model box
    Y_MU = (S_BOT + S_TOP) / 2 + 0.05
    mbox(CX_C, Y_MU, r"$\theta^-(x,c)$", "#D4E9FA", BLU)
    # metrics below box
    ax.text(CX_C, S_BOT + 0.18,
            r"obs: $85.0\%$  $|$  int: $63.4\%$  $|$  "
            r"$\mathrm{CE}_C{=}0.38$  ($\downarrow 84.4\%$)  $|$  "
            r"$\mathcal{D}_\mathrm{fid}{=}1.57$",
            ha="center", va="bottom", fontsize=8.5, color=TXT, zorder=6)

    # ════════════════════════════════════════════════════════════════════
    # CONNECTING ARROWS  (routed through the clear gap)
    # ════════════════════════════════════════════════════════════════════
    # θ₀ → θ⁻  (fine-tune, green) — exits panel bottom, enters strip top-left
    arr(CX_A, Y_M0 - 0.24, 4.40, S_TOP + 0.04,
        GRN, lw=1.4, cs="arc3,rad=0.18", zo=6)
    # label sits cleanly in the gap
    ax.text(2.95, GAP_Y,
            r"fine-tune $\mathcal{L}(\theta)$",
            ha="center", va="center", fontsize=8.0, color=GRY, zorder=7)

    # θ⁻ → θ*  (approximate oracle, dashed blue) — exits strip top-right, enters panel
    arr(8.10, S_TOP + 0.04, CX_B, Y_MS - 0.24,
        BLU, lw=1.4, ls="dashed", cs="arc3,rad=-0.18", zo=6)
    ax.text(9.55, GAP_Y,
            r"$\approx$ oracle",
            ha="center", va="center", fontsize=8.0, color=BLU, zorder=7)

    out = FIGURES_DIR / "concept_diagram.pdf"
    fig.savefig(out, bbox_inches="tight", dpi=200)
    plt.close(fig)
    print(f"Saved {out}")


# ── Main ───────────────────────────────────────────────────────────────────
# ── New Figure: Baselines Comparison ─────────────────────────────────────────
def fig_baselines_comparison(summary: dict) -> None:
    """Bar chart comparing our method (λ=0.5) against GRL and intervened-FT baselines."""
    import json as _json
    baseline_dir = FIGURES_DIR.parent.parent / "artifacts" / "baselines"
    try:
        grl_data  = _json.loads((baseline_dir / "grl.json").read_text())
        ift_data  = _json.loads((baseline_dir / "intervened_ft.json").read_text())
    except FileNotFoundError:
        print("Baselines data not found — skipping fig_baselines_comparison")
        return

    # Gather metrics
    our_run = next(r for r in summary["unlearning_runs"] if abs(r["lambda_ce"] - 0.5) < 0.01)
    entries = [
        ("Baseline $\\theta_0$",       summary["baseline"]["metrics"],  "#E05C5C", None),
        ("Intervened FT",              ift_data["metrics"],              "#FF8F00", "stripes"),
        ("Adv.\ GRL",                  grl_data["metrics"],              "#9C27B0", "stripes"),
        ("Ours ($\\lambda{=}0.5$)",    our_run["metrics"],               "#1976D2", None),
        ("Oracle $\\theta^*$",         summary["oracle"]["metrics"],     "#4CAF50", None),
    ]

    names   = [e[0] for e in entries]
    obs_acc = [e[1]["observational_accuracy"] * 100 for e in entries]
    int_acc = [e[1]["intervened_accuracy"] * 100     for e in entries]
    ce_vals = [e[1]["causal_effect_proxy"]            for e in entries]
    colors  = [e[2] for e in entries]

    x = np.arange(len(names))
    w = 0.28

    fig, axes = plt.subplots(1, 3, figsize=(9.5, 3.2))
    fig.subplots_adjust(wspace=0.38)

    for ax_idx, (vals, ylabel, title, ylim) in enumerate([
        (obs_acc, "Accuracy (%)",            "(a) Observational Accuracy", (60, 95)),
        (int_acc, "Accuracy (%)",            "(b) Intervened Accuracy",    (30, 95)),
        (ce_vals, "$\\mathrm{CE}_C(\\theta)$ (lower $=$ better)", "(c) Causal Effect Proxy", None),
    ]):
        ax = axes[ax_idx]
        bars = ax.bar(x, vals, 0.55, color=colors, alpha=0.82, edgecolor="white", linewidth=0.5)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + (0.5 if ax_idx < 2 else 0.02),
                    f"{v:.1f}" if ax_idx < 2 else f"{v:.2f}",
                    ha="center", va="bottom", fontsize=7)
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=20, ha="right", fontsize=8)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.set_title(title, fontsize=9)
        if ylim:
            ax.set_ylim(*ylim)

    out = FIGURES_DIR / "baselines_comparison.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


# ── New Figure: Extended Lambda Sweep ─────────────────────────────────────────
def fig_lambda_sweep_extended() -> None:
    """Lambda sweep with 6 values including λ=0.2 and λ=2.0."""
    import json as _json
    path = FIGURES_DIR.parent.parent / "artifacts" / "ablation_lambda" / "summary.json"
    try:
        s = _json.loads(path.read_text())
    except FileNotFoundError:
        print("Extended lambda sweep data not found — skipping")
        return

    runs   = s["unlearning_runs"]
    lambdas = [r["lambda_ce"]                              for r in runs]
    ce      = [r["metrics"]["causal_effect_proxy"]         for r in runs]
    int_acc = [r["metrics"]["intervened_accuracy"] * 100   for r in runs]
    obs_acc = [r["metrics"]["observational_accuracy"] * 100 for r in runs]
    fid     = [r["metrics"]["fidelity_to_oracle_kl"]       for r in runs]

    base_ce  = s["baseline"]["metrics"]["causal_effect_proxy"]
    oracle_ce = s["oracle"]["metrics"]["causal_effect_proxy"]
    base_int = s["baseline"]["metrics"]["intervened_accuracy"] * 100
    oracle_int = s["oracle"]["metrics"]["intervened_accuracy"] * 100
    base_obs = s["baseline"]["metrics"]["observational_accuracy"] * 100
    oracle_obs = s["oracle"]["metrics"]["observational_accuracy"] * 100

    fig, axes = plt.subplots(1, 4, figsize=(10.0, 2.8))
    color_line = "#1565C0"

    for ax, vals, ylabel, title, ref_base, ref_oracle, lower_better in [
        (axes[0], ce,      "$\\mathrm{CE}_C$",            "(a) CE Proxy $\\downarrow$",     base_ce,  oracle_ce,  True),
        (axes[1], int_acc, "Int. accuracy (%)",           "(b) Intervened Acc. $\\uparrow$",base_int, oracle_int, False),
        (axes[2], obs_acc, "Obs. accuracy (%)",           "(c) Observational Acc. $\\uparrow$", base_obs, oracle_obs, False),
        (axes[3], fid,     "$\\mathcal{D}_{\\mathrm{fid}}$", "(d) Fidelity to Oracle $\\downarrow$", None, None, True),
    ]:
        ax.plot(lambdas, vals, marker="o", color=color_line, linewidth=1.8, markersize=5, zorder=3)
        if ref_base is not None:
            ax.axhline(ref_base,   color="#E05C5C", linestyle="--", linewidth=0.9, label="Baseline")
        if ref_oracle is not None:
            ax.axhline(ref_oracle, color="#4CAF50", linestyle="--", linewidth=0.9, label="Oracle")
        ax.set_xlabel("$\\lambda$", fontsize=9)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.set_title(title, fontsize=9)
        ax.set_xticks(lambdas)
        ax.tick_params(axis="x", labelsize=7.5)
        if ref_base is not None:
            ax.legend(fontsize=7, loc="upper right" if lower_better else "lower right")

    fig.tight_layout()
    out = FIGURES_DIR / "lambda_sweep_extended.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


# ── New Figure: Epochs Ablation ────────────────────────────────────────────────
def fig_epochs_ablation() -> None:
    """Show how metrics improve with more unlearning epochs at λ=0.5."""
    import json as _json
    path = FIGURES_DIR.parent.parent / "artifacts" / "ablation_epochs" / "summary.json"
    try:
        data = _json.loads(path.read_text())
    except FileNotFoundError:
        print("Epochs ablation data not found — skipping")
        return

    runs = data["runs"]
    epochs  = [r["epochs"]                              for r in runs]
    int_acc = [r["metrics"]["intervened_accuracy"] * 100 for r in runs]
    ce      = [r["metrics"]["causal_effect_proxy"]       for r in runs]
    fid     = [r["metrics"]["fidelity_to_oracle_kl"]     for r in runs]

    # Oracle references
    oracle_path = FIGURES_DIR.parent.parent / "artifacts" / "default" / "summary.json"
    with open(oracle_path) as f:
        default_s = _json.load(f)
    oracle_int = default_s["oracle"]["metrics"]["intervened_accuracy"] * 100

    fig, axes = plt.subplots(1, 3, figsize=(8.0, 2.8))
    c = "#1565C0"

    for ax, vals, ylabel, title in [
        (axes[0], int_acc, "Intervened accuracy (%)", "(a) Int. Accuracy $\\uparrow$"),
        (axes[1], ce,      "$\\mathrm{CE}_C$",       "(b) CE Proxy $\\downarrow$"),
        (axes[2], fid,     "$\\mathcal{D}_{\\mathrm{fid}}$", "(c) Fidelity $\\downarrow$"),
    ]:
        ax.plot(epochs, vals, marker="o", color=c, linewidth=1.8, markersize=6)
        if "Int." in title:
            ax.axhline(oracle_int, color="#4CAF50", linestyle="--", linewidth=0.9, label="Oracle")
            ax.legend(fontsize=8)
        ax.set_xlabel("Unlearning epochs", fontsize=9)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.set_title(title, fontsize=9)
        ax.set_xticks(epochs)
        for i, (e, v) in enumerate(zip(epochs, vals)):
            ax.annotate(f"{v:.1f}" if "%" in ylabel else f"{v:.2f}",
                        xy=(e, v), xytext=(0, 5), textcoords="offset points",
                        ha="center", fontsize=7)

    fig.tight_layout()
    out = FIGURES_DIR / "epochs_ablation.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


# ── New Figure: Rho Sweep ──────────────────────────────────────────────────────
def fig_rho_sweep() -> None:
    """Show CE proxy and int-acc vs spurious correlation ρ for ours (λ=0.5) vs baseline."""
    import json as _json
    summary_path = FIGURES_DIR.parent.parent / "artifacts" / "ablation_rho" / "summary.json"
    try:
        data = _json.loads(summary_path.read_text())
    except FileNotFoundError:
        print("Rho sweep data not found — skipping")
        return

    rhos = sorted(float(r) for r in data["runs"].keys())
    base_ce, base_int, ours_ce, ours_int = [], [], [], []

    for rho in rhos:
        s = data["runs"][str(rho)]
        base_ce.append(s["baseline"]["metrics"]["causal_effect_proxy"])
        base_int.append(s["baseline"]["metrics"]["intervened_accuracy"] * 100)
        our_run = next(r for r in s["unlearning_runs"] if abs(r["lambda_ce"] - 0.5) < 0.01)
        ours_ce.append(our_run["metrics"]["causal_effect_proxy"])
        ours_int.append(our_run["metrics"]["intervened_accuracy"] * 100)

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.8))

    for ax, base_vals, ours_vals, ylabel, title in [
        (axes[0], base_ce, ours_ce, "$\\mathrm{CE}_C$ (lower $=$ better)", "(a) Causal Effect Proxy"),
        (axes[1], base_int, ours_int, "Intervened accuracy (%)", "(b) Intervened Accuracy"),
    ]:
        ax.plot(rhos, base_vals, marker="o", color="#E05C5C", linewidth=1.8, markersize=6,
                label="Baseline $\\theta_0$")
        ax.plot(rhos, ours_vals, marker="s", color="#1976D2", linewidth=1.8, markersize=6,
                label="Ours ($\\lambda{=}0.5$)")
        ax.set_xlabel("Spurious correlation $\\rho$", fontsize=9)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.set_title(title, fontsize=9)
        ax.set_xticks(rhos)
        ax.legend(fontsize=8)

    fig.tight_layout()
    out = FIGURES_DIR / "rho_sweep.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


def fig_representation_probing() -> None:
    """Bar chart comparing color/label probe accuracy across models."""
    import json
    probe_path = ROOT / "artifacts" / "analysis" / "representation_probing.json"
    with open(probe_path) as f:
        data = json.load(f)

    models      = ["Baseline", "Unlearned\n$\\lambda{=}0.5$", "Oracle"]
    color_probe = [data["baseline"]["color_probe_acc_int"] * 100,
                   data["unlearned_l05"]["color_probe_acc_int"] * 100,
                   data["oracle"]["color_probe_acc_int"] * 100]
    label_probe = [data["baseline"]["label_probe_acc_int"] * 100,
                   data["unlearned_l05"]["label_probe_acc_int"] * 100,
                   data["oracle"]["label_probe_acc_int"] * 100]

    x = np.arange(len(models))
    width = 0.38
    fig, ax = plt.subplots(figsize=(5.5, 3.8))
    b1 = ax.bar(x - width/2, color_probe, width, label="Color probe acc",
                color=PALETTE["baseline"], alpha=0.85, edgecolor="white", linewidth=0.6)
    b2 = ax.bar(x + width/2, label_probe, width, label="Label probe acc",
                color=PALETTE["oracle"], alpha=0.85, edgecolor="white", linewidth=0.6)
    ax.set_xticks(x); ax.set_xticklabels(models, fontsize=9)
    ax.set_ylabel("Probe Accuracy (%)", fontsize=9)
    ax.set_ylim(60, 105)
    ax.axhline(100, color="gray", lw=0.8, ls="--", alpha=0.5)
    ax.legend(fontsize=8, loc="lower right")
    ax.set_title("Linear Probe Accuracy (intervened test set, seed 42)", fontsize=9)
    for bars in [b1, b2]:
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, h + 0.4,
                    f"{h:.1f}", ha="center", va="bottom", fontsize=7.5)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "representation_probing.pdf", bbox_inches="tight")
    plt.close(fig)
    print("  representation_probing.pdf")


def fig_data_efficiency() -> None:
    """Line chart of CE proxy and intervened accuracy vs fraction of CF pairs."""
    import json
    de_path = ROOT / "artifacts" / "analysis" / "data_efficiency.json"
    with open(de_path) as f:
        data = json.load(f)

    fracs  = [r["frac"] * 100 for r in data["runs"]]
    ce     = [r["metrics"]["causal_effect_proxy"] for r in data["runs"]]
    int_a  = [r["metrics"]["intervened_accuracy"] * 100 for r in data["runs"]]
    base_ce = 2.464   # seed-42 baseline CE proxy

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.2, 3.2))

    ax1.plot(fracs, ce, "o-", color=PALETTE["l05"], lw=1.8, ms=5)
    ax1.axhline(base_ce, color=PALETTE["baseline"], lw=1.2, ls="--", label="Baseline")
    ax1.set_xlabel("Counterfactual pairs (%)", fontsize=9)
    ax1.set_ylabel("CE proxy $\\downarrow$", fontsize=9)
    ax1.set_title("CE Proxy vs.\ Data Fraction", fontsize=9)
    ax1.legend(fontsize=8); ax1.set_xscale("log")
    ax1.set_xticks(fracs); ax1.set_xticklabels([f"{int(f)}\\%" for f in fracs], fontsize=8)

    ax2.plot(fracs, int_a, "o-", color=PALETTE["l05"], lw=1.8, ms=5)
    ax2.axhline(54.8, color=PALETTE["baseline"],   lw=1.2, ls="--", label="Baseline")
    ax2.axhline(88.8, color=PALETTE["oracle"], lw=1.2, ls="--", label="Oracle")
    ax2.set_xlabel("Counterfactual pairs (%)", fontsize=9)
    ax2.set_ylabel("Intervened accuracy (%) $\\uparrow$", fontsize=9)
    ax2.set_title("Intervened Acc vs.\ Data Fraction", fontsize=9)
    ax2.legend(fontsize=8); ax2.set_xscale("log")
    ax2.set_xticks(fracs); ax2.set_xticklabels([f"{int(f)}\\%" for f in fracs], fontsize=8)

    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "data_efficiency.pdf", bbox_inches="tight")
    plt.close(fig)
    print("  data_efficiency.pdf")


def fig_noisy_counterfactuals() -> None:
    """Line chart of CE proxy vs. counterfactual noise level."""
    import json
    nc_path = ROOT / "artifacts" / "analysis" / "noisy_counterfactuals.json"
    with open(nc_path) as f:
        data = json.load(f)

    sigmas = [r["sigma"] for r in data["runs"]]
    ce     = [r["metrics"]["causal_effect_proxy"] for r in data["runs"]]
    int_a  = [r["metrics"]["intervened_accuracy"] * 100 for r in data["runs"]]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.2, 3.2))

    ax1.plot(sigmas, ce, "o-", color=PALETTE["l05"], lw=1.8, ms=5)
    ax1.axhline(2.464, color=PALETTE["baseline"],   lw=1.2, ls="--", label="Baseline")
    ax1.axhline(0.006, color=PALETTE["oracle"], lw=1.2, ls="--", label="Oracle")
    ax1.set_xlabel("Noise $\\sigma$", fontsize=9)
    ax1.set_ylabel("CE proxy $\\downarrow$", fontsize=9)
    ax1.set_title("Robustness to Noisy Counterfactuals", fontsize=9)
    ax1.legend(fontsize=8)

    ax2.plot(sigmas, int_a, "o-", color=PALETTE["l05"], lw=1.8, ms=5)
    ax2.axhline(54.8, color=PALETTE["baseline"],   lw=1.2, ls="--", label="Baseline")
    ax2.axhline(88.8, color=PALETTE["oracle"], lw=1.2, ls="--", label="Oracle")
    ax2.set_xlabel("Noise $\\sigma$", fontsize=9)
    ax2.set_ylabel("Intervened accuracy (%) $\\uparrow$", fontsize=9)
    ax2.set_title("Intervened Acc vs.\ CF Noise", fontsize=9)
    ax2.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "noisy_counterfactuals.pdf", bbox_inches="tight")
    plt.close(fig)
    print("  noisy_counterfactuals.pdf")


if __name__ == "__main__":
    summary = load_summary()
    fig_model_comparison(summary)
    fig_lambda_sweep(summary)
    fig_training_curves(summary)
    fig_pareto(summary)
    fig_dataset_examples()
    fig_qualitative_predictions()
    fig_concept_diagram()
    # New extended figures
    fig_baselines_comparison(summary)
    fig_lambda_sweep_extended()
    fig_epochs_ablation()
    fig_rho_sweep()
    # Analysis figures
    fig_representation_probing()
    fig_data_efficiency()
    fig_noisy_counterfactuals()
    print("All figures saved to", FIGURES_DIR)
