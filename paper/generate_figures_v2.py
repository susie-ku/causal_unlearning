"""
Generate v2 figures for NeurIPS paper using multi-seed analysis data.

New figures:
  - data_efficiency_v2.pdf       (multi-seed with std bands)
  - noisy_counterfactuals_v2.pdf (multi-seed with std bands)
  - distance_to_oracle.pdf       (CE vs D_fid scatter across seeds × λ)
  - waterbirds_main.pdf          (3-panel bar chart: avg, wg, CE)
  - utility_forgetting_pareto.pdf(CMNIST + Waterbirds Pareto frontier)
  - representation_probing_v2.pdf(multi-seed bar chart)
"""
from __future__ import annotations
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
FIG_DIR = Path(__file__).parent / "figures"
FIG_DIR.mkdir(exist_ok=True)
ANALYSIS = ROOT / "artifacts" / "analysis_multiseed"
WB       = ROOT / "artifacts" / "waterbirds"

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.titlesize": 10,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 8,
    "figure.dpi": 300,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linestyle": "--",
})

C_BASE   = "#E05C5C"
C_ORACLE = "#4CAF50"
C_OURS   = "#1976D2"
C_NEUTRAL = "#9E9E9E"


def _load(path):
    with open(path) as f:
        return json.load(f)


# ─────────────────────────────────────────────────────────────────────────────
# 1. Data efficiency (multi-seed with std bands)
# ─────────────────────────────────────────────────────────────────────────────
def fig_data_efficiency_v2():
    summary = _load(ANALYSIS / "data_efficiency_summary.json")
    fracs   = np.array([r["frac"] * 100 for r in summary])
    ce_m    = np.array([r["ce_mean"] for r in summary])
    ce_s    = np.array([r["ce_std"] for r in summary])
    int_m   = np.array([r["int_mean"] * 100 for r in summary])
    int_s   = np.array([r["int_std"] * 100 for r in summary])

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.4, 3.0))
    a1.fill_between(fracs, ce_m - ce_s, ce_m + ce_s, alpha=0.2, color=C_OURS)
    a1.plot(fracs, ce_m, "o-", color=C_OURS, lw=1.8, ms=5, label="Ours $(\\lambda=0.5)$")
    a1.axhline(2.00, color=C_BASE, lw=1.2, ls="--", label="Baseline")
    a1.axhline(0.006, color=C_ORACLE, lw=1.2, ls="--", label="Oracle")
    a1.set_xlabel("Counterfactual pairs (%)")
    a1.set_ylabel("CE proxy $\\downarrow$")
    a1.set_xscale("log")
    a1.set_title("CE Proxy vs.\\ Data Fraction (3 seeds)")
    a1.legend(loc="best", fontsize=7.5)
    a1.set_xticks(fracs); a1.set_xticklabels([f"{int(f)}%" for f in fracs])

    a2.fill_between(fracs, int_m - int_s, int_m + int_s, alpha=0.2, color=C_OURS)
    a2.plot(fracs, int_m, "o-", color=C_OURS, lw=1.8, ms=5, label="Ours $(\\lambda=0.5)$")
    a2.axhline(54.8, color=C_BASE, lw=1.2, ls="--", label="Baseline")
    a2.axhline(88.8, color=C_ORACLE, lw=1.2, ls="--", label="Oracle")
    a2.set_xlabel("Counterfactual pairs (%)")
    a2.set_ylabel("Intervened acc (%) $\\uparrow$")
    a2.set_xscale("log")
    a2.set_title("Intervened Accuracy vs.\\ Data Fraction")
    a2.legend(loc="best", fontsize=7.5)
    a2.set_xticks(fracs); a2.set_xticklabels([f"{int(f)}%" for f in fracs])

    fig.tight_layout()
    fig.savefig(FIG_DIR / "data_efficiency_v2.pdf", bbox_inches="tight")
    plt.close(fig)
    print("  data_efficiency_v2.pdf")


# ─────────────────────────────────────────────────────────────────────────────
# 2. Noisy CFs (multi-seed with std bands)
# ─────────────────────────────────────────────────────────────────────────────
def fig_noisy_cfs_v2():
    summary = _load(ANALYSIS / "noisy_cfs_summary.json")
    sigmas = np.array([r["sigma"] for r in summary])
    ce_m   = np.array([r["ce_mean"] for r in summary])
    ce_s   = np.array([r["ce_std"] for r in summary])
    int_m  = np.array([r["int_mean"] * 100 for r in summary])
    int_s  = np.array([r["int_std"] * 100 for r in summary])

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.4, 3.0))
    a1.fill_between(sigmas, ce_m - ce_s, ce_m + ce_s, alpha=0.2, color=C_OURS)
    a1.plot(sigmas, ce_m, "o-", color=C_OURS, lw=1.8, ms=5, label="Ours $(\\lambda=0.5)$")
    a1.axhline(2.00, color=C_BASE, lw=1.2, ls="--", label="Baseline")
    a1.axhline(0.006, color=C_ORACLE, lw=1.2, ls="--", label="Oracle")
    a1.set_xlabel("Counterfactual noise $\\sigma$")
    a1.set_ylabel("CE proxy $\\downarrow$")
    a1.set_title("Robustness to CF Noise (3 seeds)")
    a1.legend(loc="best", fontsize=7.5)

    a2.fill_between(sigmas, int_m - int_s, int_m + int_s, alpha=0.2, color=C_OURS)
    a2.plot(sigmas, int_m, "o-", color=C_OURS, lw=1.8, ms=5, label="Ours $(\\lambda=0.5)$")
    a2.axhline(54.8, color=C_BASE, lw=1.2, ls="--", label="Baseline")
    a2.axhline(88.8, color=C_ORACLE, lw=1.2, ls="--", label="Oracle")
    a2.set_xlabel("Counterfactual noise $\\sigma$")
    a2.set_ylabel("Intervened acc (%) $\\uparrow$")
    a2.set_title("Intervened Acc vs.\\ CF Noise")
    a2.legend(loc="best", fontsize=7.5)

    fig.tight_layout()
    fig.savefig(FIG_DIR / "noisy_counterfactuals_v2.pdf", bbox_inches="tight")
    plt.close(fig)
    print("  noisy_counterfactuals_v2.pdf")


# ─────────────────────────────────────────────────────────────────────────────
# 3. Distance to oracle scatter
# ─────────────────────────────────────────────────────────────────────────────
def fig_distance_to_oracle():
    pts = _load(ANALYSIS / "distance_to_oracle.json")["points"]
    style = {
        "baseline":  {"color": C_BASE,    "marker": "X", "label": "Baseline"},
        "ours_l00":  {"color": "#9E9E9E", "marker": "o", "label": "Ours $\\lambda=0$"},
        "ours_l01":  {"color": "#64B5F6", "marker": "o", "label": "Ours $\\lambda=0.1$"},
        "ours_l05":  {"color": "#1976D2", "marker": "o", "label": "Ours $\\lambda=0.5$"},
        "ours_l10":  {"color": "#0D47A1", "marker": "o", "label": "Ours $\\lambda=1.0$"},
    }
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    for tag, sty in style.items():
        xs = [p["ce"] for p in pts if p["tag"] == tag]
        ys = [p["dfid"] for p in pts if p["tag"] == tag]
        ax.scatter(xs, ys, s=55, color=sty["color"], marker=sty["marker"],
                    label=sty["label"], edgecolor="white", linewidth=0.6, alpha=0.85)
    ax.set_xlabel("CE proxy (lower $=$ more invariant)")
    ax.set_ylabel("$\\mathcal{D}_{\\mathrm{fid}}$ to oracle (lower $=$ closer)")
    ax.set_title("Distance to Oracle vs.\\ CE (3 seeds, CMNIST)")
    ax.legend(loc="upper right", fontsize=7.5, ncol=1)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "distance_to_oracle.pdf", bbox_inches="tight")
    plt.close(fig)
    print("  distance_to_oracle.pdf")


# ─────────────────────────────────────────────────────────────────────────────
# 4. Waterbirds main 3-panel chart
# ─────────────────────────────────────────────────────────────────────────────
def fig_waterbirds_main():
    agg = _load(WB / "aggregate.json")
    methods_show = [
        "baseline", "concept_erasure", "grl", "intervened_ft", "distillation",
        "ours_l0.1", "ours_l0.5", "ours_l1.0", "oracle"
    ]

    labels = {
        "baseline": "Baseline",
        "concept_erasure": "C_Erasure",
        "grl": "GRL$^g$",
        "intervened_ft": "IFT$^g$",
        # "dfr": "DFR$^g$",
        "distillation": "Distill.\n$^\\dagger$",
        "ours_l0.1": "Ours\n$\\lambda{=}0.1$",
        "ours_l0.5": "Ours\n$\\lambda{=}0.5$",
        "ours_l1.0": "Ours\n$\\lambda{=}1$",
        "oracle": "OR.",
    }

    colors = {
        "baseline": C_BASE,
        "concept_erasure": C_NEUTRAL,
        "grl": C_NEUTRAL,
        "intervened_ft": C_NEUTRAL,
        # "dfr": "#FF7043",    
        "distillation": "#FFB300",
        "ours_l0.1": "#64B5F6",
        "ours_l0.5": "#1976D2",
        "ours_l1.0": "#0D47A1",
        "oracle": C_ORACLE,
    }

    # Slightly wider and taller
    fig, axes = plt.subplots(1, 3, figsize=(12.6, 4.2))

    def _style_xticks(ax, labels_list):
        ax.set_xticks(np.arange(len(labels_list)))
        ax.set_xticklabels(labels_list, fontsize=7, rotation=20, ha="right", rotation_mode="anchor")
        ax.tick_params(axis="x", pad=4)
        ax.margins(x=0.05)

    def _bar(ax, key, title, ylabel, ylim=None, text_offset=0.005):
        xs = np.arange(len(methods_show))
        means = [agg[m].get(f"{key}_mean", 0) for m in methods_show]
        stds = [agg[m].get(f"{key}_std", 0) for m in methods_show]
        cs = [colors[m] for m in methods_show]

        bars = ax.bar(
            xs, means, yerr=stds, capsize=3,
            color=cs, edgecolor="white", linewidth=0.6, alpha=0.85
        )

        _style_xticks(ax, [labels[m] for m in methods_show])
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        if ylim is not None:
            ax.set_ylim(*ylim)

        yr = ax.get_ylim()[1] - ax.get_ylim()[0]
        for b, m, s in zip(bars, means, stds):
            ax.text(
                b.get_x() + b.get_width() / 2,
                b.get_height() + s + text_offset * yr,
                f"{m:.2f}",
                ha="center", va="bottom", fontsize=6.5
            )

    _bar(axes[0], "avg_acc", "Average Accuracy", "avg-acc", (0.6, 0.95), text_offset=0.01)
    _bar(axes[1], "wg_acc", "Worst-Group Accuracy", "wg-acc", (0.3, 0.85), text_offset=0.01)

    # CE only available for some methods
    methods_ce = [m for m in methods_show if "ce_mean" in agg.get(m, {})]
    xs = np.arange(len(methods_ce))
    means = [agg[m]["ce_mean"] for m in methods_ce]
    stds = [agg[m]["ce_std"] for m in methods_ce]
    cs = [colors[m] for m in methods_ce]

    bars = axes[2].bar(
        xs, means, yerr=stds, capsize=3,
        color=cs, edgecolor="white", linewidth=0.6, alpha=0.85
    )

    _style_xticks(axes[2], [labels[m] for m in methods_ce])
    axes[2].set_ylabel("$\\mathrm{CE}_C$ $\\downarrow$")
    axes[2].set_title("Causal Effect Proxy")
    axes[2].margins(x=0.08)

    yr = axes[2].get_ylim()[1] - axes[2].get_ylim()[0]
    for b, m, s in zip(bars, means, stds):
        axes[2].text(
            b.get_x() + b.get_width() / 2,
            b.get_height() + s + 0.02 * yr,
            f"{m:.2f}",
            ha="center", va="bottom", fontsize=6.5
        )

    fig.suptitle(
        "Waterbirds: 3-seed multi-method comparison (mean $\\pm$ std)",
        y=0.98, fontsize=10
    )

    # Manually reserve room for multi-line x labels
    fig.subplots_adjust(left=0.06, right=0.995, top=0.78, bottom=0.24, wspace=0.25)

    fig.savefig(FIG_DIR / "waterbirds_main.pdf", bbox_inches="tight")
    plt.close(fig)
    print("  waterbirds_main.pdf")


# ─────────────────────────────────────────────────────────────────────────────
# 5. Representation probing (multi-seed)
# ─────────────────────────────────────────────────────────────────────────────
def fig_probing_v2():
    s = _load(ANALYSIS / "probing_summary.json")
    models = ["baseline", "unlearn_l05", "oracle"]
    labels = ["Baseline", "Unlearned\n$\\lambda{=}0.5$", "Oracle"]
    color_m  = [s[m]["color_probe_int_mean"]*100 for m in models]
    color_se = [s[m]["color_probe_int_std"]*100 for m in models]
    label_m  = [s[m]["label_probe_int_mean"]*100 for m in models]
    label_se = [s[m]["label_probe_int_std"]*100 for m in models]

    x = np.arange(len(models)); w = 0.36
    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    ax.bar(x - w/2, color_m, w, yerr=color_se, capsize=3,
            color=C_BASE, alpha=0.85, edgecolor="white", linewidth=0.6,
            label="Color probe acc")
    ax.bar(x + w/2, label_m, w, yerr=label_se, capsize=3,
            color=C_ORACLE, alpha=0.85, edgecolor="white", linewidth=0.6,
            label="Label probe acc")
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylabel("Probe Accuracy (%)")
    ax.set_ylim(40, 110)
    ax.axhline(100, color="gray", lw=0.8, ls="--", alpha=0.4)
    ax.legend(loc="lower right", fontsize=8)
    ax.set_title("Linear Probe Accuracy on Intervened Data (3 seeds, mean $\\pm$ std)")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "representation_probing_v2.pdf", bbox_inches="tight")
    plt.close(fig)
    print("  representation_probing_v2.pdf")


# ─────────────────────────────────────────────────────────────────────────────
# 6. Pareto: utility vs forgetting (joint CMNIST + Waterbirds)
# ─────────────────────────────────────────────────────────────────────────────
def fig_pareto_v2():
    """CMNIST: int_acc vs CE for baseline, ours sweep, oracle.  Adds 95% bootstrap
    CI ellipses (drawn as cross-hairs) per (method, lambda) computed from per-seed points."""
    from collections import defaultdict
    pts = _load(ANALYSIS / "distance_to_oracle.json")["points"]
    fig, ax = plt.subplots(figsize=(5.4, 3.6))
    style = {
        "baseline":  ("Baseline",        C_BASE,    "X", 75),
        "ours_l00":  ("Ours $\\lambda=0$",   "#9E9E9E", "o", 55),
        "ours_l01":  ("Ours $\\lambda=0.1$", "#64B5F6", "o", 55),
        "ours_l05":  ("Ours $\\lambda=0.5$", "#1976D2", "o", 55),
        "ours_l10":  ("Ours $\\lambda=1.0$", "#0D47A1", "o", 55),
    }
    rng = np.random.default_rng(42)
    grouped = defaultdict(list)
    for p in pts:
        grouped[p["tag"]].append((p["ce"], p["int_acc"]*100))
    seen = set()
    for tag, (lbl, col, mk, sz) in style.items():
        rows = grouped.get(tag, [])
        if not rows: continue
        ce_arr = np.array([r[0] for r in rows]); int_arr = np.array([r[1] for r in rows])
        # Per-seed scatter
        for ce, ia in rows:
            ax.scatter(ce, ia, s=sz, color=col, marker=mk,
                        edgecolor="white", linewidth=0.6, alpha=0.6,
                        label=lbl if tag not in seen else None)
            seen.add(tag)
        # Mean point + bootstrap 95% CI (cross-hair)
        mu_ce = ce_arr.mean(); mu_ia = int_arr.mean()
        if len(rows) >= 2:
            n = len(rows)
            ce_boot = []; int_boot = []
            for _ in range(2000):
                idx = rng.integers(0, n, size=n)
                ce_boot.append(ce_arr[idx].mean()); int_boot.append(int_arr[idx].mean())
            ce_lo, ce_hi = np.quantile(ce_boot, [0.025, 0.975])
            int_lo, int_hi = np.quantile(int_boot, [0.025, 0.975])
            ax.errorbar(mu_ce, mu_ia, xerr=[[mu_ce-ce_lo],[ce_hi-mu_ce]],
                        yerr=[[mu_ia-int_lo],[int_hi-mu_ia]],
                        color=col, fmt="none", lw=1.4, capsize=3, alpha=0.95)
        ax.scatter(mu_ce, mu_ia, s=110, color=col, marker=mk,
                   edgecolor="black", linewidth=0.8, zorder=5)
    ax.axhline(88.8, color=C_ORACLE, lw=1.0, ls="--", label="Oracle int-acc")
    ax.set_xlabel("CE proxy (lower $=$ more invariant)")
    ax.set_ylabel("Intervened accuracy (%) $\\uparrow$")
    ax.set_title("Utility-Forgetting Pareto Frontier\n(CMNIST, 3 seeds; 95\\% bootstrap CI on means)")
    ax.legend(loc="lower left", fontsize=7.5)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "utility_forgetting_pareto.pdf", bbox_inches="tight")
    plt.close(fig)
    print("  utility_forgetting_pareto.pdf")


if __name__ == "__main__":
    fig_data_efficiency_v2()
    fig_noisy_cfs_v2()
    fig_distance_to_oracle()
    fig_waterbirds_main()
    fig_probing_v2()
    fig_pareto_v2()
    print(f"\nAll v2 figures saved to {FIG_DIR}")
