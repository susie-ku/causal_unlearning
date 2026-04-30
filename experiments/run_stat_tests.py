"""
Statistical significance tests for headline comparisons.

For each headline metric pair (method_A vs method_B), compute:
  - per-seed paired difference
  - Wilcoxon signed-rank test (non-parametric, fewer assumptions than paired t)
  - 95% bootstrap CI on the mean paired difference

Comparisons emitted:
  CMNIST (CE proxy, lower better):
    ours_lambda_0.5 vs baseline
    ours_lambda_0.5 vs intervened_ft
    ours_lambda_0.5 vs naive_ft
  Waterbirds (CE, lower better):
    ours_lambda_0.5 vs baseline
    ours_lambda_0.5 vs distillation
    ours_lambda_0.5 vs concept_erasure
  Waterbirds (avg-acc, higher better):
    ours_lambda_0.5 vs baseline
  Waterbirds (wg-acc, higher better):
    ours_lambda_0.1 vs baseline
    ours_lambda_0.1 vs concept_erasure

Outputs: artifacts/stat_tests.json (also printed to stdout).
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parent.parent


def boot_ci(diffs, n_boot=10000, alpha=0.05, seed=42):
    rng = np.random.default_rng(seed)
    diffs = np.asarray(diffs, dtype=float)
    n = len(diffs)
    means = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        means.append(diffs[idx].mean())
    means = np.array(means)
    lo = np.quantile(means, alpha/2)
    hi = np.quantile(means, 1 - alpha/2)
    return float(lo), float(hi)


def paired_test(name, A_per_seed, B_per_seed, *, lower_is_better=True):
    """A is 'ours', B is comparator.  Difference d = A - B (lower is better => d<0 is good)."""
    A = np.array(A_per_seed); B = np.array(B_per_seed)
    assert A.shape == B.shape, f"{A.shape} vs {B.shape}"
    diffs = A - B
    res = {
        "name": name,
        "n": len(diffs),
        "A_mean": float(A.mean()), "A_std": float(A.std()),
        "B_mean": float(B.mean()), "B_std": float(B.std()),
        "diff_mean": float(diffs.mean()),
        "diff_std": float(diffs.std()),
        "lower_is_better": lower_is_better,
    }
    if len(diffs) >= 3 and not np.allclose(diffs, 0):
        try:
            stat, p = wilcoxon(diffs, alternative="two-sided")
            res["wilcoxon_stat"] = float(stat); res["wilcoxon_p"] = float(p)
        except ValueError as e:
            res["wilcoxon_error"] = str(e)
    lo, hi = boot_ci(diffs)
    res["boot_ci_95"] = [lo, hi]
    return res


# ─────────────────────────────────────────────────────────────────────────────
# CMNIST helpers
# ─────────────────────────────────────────────────────────────────────────────
def cmnist_per_seed():
    """Return per-seed metrics for baseline, ours_l0.5, naive_ft, intervened_ft, distillation."""
    out = {}
    for s in [0, 42, 123]:
        d = json.load(open(ROOT/"artifacts"/"multi_seed"/f"seed_{s}"/"summary.json"))
        out[s] = {
            "baseline":   d["baseline"]["metrics"],
            "ours_l0.5":  next(r for r in d["unlearning_runs"] if abs(r["lambda_ce"]-0.5) < 0.01)["metrics"],
            "ours_l0.1":  next(r for r in d["unlearning_runs"] if abs(r["lambda_ce"]-0.1) < 0.01)["metrics"],
            "ours_l1.0":  next(r for r in d["unlearning_runs"] if abs(r["lambda_ce"]-1.0) < 0.01)["metrics"],
        }
    bls = json.load(open(ROOT/"artifacts"/"baselines_multiseed"/"runs.json"))
    for method, runs in bls.items():
        for r in runs:
            out[r["seed"]][method] = r["metrics"]
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Waterbirds helpers
# ─────────────────────────────────────────────────────────────────────────────
def wb_per_seed():
    out = {}
    for s in [0, 42, 123]:
        d = json.load(open(ROOT/"artifacts"/"waterbirds"/f"seed_{s}"/"summary.json"))
        out[s] = {}
        for method in ["baseline", "oracle", "naive_ft", "intervened_ft", "grl",
                        "concept_erasure", "distillation",
                        "ours_l0.1", "ours_l0.5", "ours_l1.0"]:
            if method in d:
                out[s][method] = d[method]["metrics"]
        # baseline CE comes from ce_metrics
        if "ce_metrics" in d.get("baseline", {}):
            out[s]["baseline"]["ce_proxy"] = d["baseline"]["ce_metrics"].get("ce_proxy")
            out[s]["baseline"]["fidelity_kl"] = d["baseline"]["ce_metrics"].get("fidelity_kl")
        # DFR
        import torch
        dfr_path = ROOT/"artifacts"/"waterbirds"/f"seed_{s}"/"dfr.pt"
        if dfr_path.exists():
            ckpt = torch.load(dfr_path, map_location="cpu", weights_only=False)
            out[s]["dfr"] = ckpt["metrics"]
    return out


# ─────────────────────────────────────────────────────────────────────────────
# CelebA helpers
# ─────────────────────────────────────────────────────────────────────────────
def cb_per_seed():
    out = {}
    for s in [0, 42, 123]:
        main_p = ROOT/"artifacts"/"celeba"/f"seed_{s}"/"summary.json"
        bl_p   = ROOT/"artifacts"/"celeba"/f"seed_{s}"/"baselines_summary.json"
        if not main_p.exists(): continue
        d = json.load(open(main_p))
        out[s] = {}
        for method in ["baseline", "oracle", "ours_l0.1", "ours_l0.5", "ours_l1.0"]:
            if method in d:
                out[s][method] = d[method]["metrics"]
        if "ce_metrics" in d.get("baseline", {}):
            out[s]["baseline"]["ce_proxy"] = d["baseline"]["ce_metrics"].get("ce_proxy")
            out[s]["baseline"]["fidelity_kl"] = d["baseline"]["ce_metrics"].get("fidelity_kl")
        if bl_p.exists():
            db = json.load(open(bl_p))
            for method in ["naive_ft","intervened_ft","grl","concept_erasure","distillation","dfr"]:
                if method in db:
                    out[s][method] = db[method]["metrics"]
    return out


def main():
    cm = cmnist_per_seed()
    wb = wb_per_seed()
    cb = cb_per_seed()
    seeds = [0, 42, 123]

    def get(per_seed, method, key):
        return [per_seed[s][method][key] for s in seeds]

    results = []

    # ─ CMNIST: CE (causal_effect_proxy)
    print("CMNIST CE proxy (lower better):")
    for comparator in ["baseline", "naive_ft", "intervened_ft", "distillation"]:
        A = get(cm, "ours_l0.5", "causal_effect_proxy")
        B = get(cm, comparator, "causal_effect_proxy")
        r = paired_test(f"CMNIST CE: ours_l0.5 vs {comparator}", A, B, lower_is_better=True)
        results.append(r)
        print(f"  ours_l0.5 ({np.mean(A):.3f}) vs {comparator} ({np.mean(B):.3f}): "
              f"diff={r['diff_mean']:+.3f} 95%CI=[{r['boot_ci_95'][0]:+.3f}, {r['boot_ci_95'][1]:+.3f}] "
              f"Wilcoxon p={r.get('wilcoxon_p','--')}")

    # ─ CMNIST: int-acc (higher better)
    print("\nCMNIST intervened accuracy (higher better):")
    for comparator in ["baseline", "naive_ft", "intervened_ft"]:
        A = get(cm, "ours_l0.5", "intervened_accuracy")
        B = get(cm, comparator, "intervened_accuracy")
        r = paired_test(f"CMNIST int-acc: ours_l0.5 vs {comparator}", A, B, lower_is_better=False)
        results.append(r)
        print(f"  ours_l0.5 ({np.mean(A):.3f}) vs {comparator} ({np.mean(B):.3f}): "
              f"diff={r['diff_mean']:+.3f} 95%CI=[{r['boot_ci_95'][0]:+.3f}, {r['boot_ci_95'][1]:+.3f}] "
              f"Wilcoxon p={r.get('wilcoxon_p','--')}")

    # ─ Waterbirds: CE
    print("\nWaterbirds CE proxy (lower better):")
    for comparator in ["baseline", "concept_erasure", "intervened_ft", "distillation", "dfr"]:
        A = get(wb, "ours_l0.5", "ce_proxy")
        B = get(wb, comparator, "ce_proxy")
        r = paired_test(f"Waterbirds CE: ours_l0.5 vs {comparator}", A, B, lower_is_better=True)
        results.append(r)
        print(f"  ours_l0.5 ({np.mean(A):.3f}) vs {comparator} ({np.mean(B):.3f}): "
              f"diff={r['diff_mean']:+.3f} 95%CI=[{r['boot_ci_95'][0]:+.3f}, {r['boot_ci_95'][1]:+.3f}] "
              f"Wilcoxon p={r.get('wilcoxon_p','--')}")

    # ─ Waterbirds: avg-acc
    print("\nWaterbirds avg-acc (higher better):")
    for comparator in ["baseline", "concept_erasure"]:
        A = get(wb, "ours_l0.5", "avg_acc")
        B = get(wb, comparator, "avg_acc")
        r = paired_test(f"Waterbirds avg: ours_l0.5 vs {comparator}", A, B, lower_is_better=False)
        results.append(r)
        print(f"  ours_l0.5 ({np.mean(A):.3f}) vs {comparator} ({np.mean(B):.3f}): "
              f"diff={r['diff_mean']:+.3f} 95%CI=[{r['boot_ci_95'][0]:+.3f}, {r['boot_ci_95'][1]:+.3f}] "
              f"Wilcoxon p={r.get('wilcoxon_p','--')}")

    # ─ Waterbirds: wg-acc
    print("\nWaterbirds wg-acc (higher better):")
    for our_lam, comparator in [("ours_l0.1","baseline"), ("ours_l0.1","concept_erasure"),
                                 ("ours_l0.1","dfr")]:
        A = get(wb, our_lam, "worst_group_acc")
        B = get(wb, comparator, "worst_group_acc")
        r = paired_test(f"Waterbirds wg: {our_lam} vs {comparator}", A, B, lower_is_better=False)
        results.append(r)
        print(f"  {our_lam} ({np.mean(A):.3f}) vs {comparator} ({np.mean(B):.3f}): "
              f"diff={r['diff_mean']:+.3f} 95%CI=[{r['boot_ci_95'][0]:+.3f}, {r['boot_ci_95'][1]:+.3f}] "
              f"Wilcoxon p={r.get('wilcoxon_p','--')}")

    # ─ CelebA: CE
    print("\nCelebA CE proxy (lower better):")
    for our_lam in ["ours_l0.1", "ours_l0.5", "ours_l1.0"]:
        for comparator in ["baseline", "concept_erasure", "dfr", "distillation"]:
            try:
                A = get(cb, our_lam, "ce_proxy")
                B = get(cb, comparator, "ce_proxy")
            except KeyError:
                continue
            r = paired_test(f"CelebA CE: {our_lam} vs {comparator}", A, B, lower_is_better=True)
            results.append(r)
            print(f"  {our_lam} ({np.mean(A):.3f}) vs {comparator} ({np.mean(B):.3f}): "
                  f"diff={r['diff_mean']:+.3f} 95%CI=[{r['boot_ci_95'][0]:+.3f}, {r['boot_ci_95'][1]:+.3f}]")

    # ─ CelebA: wg-acc
    print("\nCelebA wg-acc (higher better):")
    for our_lam in ["ours_l0.1", "ours_l0.5", "ours_l1.0"]:
        for comparator in ["baseline", "concept_erasure", "dfr"]:
            try:
                A = get(cb, our_lam, "worst_group_acc")
                B = get(cb, comparator, "worst_group_acc")
            except KeyError:
                continue
            r = paired_test(f"CelebA wg: {our_lam} vs {comparator}", A, B, lower_is_better=False)
            results.append(r)
            print(f"  {our_lam} ({np.mean(A):.3f}) vs {comparator} ({np.mean(B):.3f}): "
                  f"diff={r['diff_mean']:+.3f} 95%CI=[{r['boot_ci_95'][0]:+.3f}, {r['boot_ci_95'][1]:+.3f}]")

    out_path = ROOT / "artifacts" / "stat_tests.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
