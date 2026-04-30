"""Aggregate CelebA multi-seed results into mean ± std summary."""
from __future__ import annotations
import json
from pathlib import Path
from collections import defaultdict
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
CB_DIR = ROOT / "artifacts" / "celeba"
SEEDS = [0, 42, 123]
METHODS_MAIN = ["baseline", "oracle", "ours_l0.1", "ours_l0.5", "ours_l1.0"]
METHODS_BL   = ["naive_ft", "intervened_ft", "grl", "concept_erasure",
                 "distillation", "dfr"]


def aggregate():
    per_method = defaultdict(list)

    for seed in SEEDS:
        # Main run
        main_path = CB_DIR / f"seed_{seed}" / "summary.json"
        if main_path.exists():
            d = json.load(open(main_path))
            for method in METHODS_MAIN:
                if method not in d: continue
                m = d[method]["metrics"]
                ce = m.get("ce_proxy")
                fid = m.get("fidelity_kl")
                if method == "baseline":
                    ce  = d["baseline"].get("ce_metrics", {}).get("ce_proxy", ce)
                    fid = d["baseline"].get("ce_metrics", {}).get("fidelity_kl", fid)
                per_method[method].append({
                    "seed": seed,
                    "avg_acc": m.get("avg_acc"),
                    "wg_acc":  m.get("worst_group_acc"),
                    "ce":      ce,
                    "fid":     fid,
                })

        # Baselines
        bl_path = CB_DIR / f"seed_{seed}" / "baselines_summary.json"
        if bl_path.exists():
            db = json.load(open(bl_path))
            for method in METHODS_BL:
                if method in db:
                    m = db[method]["metrics"]
                    per_method[method].append({
                        "seed": seed,
                        "avg_acc": m.get("avg_acc"),
                        "wg_acc":  m.get("worst_group_acc"),
                        "ce":      m.get("ce_proxy"),
                        "fid":     m.get("fidelity_kl"),
                    })

    summary = {}
    for method in METHODS_MAIN + METHODS_BL:
        if not per_method[method]:
            continue
        rows = per_method[method]
        s = {"n_seeds": len(rows)}
        for k in ["avg_acc", "wg_acc", "ce", "fid"]:
            vals = [r[k] for r in rows if r[k] is not None]
            if vals:
                s[k+"_mean"] = float(np.mean(vals))
                s[k+"_std"] = float(np.std(vals))
        s["per_seed"] = rows
        summary[method] = s

    with open(CB_DIR / "aggregate.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"{'Method':<20} {'n':>2} {'avg':>14} {'wg':>14} {'CE':>14} {'fid':>14}")
    print("-"*82)
    for method in METHODS_MAIN + METHODS_BL:
        if method not in summary: continue
        s = summary[method]
        def fmt(m, sd):
            if m is None: return "    --       "
            return f"{m:.3f}±{sd:.3f}".rjust(14)
        am = s.get("avg_acc_mean"); a_s = s.get("avg_acc_std")
        wm = s.get("wg_acc_mean");  ws = s.get("wg_acc_std")
        cm = s.get("ce_mean");      cs = s.get("ce_std")
        fm = s.get("fid_mean");     fs = s.get("fid_std")
        print(f"{method:<20} {s['n_seeds']:>2} {fmt(am,a_s)} {fmt(wm,ws)} "
              f"{fmt(cm,cs) if cm is not None else '       --     '} "
              f"{fmt(fm,fs) if fm is not None else '       --     '}")


if __name__ == "__main__":
    aggregate()
