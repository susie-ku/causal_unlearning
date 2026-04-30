"""
Aggregate Waterbirds multi-seed results into a single summary table.

Reads artifacts/waterbirds/seed_{0,42,123}/summary.json and emits:
  artifacts/waterbirds/aggregate.json    - mean/std per method
  artifacts/waterbirds/aggregate.csv     - tabular form
"""
from __future__ import annotations
import json
from pathlib import Path
from collections import defaultdict
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
WB_DIR = ROOT / "artifacts" / "waterbirds"
SEEDS  = [0, 42, 123]

METHODS = ["baseline", "oracle", "naive_ft", "intervened_ft", "grl",
           "concept_erasure", "dfr", "distillation",
           "ours_l0.1", "ours_l0.5", "ours_l1.0"]


def aggregate():
    per_method = defaultdict(list)
    for seed in SEEDS:
        path = WB_DIR / f"seed_{seed}" / "summary.json"
        if not path.exists():
            print(f"  SKIP seed {seed}: {path} not found")
            continue
        with open(path) as f:
            d = json.load(f)
        for method in METHODS:
            if method == "dfr":
                # DFR was run separately and saved as dfr.pt
                dfr_path = WB_DIR / f"seed_{seed}" / "dfr.pt"
                if dfr_path.exists():
                    import torch
                    dfr_ckpt = torch.load(dfr_path, map_location="cpu", weights_only=False)
                    m = dfr_ckpt["metrics"]
                else:
                    continue
            elif method in d:
                m = d[method]["metrics"]
            else:
                continue
            if True:
                # baseline also has ce_metrics (CF-enabled)
                ce_proxy = m.get("ce_proxy")
                fid = m.get("fidelity_kl")
                if method == "baseline":
                    ce_proxy = d["baseline"].get("ce_metrics", {}).get("ce_proxy", ce_proxy)
                    fid = d["baseline"].get("ce_metrics", {}).get("fidelity_kl", fid)
                per_method[method].append({
                    "seed": seed,
                    "avg_acc": m.get("avg_acc"),
                    "wg_acc":  m.get("worst_group_acc"),
                    "ce":      ce_proxy,
                    "fid":     fid,
                })

    summary = {}
    for method in METHODS:
        if not per_method[method]: continue
        rows = per_method[method]
        s = {"n_seeds": len(rows)}
        for k in ["avg_acc", "wg_acc", "ce", "fid"]:
            vals = [r[k] for r in rows if r[k] is not None]
            if vals:
                s[k+"_mean"] = float(np.mean(vals))
                s[k+"_std"]  = float(np.std(vals))
        s["per_seed"] = rows
        summary[method] = s

    with open(WB_DIR / "aggregate.json", "w") as f:
        json.dump(summary, f, indent=2)

    # CSV
    lines = ["method,n_seeds,avg_mean,avg_std,wg_mean,wg_std,ce_mean,ce_std,fid_mean,fid_std"]
    for method in METHODS:
        if method not in summary: continue
        s = summary[method]
        lines.append(f"{method},{s['n_seeds']},"
                     f"{s.get('avg_acc_mean','--'):.4f},{s.get('avg_acc_std','--'):.4f},"
                     f"{s.get('wg_acc_mean','--'):.4f},{s.get('wg_acc_std','--'):.4f},"
                     f"{s.get('ce_mean','--')},{s.get('ce_std','--')},"
                     f"{s.get('fid_mean','--')},{s.get('fid_std','--')}")
    with open(WB_DIR / "aggregate.csv", "w") as f:
        f.write("\n".join(lines))

    # Pretty print
    print(f"{'Method':<18} {'n':>2} {'avg':>14} {'wg':>14} {'CE':>14} {'fid':>14}")
    print("-"*80)
    for method in METHODS:
        if method not in summary: continue
        s = summary[method]
        am = s.get("avg_acc_mean"); a_s = s.get("avg_acc_std")
        wm = s.get("wg_acc_mean");  ws = s.get("wg_acc_std")
        cm = s.get("ce_mean");      cs = s.get("ce_std")
        fm = s.get("fid_mean");     fs = s.get("fid_std")
        def fmt(m, s):
            if m is None: return "    --       "
            return f"{m:.3f}±{s:.3f}".rjust(14)
        print(f"{method:<18} {s['n_seeds']:>2} {fmt(am,a_s)} {fmt(wm,ws)} "
              f"{fmt(cm,cs) if cm is not None else '       --     '} "
              f"{fmt(fm,fs) if fm is not None else '       --     '}")
    return summary


if __name__ == "__main__":
    aggregate()
