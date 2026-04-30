# Causal Machine Unlearning via Counterfactual World Modeling

This repository implements a post-hoc concept-level
unlearning method based on a paired-counterfactual causal-effect (CE) penalty,
and reproduces all main experiments on Colored MNIST and Waterbirds.

---

## Headline numbers (3 seeds)

**Colored MNIST** ($\rho=0.9$ spurious color $\to$ digit; SmallCNN; $\lambda{=}0.5$, 2 epochs):

| | obs-acc | int-acc | $\mathrm{CE}_C$ | $\mathcal{D}_\mathrm{fid}$ |
|---|---|---|---|---|
| Baseline   | $83.4 \pm 3.0$ | $54.8 \pm 5.0$ | $2.00 \pm 0.40$ | --- |
| **Ours**   | $87.2 \pm 2.2$ | $73.9 \pm 9.1$ | $0.27 \pm 0.10$ | $0.94 \pm 0.54$ |
| Oracle     | $88.8 \pm 2.3$ | $88.8 \pm 2.4$ | $0.006 \pm 0.001$ | --- |

**Waterbirds** (95% spurious background $\to$ bird type; ResNet-18; 5 ft epochs):

| Method | avg-acc | wg-acc | $\mathrm{CE}_C$ |
|---|---|---|---|
| Baseline (ERM) | $83.5\pm1.6$ | $57.6\pm1.2$ | $1.56\pm0.34$ |
| Concept erasure | $83.2\pm0.7$ | $56.1\pm4.2$ | $1.58\pm0.13$ |
| GRL (group labels) | $75.8\pm8.5$ | $50.0\pm10.5$ | $1.13\pm0.85$ |
| Intervened FT (group labels) | $82.0\pm1.1$ | $65.5\pm4.8$ | $1.74\pm0.04$ |
| **DFR (group-balanced val)** | $\mathbf{90.3\pm0.5}$ | $\mathbf{87.3\pm1.2}$ | $0.41\pm0.02$ |
| Oracle distill. | $84.6\pm1.0$ | $68.5\pm4.8$ | $1.29\pm0.15$ |
| **Ours** $\lambda{=}0.5$ (only paired CFs) | $85.2\pm1.1$ | $58.5\pm9.6$ | $\mathbf{0.97\pm0.24}$ |
| **Ours** $\lambda{=}0.1$ (only paired CFs) | $83.1\pm1.7$ | $63.7\pm5.5$ | $1.43\pm0.50$ |
| Oracle (group-balanced training) | $82.8\pm2.1$ | $64.7\pm2.4$ | --- |

DFR with a held-out group-balanced reweighting set is the strongest method on
Waterbirds.  Our method is the strongest \emph{single-environment} post-hoc
method (no group-balanced pool, no oracle access) on average accuracy and
$\mathrm{CE}_C$.

**CelebA blond/gender** (ResNet-50, 20k train subset, 3 ft epochs):

| Method | avg-acc | wg-acc | $\mathrm{CE}_C$ | $\mathcal{D}_\mathrm{fid}$ |
|---|---|---|---|---|
| Baseline (ERM) | $95.5\pm0.4$ | $51.4\pm4.2$ | $0.47\pm0.07$ | --- |
| Naive FT | $94.7\pm1.1$ | $59.6\pm14.9$ | $0.55\pm0.15$ | $0.62\pm0.18$ |
| Concept erasure | $95.6\pm0.5$ | $53.1\pm4.3$ | $0.64\pm0.18$ | $1.09\pm0.38$ |
| Intervened FT (group lab) | $92.3\pm0.9$ | $\mathbf{76.2\pm5.3}$ | $0.81\pm0.07$ | $0.46\pm0.10$ |
| GRL (group lab) | $93.4\pm2.1$ | $58.6\pm18.4$ | $0.28\pm0.16$ | $0.42\pm0.12$ |
| **DFR (group-balanced val)** | $93.8\pm0.4$ | $73.2\pm2.6$ | $0.53\pm0.01$ | $0.53\pm0.04$ |
| Oracle distill. | $94.2\pm0.9$ | $67.9\pm1.1$ | $0.51\pm0.07$ | $0.44\pm0.07$ |
| **Ours** $\lambda{=}0.1$ (only paired CFs) | $95.0\pm0.1$ | $63.8\pm7.5$ | $0.35\pm0.05$ | $\mathbf{0.08\pm0.01}$ |
| **Ours** $\lambda{=}1.0$ (only paired CFs) | $95.2\pm0.3$ | $45.1\pm16.1$ | $\mathbf{0.16\pm0.04}$ | $0.10\pm0.01$ |
| Oracle (group-balanced training) | $93.5\pm0.7$ | $66.4\pm7.9$ | --- | --- |

Ours $\lambda{=}0.1$ closes 83% of the oracle worst-group gap on CelebA, with
the lowest distance to oracle and a statistically supported wg-acc gain over
baseline ($+12.4$pp, 95% CI $[+6.5, +17.9]$).

---

## Setup

```bash
conda create -n causal python=3.10 -y
conda activate causal
pip install torch==2.4.1+cu121 torchvision \
    --index-url https://download.pytorch.org/whl/cu121
pip install numpy scipy scikit-learn matplotlib pandas Pillow tqdm
pip install -e .
```

Python ≥3.10 and PyTorch 2.4.1+ are recommended.

### Datasets

- **Colored MNIST** is built deterministically from `torchvision.datasets.MNIST`
  (downloaded automatically into `data/MNIST/`).
- **Waterbirds**: download from
  `https://nlp.stanford.edu/data/dro/waterbird_complete95_forest2water2.tar.gz`
  and extract into `data/waterbird_complete95_forest2water2/`.
- **CelebA**: `python -c "from torchvision import datasets;
  datasets.CelebA(root='data', split='train', download=True)"` (requires
  `pip install gdown`).  Extract `img_align_celeba.zip` to
  `data/celeba/img_align_celeba/`.  If the metadata files fail (gdrive rate
  limit), construct the partition file manually using the standard CelebA
  bounds (train: 1--162770, val: 162771--182637, test: 182638--202599); see
  `experiments/run_celeba.py` for an example.

---

## Reproducing the main experiments

### Colored MNIST (CPU, ~10 min total per seed)

```bash
# Default pipeline at seed 42 (baseline, oracle, lambda sweep, default plots)
python -m causal_unlearning.cli run-default --seed 42

# Multi-seed run (seeds 0, 42, 123) used for the main table
python experiments/run_extended.py --task multi_seed

# Other ablations referenced in the paper
python experiments/run_extended.py --task baselines      # GRL + IFT
python experiments/run_extended.py --task lambda_ablation
python experiments/run_extended.py --task beta_ablation
python experiments/run_extended.py --task epochs_ablation
python experiments/run_extended.py --task rho_sweep
```

Outputs land under `artifacts/{default,multi_seed,baselines,ablation_*}/`.

### Waterbirds (GPU; ~50 min wall time for 3 seeds in parallel)

```bash
# One seed at a time (or run 3 in parallel on different GPUs):
CUDA_VISIBLE_DEVICES=4 python experiments/run_waterbirds_v2.py \
    --seed 0   --train-epochs 20 --unlearn-epochs 5 &
CUDA_VISIBLE_DEVICES=5 python experiments/run_waterbirds_v2.py \
    --seed 42  --train-epochs 20 --unlearn-epochs 5 &
CUDA_VISIBLE_DEVICES=6 python experiments/run_waterbirds_v2.py \
    --seed 123 --train-epochs 20 --unlearn-epochs 5 &
wait

python experiments/aggregate_waterbirds.py
```

Each method is checkpoint-cached; re-running skips already-trained models.
Pass `--force` to retrain.

### DFR baseline on Waterbirds (GPU; ~5 min/seed)

```bash
for s in 0 42 123; do
  CUDA_VISIBLE_DEVICES=$((s%3+4)) python experiments/run_dfr_waterbirds.py \
      --seed $s --n-epochs 10
done
python experiments/aggregate_waterbirds.py
```

### CelebA (GPU; ~90 min wall time for 3 seeds in parallel)

```bash
# Main pipeline (baseline, oracle, ours-sweep) — 3 seeds in parallel:
CUDA_VISIBLE_DEVICES=4 python experiments/run_celeba.py --seed 0   --n-train 6 --n-unlearn 3 &
CUDA_VISIBLE_DEVICES=5 python experiments/run_celeba.py --seed 42  --n-train 6 --n-unlearn 3 &
CUDA_VISIBLE_DEVICES=6 python experiments/run_celeba.py --seed 123 --n-train 6 --n-unlearn 3 &
wait

# Baselines (Naive FT, IFT, GRL, concept erasure, distillation, DFR):
CUDA_VISIBLE_DEVICES=4 python experiments/run_celeba_baselines.py --seed 0   &
CUDA_VISIBLE_DEVICES=5 python experiments/run_celeba_baselines.py --seed 42  &
CUDA_VISIBLE_DEVICES=6 python experiments/run_celeba_baselines.py --seed 123 &
wait

python experiments/aggregate_celeba.py
```

### Statistical tests

```bash
python experiments/run_stat_tests.py
# 95% bootstrap CIs and Wilcoxon results saved to artifacts/stat_tests.json
```

### Multi-seed analysis (CMNIST; CPU; ~30 min)

```bash
python experiments/run_analysis_multiseed.py --task probing
python experiments/run_analysis_multiseed.py --task data_efficiency
python experiments/run_analysis_multiseed.py --task noisy_cfs
python experiments/run_analysis_multiseed.py --task dist
```

Outputs land under `artifacts/analysis_multiseed/`.

### Figures and PDF

```bash
cd paper
python generate_figures.py     # original CMNIST figures
python generate_figures_v2.py  # multi-seed + Waterbirds figures
pdflatex paper.tex && bibtex paper && pdflatex paper.tex && pdflatex paper.tex
```

---

## Method in 30 lines

```python
def composite_loss(model, batch, baseline_params, lambda_ce=0.5, lambda_loc=1e-3):
    x, y = batch["image"], batch["label"]
    cf   = batch["counterfactual"]                     # paired CF (same Y)
    logits, cf_logits = model(x), model(cf)
    retain = F.cross_entropy(logits, y)                # task utility
    ce     = symmetric_kl(logits, cf_logits).mean()    # causal-effect proxy
    loc    = sum(((p - baseline_params[n])**2).mean()
                  for n, p in model.named_parameters()) / len(baseline_params)
    return retain + lambda_ce * ce + lambda_loc * loc
```

`symmetric_kl` is $\frac{1}{2}[\mathrm{KL}(p\|q) + \mathrm{KL}(q\|p)]$ between
`softmax(logits)` and `softmax(cf_logits)`.  Proposition 1 shows that the
population minimiser of this CE term satisfies counterfactual output invariance.

---

## Repository layout

```
causal_unlearning_new/
├── src/causal_unlearning/         # package: data, models, training, baselines, metrics
├── experiments/
│   ├── run_extended.py            # CMNIST extended sweeps
│   ├── run_waterbirds_v2.py       # multi-seed Waterbirds protocol
│   ├── run_analysis_multiseed.py  # multi-seed CMNIST analysis
│   └── aggregate_waterbirds.py    # mean/std aggregation
├── paper/
│   ├── paper.tex                  # NeurIPS submission
│   ├── references.bib
│   ├── generate_figures.py        # original figures
│   ├── generate_figures_v2.py     # multi-seed/Waterbirds figures
│   └── figures/                   # 14 PDFs
├── artifacts/                     # all logs, JSONs, checkpoints (per-seed)
│   ├── default/
│   ├── multi_seed/
│   ├── baselines/
│   ├── ablation_{lambda,beta,epochs,rho}/
│   ├── analysis/                  # single-seed analysis (legacy)
│   ├── analysis_multiseed/        # 3-seed analysis (paper)
│   └── waterbirds/seed_{0,42,123}/
└── README.md
```

---

## Experiment ledger

`artifacts/experiment_ledger.json` lists every experiment with its config,
seeds, runtime, and key metrics.  Numbers in the paper are sourced directly
from these JSON files (see `paper/generate_figures*.py`).

---
```
