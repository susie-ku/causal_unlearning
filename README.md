# Causal Machine Unlearning via Counterfactual World Modeling

> **Kseniia Kuvshinova · Mohammed Talha Alam**  
> Machine Learning Department, Mohamed Bin Zayed University of AI  
> [[Paper]](https://arxiv.org/abs/xxxx) · [[Code]](https://github.com/talha-alam/causal-unlearning)

---

## What is this?

Most machine unlearning research focuses on *sample-level* forgetting — making a model behave as if certain training rows had been deleted. But many real-world requests are different: a model that learned to classify birds by their background, or predict hair colour using gender, has absorbed a **spurious correlation** woven through the entire dataset. There is no finite set of rows to delete.

This work frames the problem causally. Using Pearl's do-calculus, we ask:

> *How would the model behave if the spurious concept had been **causally independent** of the label during training?*

We fine-tune a pretrained model with a three-term objective:

| Term | Purpose |
|---|---|
| **Task loss** | Retain classification accuracy |
| **Paired-counterfactual KL penalty** | Reduce the concept's causal influence on predictions |
| **Weight locality** | Stay close to the original model, avoiding catastrophic forgetting |

The result is a model whose output distribution approximates what you would get by retraining from scratch on a causally intervened dataset — without actually retraining.

---

## Key Results at a Glance

### Colored MNIST (ρ = 0.9 spurious colour → digit; SmallCNN)

| Model | Obs. Acc. | Int. Acc. | CE Proxy ↓ | D_fid ↓ |
|---|---|---|---|---|
| Baseline | 83.4 ± 3.0 | 54.8 ± 5.0 | 2.00 ± 0.40 | — |
| **Ours** (λ=0.5) | **87.2 ± 2.2** | **73.9 ± 9.1** | **0.27 ± 0.10** | 0.94 ± 0.54 |
| Oracle (retrained) | 88.8 ± 2.3 | 88.8 ± 2.4 | 0.006 ± 0.001 | — |

**86.9% causal effect reduction** in 2 fine-tuning epochs, no group labels required.

### Waterbirds (95% spurious background; ResNet-18)

Our method is the **strongest single-environment post-hoc method** — no group-balanced pool, no oracle access — on average accuracy and CE reduction. DFR wins overall but requires a held-out group-balanced validation set.

### CelebA Blond/Gender (ResNet-50)

Our λ=0.1 variant closes **83% of the oracle worst-group gap**, with the lowest fidelity distance to the oracle and a statistically supported +12.4 pp worst-group gain over baseline (95% CI: [+6.5, +17.9]).

---

## How It Works

```
Training distribution (w):          Intervened world (wτ):
    Y ──► C (ρ = 0.9)          →       Y    C  (C ⊥ Y)
    └──► X (image)                      └──► X (image)
                                    do(C ~ Bern(1/2))
```

For each training example `(x, c, y)` we construct a **paired counterfactual** `(x, c', y)` — same image, flipped concept value. The causal-effect (CE) penalty minimises the symmetric KL divergence between the model's predictions on both:

```python
def composite_loss(model, batch, baseline_params, lambda_ce=0.5, lambda_loc=1e-3):
    x, y   = batch["image"], batch["label"]
    cf     = batch["counterfactual"]          # same image, flipped concept

    logits    = model(x)
    cf_logits = model(cf)

    retain = F.cross_entropy(logits, y)       # keep task utility
    ce     = symmetric_kl(logits, cf_logits).mean()   # forget the shortcut
    loc    = sum(((p - baseline_params[n])**2).mean()
                 for n, p in model.named_parameters()) / len(baseline_params)

    return retain + lambda_ce * ce + lambda_loc * loc
```

`symmetric_kl(p, q)` = ½[KL(p ∥ q) + KL(q ∥ p)] between softmax outputs.

**Proposition 1** (proved in §4): Population minimisation of this CE term implies that the model's output distribution is invariant to interventions on the spurious concept C, for almost every input X.

---

## Installation

```bash
conda create -n causal python=3.10 -y
conda activate causal

# PyTorch (CUDA 12.1)
pip install torch==2.4.1+cu121 torchvision \
    --index-url https://download.pytorch.org/whl/cu121

# Other dependencies
pip install numpy scipy scikit-learn matplotlib pandas Pillow tqdm

# Install this package
pip install -e .
```

**Requirements:** Python ≥ 3.10, PyTorch ≥ 2.4.1

---

## Datasets

### Colored MNIST
Built automatically from `torchvision.datasets.MNIST` — no manual download needed.
```bash
# MNIST is downloaded automatically to data/MNIST/ on first run
python -m causal_unlearning.cli run-default --seed 42
```

### Waterbirds
```bash
wget https://nlp.stanford.edu/data/dro/waterbird_complete95_forest2water2.tar.gz
tar -xzf waterbird_complete95_forest2water2.tar.gz -C data/
```

### CelebA
```bash
pip install gdown
python -c "from torchvision import datasets; datasets.CelebA(root='data', split='train', download=True)"
# Then extract img_align_celeba.zip to data/celeba/img_align_celeba/
```
> **Note:** If the GDrive download hits a rate limit, construct the partition file manually using the standard CelebA image-ID bounds (train: 1–162770, val: 162771–182637, test: 182638–202599). See `experiments/run_celeba.py` for an example.

---

## Reproducing Experiments

### 1 · Colored MNIST
*CPU only · ~10 min per seed*

```bash
# Quick start: single seed, default settings (baseline + oracle + λ sweep + plots)
python -m causal_unlearning.cli run-default --seed 42

# Full multi-seed run (seeds 0, 42, 123) used for the paper tables
python experiments/run_extended.py --task multi_seed
```

**Ablations:**
```bash
python experiments/run_extended.py --task baselines         # GRL + Intervened FT
python experiments/run_extended.py --task lambda_ablation   # λ ∈ {0, 0.1, 0.2, 0.5, 1.0, 2.0}
python experiments/run_extended.py --task beta_ablation     # locality weight β
python experiments/run_extended.py --task epochs_ablation   # 1, 2, 5, 10 epochs
python experiments/run_extended.py --task rho_sweep         # spurious strength ρ
```

All outputs land in `artifacts/{multi_seed,baselines,ablation_*}/`.

---

### 2 · Waterbirds
*GPU recommended · ~50 min wall time (3 seeds in parallel)*

```bash
# Run 3 seeds in parallel on separate GPUs
CUDA_VISIBLE_DEVICES=0 python experiments/run_waterbirds_v2.py \
    --seed 0   --train-epochs 20 --unlearn-epochs 5 &
CUDA_VISIBLE_DEVICES=1 python experiments/run_waterbirds_v2.py \
    --seed 42  --train-epochs 20 --unlearn-epochs 5 &
CUDA_VISIBLE_DEVICES=2 python experiments/run_waterbirds_v2.py \
    --seed 123 --train-epochs 20 --unlearn-epochs 5 &
wait

# Aggregate results across seeds
python experiments/aggregate_waterbirds.py
```

> Models are checkpoint-cached per seed. Re-running skips already-trained checkpoints. Pass `--force` to retrain from scratch.

**DFR baseline** (~5 min/seed):
```bash
for s in 0 42 123; do
  python experiments/run_dfr_waterbirds.py --seed $s --n-epochs 10
done
python experiments/aggregate_waterbirds.py
```

---

### 3 · CelebA
*GPU recommended · ~90 min wall time (3 seeds in parallel)*

```bash
# Main pipeline: baseline, oracle, ours λ-sweep
CUDA_VISIBLE_DEVICES=0 python experiments/run_celeba.py --seed 0   --n-train 6 --n-unlearn 3 &
CUDA_VISIBLE_DEVICES=1 python experiments/run_celeba.py --seed 42  --n-train 6 --n-unlearn 3 &
CUDA_VISIBLE_DEVICES=2 python experiments/run_celeba.py --seed 123 --n-train 6 --n-unlearn 3 &
wait

# Baselines: Naive FT, IFT, GRL, concept erasure, distillation, DFR
CUDA_VISIBLE_DEVICES=0 python experiments/run_celeba_baselines.py --seed 0   &
CUDA_VISIBLE_DEVICES=1 python experiments/run_celeba_baselines.py --seed 42  &
CUDA_VISIBLE_DEVICES=2 python experiments/run_celeba_baselines.py --seed 123 &
wait

python experiments/aggregate_celeba.py
```

---

### 4 · Statistical Tests & Analysis
```bash
# 95% bootstrap CIs + Wilcoxon results → artifacts/stat_tests.json
python experiments/run_stat_tests.py

# Multi-seed analysis: probing, data efficiency, noisy CFs, distributions
python experiments/run_analysis_multiseed.py --task probing
python experiments/run_analysis_multiseed.py --task data_efficiency
python experiments/run_analysis_multiseed.py --task noisy_cfs
python experiments/run_analysis_multiseed.py --task dist
```

Outputs land in `artifacts/analysis_multiseed/`.

---

## Hyperparameters

| Hyperparameter | Baseline / Oracle | Unlearning |
|---|---|---|
| Optimizer | AdamW | AdamW |
| Learning rate | 1 × 10⁻³ | 5 × 10⁻⁴ |
| Weight decay | 1 × 10⁻⁴ | 0 |
| Batch size | 128 | 128 |
| Epochs | 5 | 2 (default), up to 10 |
| λ (CE penalty) | — | {0.0, 0.1, 0.2, 0.5, 1.0, 2.0} |
| β (locality) | — | {0, 10⁻⁴, 10⁻³, 10⁻²} |
| Seeds | {0, 42, 123} | {0, 42, 123} |

**Recommended operating points:**
- Colored MNIST → λ = 0.5 (best utility–forgetting balance)
- Waterbirds → λ = 0.5 (avg-acc + CE), λ = 0.1 (worst-group)
- CelebA → λ = 0.1 (minority group is ~1% of training; aggressive CE suppression hurts)

---

## Repository Layout

```
causal_unlearning/
│
├── src/causal_unlearning/         # Core package
│   ├── data/                      # Dataset loaders & counterfactual construction
│   ├── models/                    # SmallCNN, ResNet wrappers
│   ├── training/                  # Composite loss, fine-tuning loops
│   ├── baselines/                 # GRL, concept erasure, DFR, distillation
│   └── metrics/                   # CE proxy, Dfid, probing utilities
│
├── experiments/
│   ├── run_extended.py            # CMNIST sweeps & ablations
│   ├── run_waterbirds_v2.py       # Multi-seed Waterbirds protocol
│   ├── run_dfr_waterbirds.py      # DFR baseline for Waterbirds
│   ├── run_celeba.py              # CelebA main pipeline
│   ├── run_celeba_baselines.py    # CelebA baselines
│   ├── run_analysis_multiseed.py  # Probing, data-efficiency, noisy-CF analysis
│   ├── run_stat_tests.py          # Bootstrap CIs & Wilcoxon tests
│   ├── aggregate_waterbirds.py    # Seed aggregation for Waterbirds
│   └── aggregate_celeba.py        # Seed aggregation for CelebA
│
└── artifacts/                     # All logs, JSONs, checkpoints (auto-generated)
    ├── default/
    ├── multi_seed/
    ├── baselines/
    ├── ablation_{lambda,beta,epochs,rho}/
    ├── analysis_multiseed/
    ├── waterbirds/seed_{0,42,123}/
    ├── celeba/seed_{0,42,123}/
    └── experiment_ledger.json     # Full record of every experiment run
```

> **Experiment ledger:** `artifacts/experiment_ledger.json` records every experiment with its config, seeds, runtime, and key metrics. All numbers in the paper are sourced directly from these JSON files.

---

## What the Method Does (and Does Not Do)

**It does:**
- Reduce a spurious concept's causal influence on the model's *output* distribution
- Work with as few as 5% of available counterfactual pairs (76% CE reduction)
- Degrade gracefully under moderate counterfactual noise (up to σ ≈ 0.10)
- Operate post-hoc, without group labels, a balanced validation set, or oracle access

**It does not:**
- Erase the concept from internal representations — colour remains 100% linearly decodable after unlearning (by design: this matches the oracle's own regime)
- Provide certified sample-level unlearning or privacy guarantees
- Match group-supervised methods (DFR, IFT) on worst-group accuracy when group labels are available

---

## Citation

```bibtex
@inproceedings{kuvshinova2026causal,
  title     = {Causal Machine Unlearning via Counterfactual World Modeling},
  author    = {Kuvshinova, Kseniia and Alam, Mohammed Talha},
  booktitle = {Preprint},
  year      = {2026}
}
```

---

## Contact

Questions? Open an issue or reach out to the authors:

- **Kseniia Kuvshinova** — kseniia.kuvshinova@mbzuai.ac.ae
- **Mohammed Talha Alam** — mohammed.alam@mbzuai.ac.ae
