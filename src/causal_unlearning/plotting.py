from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

from causal_unlearning.utils import ensure_dir


def _records_from_summary(summary: dict) -> list[dict]:
    records = [
        {
            "name": summary["baseline"]["name"],
            **summary["baseline"]["metrics"],
        },
        {
            "name": summary["oracle"]["name"],
            **summary["oracle"]["metrics"],
        },
    ]
    for run in summary.get("unlearning_runs", []):
        records.append({"name": run["name"], **run["metrics"], "lambda_ce": run["lambda_ce"]})
    return records


def plot_summary(summary: dict, output_dir: str | Path) -> None:
    output_dir = ensure_dir(output_dir)
    records = _records_from_summary(summary)
    _plot_model_comparison(records, output_dir / "model_comparison.png")
    if summary.get("unlearning_runs"):
        _plot_unlearning_sweep(summary["unlearning_runs"], output_dir / "unlearning_sweep.png")


def _plot_model_comparison(records: list[dict], output_path: Path) -> None:
    names = [record["name"] for record in records]
    observational = [record["observational_accuracy"] for record in records]
    intervened = [record["intervened_accuracy"] for record in records]
    width = 0.38
    positions = list(range(len(records)))

    plt.figure(figsize=(10, 5))
    plt.bar([p - width / 2 for p in positions], observational, width=width, label="Observational")
    plt.bar([p + width / 2 for p in positions], intervened, width=width, label="Intervened")
    plt.xticks(positions, names, rotation=20, ha="right")
    plt.ylim(0.0, 1.0)
    plt.ylabel("Accuracy")
    plt.title("Observed vs intervened accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def _plot_unlearning_sweep(runs: list[dict], output_path: Path) -> None:
    lambdas = [run["lambda_ce"] for run in runs]
    cec = [run["metrics"]["causal_effect_proxy"] for run in runs]
    intervened_accuracy = [run["metrics"]["intervened_accuracy"] for run in runs]
    fidelity = [run["metrics"].get("fidelity_to_oracle_kl", 0.0) for run in runs]

    figure, axes = plt.subplots(1, 3, figsize=(14, 4))
    axes[0].plot(lambdas, cec, marker="o")
    axes[0].set_title("Causal effect proxy")
    axes[0].set_xlabel("lambda_ce")
    axes[0].set_ylabel("Symmetric KL")

    axes[1].plot(lambdas, intervened_accuracy, marker="o", color="tab:green")
    axes[1].set_title("Intervened accuracy")
    axes[1].set_xlabel("lambda_ce")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_ylim(0.0, 1.0)

    axes[2].plot(lambdas, fidelity, marker="o", color="tab:red")
    axes[2].set_title("Fidelity to oracle")
    axes[2].set_xlabel("lambda_ce")
    axes[2].set_ylabel("KL to oracle")

    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)

