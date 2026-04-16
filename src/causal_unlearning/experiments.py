from __future__ import annotations

from copy import deepcopy
from pathlib import Path

from causal_unlearning.config import DataConfig, RunConfig, TrainConfig, UnlearningConfig
from causal_unlearning.datasets import DatasetBundle, build_dataloaders
from causal_unlearning.metrics import evaluate_model
from causal_unlearning.models import build_model
from causal_unlearning.plotting import plot_summary
from causal_unlearning.training import (
    load_checkpoint,
    resolve_device,
    save_checkpoint,
    train_supervised,
    train_unlearning,
)
from causal_unlearning.utils import ensure_dir, format_float_tag, save_json, set_seed


def _artifact_dirs(output_dir: str | Path) -> dict[str, Path]:
    root = ensure_dir(output_dir)
    return {
        "root": root,
        "checkpoints": ensure_dir(root / "checkpoints"),
        "metrics": ensure_dir(root / "metrics"),
        "plots": ensure_dir(root / "plots"),
    }


def _load_model_from_checkpoint(checkpoint_path: str | Path, *, device: str) -> tuple:
    payload = load_checkpoint(checkpoint_path, device=device)
    model = build_model(payload["model_name"])
    model.load_state_dict(payload["state_dict"])
    model.to(resolve_device(device))
    return model, payload


def train_single_model(
    *,
    name: str,
    world: str,
    data_config: DataConfig,
    train_config: TrainConfig,
    output_dir: str | Path,
) -> dict:
    if world not in {"observational", "intervened"}:
        raise ValueError(f"Unsupported world '{world}'.")
    set_seed(data_config.seed)
    artifact_dirs = _artifact_dirs(output_dir)
    loaders = build_dataloaders(data_config)
    model = build_model("small_cnn")
    train_loader = loaders.observational_train if world == "observational" else loaders.intervened_train
    model, history = train_supervised(model, train_loader, loaders, train_config)
    device = resolve_device(train_config.device)
    metrics = evaluate_model(model, loaders.as_dict(), device)
    checkpoint_path = artifact_dirs["checkpoints"] / f"{name}.pt"
    save_checkpoint(
        checkpoint_path,
        model,
        model_name="small_cnn",
        history=history,
        metadata={
            "name": name,
            "world": world,
            "data_config": data_config,
            "train_config": train_config,
            "metrics": metrics,
        },
    )
    result = {
        "name": name,
        "world": world,
        "checkpoint": str(checkpoint_path),
        "history": history,
        "metrics": metrics,
    }
    save_json(artifact_dirs["metrics"] / f"{name}.json", result)
    return result


def unlearn_from_checkpoint(
    *,
    name: str,
    baseline_checkpoint: str | Path,
    oracle_checkpoint: str | Path | None,
    data_config: DataConfig,
    unlearning_config: UnlearningConfig,
    output_dir: str | Path,
) -> dict:
    set_seed(data_config.seed)
    artifact_dirs = _artifact_dirs(output_dir)
    loaders = build_dataloaders(data_config)
    device = unlearning_config.device

    baseline_model, baseline_payload = _load_model_from_checkpoint(baseline_checkpoint, device=device)
    oracle_model = None
    if oracle_checkpoint is not None:
        oracle_model, _ = _load_model_from_checkpoint(oracle_checkpoint, device=device)

    model = build_model(baseline_payload["model_name"])
    model.load_state_dict(deepcopy(baseline_payload["state_dict"]))
    model, history = train_unlearning(
        model,
        loaders.observational_train,
        loaders,
        unlearning_config,
        oracle_model=oracle_model,
    )
    metrics = evaluate_model(model, loaders.as_dict(), resolve_device(device), oracle_model=oracle_model)
    checkpoint_path = artifact_dirs["checkpoints"] / f"{name}.pt"
    save_checkpoint(
        checkpoint_path,
        model,
        model_name=baseline_payload["model_name"],
        history=history,
        metadata={
            "name": name,
            "baseline_checkpoint": str(baseline_checkpoint),
            "oracle_checkpoint": None if oracle_checkpoint is None else str(oracle_checkpoint),
            "data_config": data_config,
            "unlearning_config": unlearning_config,
            "metrics": metrics,
        },
    )
    result = {
        "name": name,
        "checkpoint": str(checkpoint_path),
        "history": history,
        "metrics": metrics,
        "lambda_ce": unlearning_config.lambda_ce,
        "lambda_locality": unlearning_config.lambda_locality,
    }
    save_json(artifact_dirs["metrics"] / f"{name}.json", result)
    return result


def evaluate_checkpoints(
    *,
    checkpoint_paths: list[str],
    oracle_checkpoint: str | None,
    data_config: DataConfig,
    device: str,
    output_json: str | None = None,
) -> dict:
    set_seed(data_config.seed)
    loaders = build_dataloaders(data_config)
    oracle_model = None
    if oracle_checkpoint is not None:
        oracle_model, _ = _load_model_from_checkpoint(oracle_checkpoint, device=device)

    results: dict[str, dict] = {}
    resolved_device = resolve_device(device)
    for checkpoint_path in checkpoint_paths:
        model, payload = _load_model_from_checkpoint(checkpoint_path, device=device)
        results[Path(checkpoint_path).stem] = {
            "checkpoint": checkpoint_path,
            "model_name": payload["model_name"],
            "metrics": evaluate_model(model, loaders.as_dict(), resolved_device, oracle_model=oracle_model),
        }

    if output_json is not None:
        save_json(output_json, results)
    return results


def run_full_pipeline(config: RunConfig) -> dict:
    set_seed(config.data.seed)
    artifact_dirs = _artifact_dirs(config.output_dir)
    save_json(artifact_dirs["root"] / "config.json", config.to_dict())

    loaders = build_dataloaders(config.data)
    baseline_model = build_model("small_cnn")
    baseline_model, baseline_history = train_supervised(
        baseline_model,
        loaders.observational_train,
        loaders,
        config.baseline,
    )
    baseline_metrics = evaluate_model(
        baseline_model,
        loaders.as_dict(),
        resolve_device(config.baseline.device),
    )
    baseline_checkpoint = artifact_dirs["checkpoints"] / "baseline.pt"
    save_checkpoint(
        baseline_checkpoint,
        baseline_model,
        model_name="small_cnn",
        history=baseline_history,
        metadata={"world": "observational", "metrics": baseline_metrics},
    )

    oracle_model = build_model("small_cnn")
    oracle_model, oracle_history = train_supervised(
        oracle_model,
        loaders.intervened_train,
        loaders,
        config.oracle,
    )
    oracle_metrics = evaluate_model(
        oracle_model,
        loaders.as_dict(),
        resolve_device(config.oracle.device),
    )
    oracle_checkpoint = artifact_dirs["checkpoints"] / "oracle.pt"
    save_checkpoint(
        oracle_checkpoint,
        oracle_model,
        model_name="small_cnn",
        history=oracle_history,
        metadata={"world": "intervened", "metrics": oracle_metrics},
    )

    baseline_result = {
        "name": "baseline",
        "checkpoint": str(baseline_checkpoint),
        "history": baseline_history,
        "metrics": baseline_metrics,
    }
    oracle_result = {
        "name": "oracle",
        "checkpoint": str(oracle_checkpoint),
        "history": oracle_history,
        "metrics": oracle_metrics,
    }
    save_json(artifact_dirs["metrics"] / "baseline.json", baseline_result)
    save_json(artifact_dirs["metrics"] / "oracle.json", oracle_result)

    unlearning_runs = []
    for lambda_ce in config.lambda_ce_values:
        unlearning_config = UnlearningConfig(
            epochs=config.unlearning.epochs,
            lr=config.unlearning.lr,
            weight_decay=config.unlearning.weight_decay,
            device=config.unlearning.device,
            lambda_ce=lambda_ce,
            lambda_locality=config.unlearning.lambda_locality,
        )
        model = build_model("small_cnn")
        model.load_state_dict(deepcopy(baseline_model.state_dict()))
        model, history = train_unlearning(
            model,
            loaders.observational_train,
            loaders,
            unlearning_config,
            oracle_model=oracle_model,
        )
        metrics = evaluate_model(
            model,
            loaders.as_dict(),
            resolve_device(unlearning_config.device),
            oracle_model=oracle_model,
        )
        name = f"unlearn_lambda_{format_float_tag(lambda_ce)}"
        checkpoint_path = artifact_dirs["checkpoints"] / f"{name}.pt"
        save_checkpoint(
            checkpoint_path,
            model,
            model_name="small_cnn",
            history=history,
            metadata={"lambda_ce": lambda_ce, "metrics": metrics},
        )
        run = {
            "name": name,
            "lambda_ce": lambda_ce,
            "checkpoint": str(checkpoint_path),
            "history": history,
            "metrics": metrics,
        }
        unlearning_runs.append(run)
        save_json(artifact_dirs["metrics"] / f"{name}.json", run)

    summary = {
        "config": config.to_dict(),
        "baseline": baseline_result,
        "oracle": oracle_result,
        "unlearning_runs": unlearning_runs,
    }
    save_json(artifact_dirs["root"] / "summary.json", summary)
    plot_summary(summary, artifact_dirs["plots"])
    return summary
