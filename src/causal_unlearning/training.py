from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Mapping

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from causal_unlearning.config import TrainConfig, UnlearningConfig
from causal_unlearning.metrics import evaluate_model, symmetric_kl


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def _locality_penalty(model, reference_parameters: Mapping[str, torch.Tensor]) -> torch.Tensor:
    penalties = []
    for name, parameter in model.named_parameters():
        penalties.append(F.mse_loss(parameter, reference_parameters[name], reduction="mean"))
    return torch.stack(penalties).mean()


def _epoch_summary(epoch: int, losses: dict[str, float], metrics: dict[str, float]) -> dict[str, float]:
    payload = {"epoch": float(epoch)}
    payload.update(losses)
    payload.update(metrics)
    return payload


def train_supervised(
    model,
    train_loader: DataLoader,
    evaluation_loaders,
    config: TrainConfig,
) -> tuple[torch.nn.Module, list[dict[str, float]]]:
    device = resolve_device(config.device)
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    history: list[dict[str, float]] = []

    for epoch in range(1, config.epochs + 1):
        model.train()
        total_loss = 0.0
        total_correct = 0
        total_examples = 0
        for batch in train_loader:
            images = batch["image"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = F.cross_entropy(logits, labels)
            loss.backward()
            optimizer.step()

            total_loss += float(loss.item()) * labels.shape[0]
            total_correct += int((logits.argmax(dim=1) == labels).sum().item())
            total_examples += labels.shape[0]

        losses = {
            "train_loss": total_loss / max(total_examples, 1),
            "train_accuracy": total_correct / max(total_examples, 1),
        }
        metrics = evaluate_model(model, evaluation_loaders.as_dict(), device)
        history.append(_epoch_summary(epoch, losses, metrics))
        print(
            f"  [supervised] epoch {epoch}/{config.epochs}"
            f"  loss={losses['train_loss']:.4f}"
            f"  acc={losses['train_accuracy']:.3f}"
            f"  obs_acc={metrics.get('observational_accuracy', float('nan')):.3f}"
            f"  int_acc={metrics.get('intervened_accuracy', float('nan')):.3f}",
            flush=True,
        )

    return model, history


def train_unlearning(
    model,
    train_loader: DataLoader,
    evaluation_loaders,
    config: UnlearningConfig,
    *,
    oracle_model=None,
) -> tuple[torch.nn.Module, list[dict[str, float]]]:
    device = resolve_device(config.device)
    model.to(device)
    if oracle_model is not None:
        oracle_model.to(device)
    reference_parameters = {
        name: parameter.detach().clone()
        for name, parameter in model.named_parameters()
    }
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    history: list[dict[str, float]] = []

    print(f"  [unlearning] lambda_ce={config.lambda_ce}  lambda_locality={config.lambda_locality}", flush=True)
    for epoch in range(1, config.epochs + 1):
        model.train()
        total_loss = 0.0
        total_retain = 0.0
        total_ce = 0.0
        total_locality = 0.0
        total_correct = 0
        total_examples = 0

        for batch in train_loader:
            factual = batch["image"].to(device, non_blocking=True)
            counterfactual = batch["counterfactual"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            factual_logits = model(factual)
            counterfactual_logits = model(counterfactual)

            retain_loss = F.cross_entropy(factual_logits, labels)
            ce_loss = symmetric_kl(factual_logits, counterfactual_logits).mean()
            locality_loss = _locality_penalty(model, reference_parameters)
            loss = retain_loss + (config.lambda_ce * ce_loss) + (config.lambda_locality * locality_loss)

            loss.backward()
            optimizer.step()

            batch_size = labels.shape[0]
            total_loss += float(loss.item()) * batch_size
            total_retain += float(retain_loss.item()) * batch_size
            total_ce += float(ce_loss.item()) * batch_size
            total_locality += float(locality_loss.item()) * batch_size
            total_correct += int((factual_logits.argmax(dim=1) == labels).sum().item())
            total_examples += batch_size

        losses = {
            "train_loss": total_loss / max(total_examples, 1),
            "retain_loss": total_retain / max(total_examples, 1),
            "causal_effect_loss": total_ce / max(total_examples, 1),
            "locality_loss": total_locality / max(total_examples, 1),
            "train_accuracy": total_correct / max(total_examples, 1),
        }
        metrics = evaluate_model(model, evaluation_loaders.as_dict(), device, oracle_model=oracle_model)
        history.append(_epoch_summary(epoch, losses, metrics))
        print(
            f"  [unlearning]  epoch {epoch}/{config.epochs}"
            f"  loss={losses['train_loss']:.4f}"
            f"  retain={losses['retain_loss']:.4f}"
            f"  ce={losses['causal_effect_loss']:.4f}"
            f"  acc={losses['train_accuracy']:.3f}",
            flush=True,
        )

    return model, history


def save_checkpoint(
    path: str | Path,
    model,
    *,
    model_name: str,
    history: list[dict[str, float]],
    metadata: dict,
) -> None:
    payload = {
        "model_name": model_name,
        "state_dict": deepcopy(model.state_dict()),
        "history": history,
        "metadata": metadata,
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def load_checkpoint(path: str | Path, device: torch.device | str = "cpu") -> dict:
    resolved_device = resolve_device(device) if isinstance(device, str) else device
    return torch.load(Path(path), map_location=resolved_device)
