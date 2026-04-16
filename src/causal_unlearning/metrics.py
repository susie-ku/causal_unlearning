from __future__ import annotations

from typing import Mapping

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader


def _kl_per_sample(logits_p: torch.Tensor, logits_q: torch.Tensor) -> torch.Tensor:
    log_q = F.log_softmax(logits_q, dim=1)
    probs_p = F.softmax(logits_p, dim=1)
    return F.kl_div(log_q, probs_p, reduction="none").sum(dim=1)


def symmetric_kl(logits_a: torch.Tensor, logits_b: torch.Tensor) -> torch.Tensor:
    return 0.5 * (_kl_per_sample(logits_a, logits_b) + _kl_per_sample(logits_b, logits_a))


@torch.inference_mode()
def accuracy(model, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    total_correct = 0
    total_examples = 0
    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)
        predictions = model(images).argmax(dim=1)
        total_correct += int((predictions == labels).sum().item())
        total_examples += labels.shape[0]
    return total_correct / max(total_examples, 1)


@torch.inference_mode()
def causal_effect_proxy(model, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    total_effect = 0.0
    total_examples = 0
    for batch in loader:
        factual = batch["image"].to(device, non_blocking=True)
        counterfactual = batch["counterfactual"].to(device, non_blocking=True)
        effect = symmetric_kl(model(factual), model(counterfactual))
        total_effect += float(effect.sum().item())
        total_examples += effect.shape[0]
    return total_effect / max(total_examples, 1)


@torch.inference_mode()
def fidelity_to_oracle(candidate_model, oracle_model, loader: DataLoader, device: torch.device) -> float:
    candidate_model.eval()
    oracle_model.eval()
    total_distance = 0.0
    total_examples = 0
    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        distance = _kl_per_sample(candidate_model(images), oracle_model(images))
        total_distance += float(distance.sum().item())
        total_examples += distance.shape[0]
    return total_distance / max(total_examples, 1)


def evaluate_model(
    model,
    loaders: Mapping[str, DataLoader],
    device: torch.device,
    *,
    oracle_model=None,
) -> dict[str, float]:
    metrics = {
        "observational_accuracy": accuracy(model, loaders["observational_eval"], device),
        "intervened_accuracy": accuracy(model, loaders["intervened_eval"], device),
        "causal_effect_proxy": causal_effect_proxy(model, loaders["intervened_eval"], device),
    }
    if oracle_model is not None:
        metrics["fidelity_to_oracle_kl"] = fidelity_to_oracle(model, oracle_model, loaders["intervened_eval"], device)
    return metrics

