"""Competing baseline unlearning methods for comparison."""
from __future__ import annotations

from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from causal_unlearning.config import DataConfig, UnlearningConfig
from causal_unlearning.metrics import evaluate_model, symmetric_kl
from causal_unlearning.models import SmallCNN
from causal_unlearning.training import _locality_penalty, _epoch_summary, resolve_device


class GradientReversalFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.save_for_backward(torch.tensor(alpha))
        return x.clone()

    @staticmethod
    def backward(ctx, grad_output):
        alpha = ctx.saved_tensors[0].item()
        return -alpha * grad_output, None


def gradient_reversal(x: torch.Tensor, alpha: float = 1.0) -> torch.Tensor:
    return GradientReversalFunction.apply(x, alpha)


class ColorDiscriminator(nn.Module):
    """Binary discriminator that predicts color (0=red, 1=green) from features."""

    def __init__(self, in_features: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 2),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features)


class GRLUnlearner(nn.Module):
    """SmallCNN + adversarial color discriminator via gradient reversal."""

    def __init__(self, base_model: SmallCNN) -> None:
        super().__init__()
        self.features = base_model.features
        self.classifier = base_model.classifier
        self.discriminator = ColorDiscriminator(in_features=128)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x).flatten(1))

    def forward_with_disc(
        self, x: torch.Tensor, alpha: float
    ) -> tuple[torch.Tensor, torch.Tensor]:
        feats = self.features(x).flatten(1)
        label_logits = self.classifier(feats)
        disc_logits = self.discriminator(gradient_reversal(feats, alpha))
        return label_logits, disc_logits


def train_grl_unlearning(
    model: SmallCNN,
    train_loader: DataLoader,
    evaluation_loaders,
    config: UnlearningConfig,
    *,
    oracle_model=None,
    grl_alpha: float = 1.0,
) -> tuple[GRLUnlearner, list[dict[str, float]]]:
    """Adversarial debiasing via gradient reversal (domain-adversarial training)."""
    device = resolve_device(config.device)
    grl_model = GRLUnlearner(deepcopy(model)).to(device)
    if oracle_model is not None:
        oracle_model.to(device)

    reference_parameters = {
        name: param.detach().clone()
        for name, param in grl_model.named_parameters()
        if "discriminator" not in name
    }

    optimizer = torch.optim.AdamW(grl_model.parameters(), lr=config.lr)
    history: list[dict[str, float]] = []

    print(f"  [grl] alpha={grl_alpha}  lambda_locality={config.lambda_locality}", flush=True)
    for epoch in range(1, config.epochs + 1):
        grl_model.train()
        total_loss = total_retain = total_adv = total_loc = 0.0
        total_correct = total_examples = 0

        for batch in train_loader:
            factual = batch["image"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True)
            colors = batch["color"].to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            label_logits, disc_logits = grl_model.forward_with_disc(factual, grl_alpha)

            retain_loss = F.cross_entropy(label_logits, labels)
            adv_loss = F.cross_entropy(disc_logits, colors)
            loc_params = {
                k: v for k, v in grl_model.named_parameters() if "discriminator" not in k
            }
            locality_loss = _locality_penalty_named(loc_params, reference_parameters)
            loss = retain_loss + config.lambda_ce * adv_loss + config.lambda_locality * locality_loss
            loss.backward()
            optimizer.step()

            bs = labels.shape[0]
            total_loss += float(loss.item()) * bs
            total_retain += float(retain_loss.item()) * bs
            total_adv += float(adv_loss.item()) * bs
            total_loc += float(locality_loss.item()) * bs
            total_correct += int((label_logits.argmax(1) == labels).sum().item())
            total_examples += bs

        losses = {
            "train_loss": total_loss / max(total_examples, 1),
            "retain_loss": total_retain / max(total_examples, 1),
            "adv_loss": total_adv / max(total_examples, 1),
            "locality_loss": total_loc / max(total_examples, 1),
            "train_accuracy": total_correct / max(total_examples, 1),
        }

        # Wrap GRL model for evaluate_model (it exposes standard forward())
        class _Wrapper(nn.Module):
            def __init__(self, m): super().__init__(); self.m = m
            def forward(self, x): return self.m(x)

        wrapper = _Wrapper(grl_model).to(device)
        metrics = evaluate_model(wrapper, evaluation_loaders.as_dict(), device, oracle_model=oracle_model)
        history.append(_epoch_summary(epoch, losses, metrics))
        print(
            f"  [grl]  epoch {epoch}/{config.epochs}"
            f"  loss={losses['train_loss']:.4f}"
            f"  adv={losses['adv_loss']:.4f}"
            f"  acc={losses['train_accuracy']:.3f}",
            flush=True,
        )

    return grl_model, history


def _locality_penalty_named(
    named_params: dict[str, torch.Tensor],
    reference: dict[str, torch.Tensor],
) -> torch.Tensor:
    penalties = []
    for name, param in named_params.items():
        if name in reference:
            penalties.append(F.mse_loss(param, reference[name], reduction="mean"))
    if not penalties:
        return torch.tensor(0.0)
    return torch.stack(penalties).mean()


def train_intervened_finetuning(
    model: SmallCNN,
    intervened_train_loader: DataLoader,
    evaluation_loaders,
    config: UnlearningConfig,
    *,
    oracle_model=None,
) -> tuple[SmallCNN, list[dict[str, float]]]:
    """Naive fine-tuning on the intervened dataset (C independent of Y).

    This is a strong oracle-adjacent baseline when paired counterfactuals are
    available as intervened samples.  It directly optimises on the distribution
    where the spurious correlation has been removed, without any causal penalty.
    """
    device = resolve_device(config.device)
    ft_model = deepcopy(model).to(device)
    if oracle_model is not None:
        oracle_model.to(device)

    reference_parameters = {
        name: param.detach().clone() for name, param in ft_model.named_parameters()
    }
    optimizer = torch.optim.AdamW(ft_model.parameters(), lr=config.lr)
    history: list[dict[str, float]] = []

    print(f"  [intervened-ft] lambda_locality={config.lambda_locality}", flush=True)
    for epoch in range(1, config.epochs + 1):
        ft_model.train()
        total_loss = total_retain = total_loc = 0.0
        total_correct = total_examples = 0

        for batch in intervened_train_loader:
            images = batch["image"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            logits = ft_model(images)
            retain_loss = F.cross_entropy(logits, labels)
            locality_loss = _locality_penalty(ft_model, reference_parameters)
            loss = retain_loss + config.lambda_locality * locality_loss
            loss.backward()
            optimizer.step()

            bs = labels.shape[0]
            total_loss += float(loss.item()) * bs
            total_retain += float(retain_loss.item()) * bs
            total_loc += float(locality_loss.item()) * bs
            total_correct += int((logits.argmax(1) == labels).sum().item())
            total_examples += bs

        losses = {
            "train_loss": total_loss / max(total_examples, 1),
            "retain_loss": total_retain / max(total_examples, 1),
            "locality_loss": total_loc / max(total_examples, 1),
            "train_accuracy": total_correct / max(total_examples, 1),
        }
        metrics = evaluate_model(ft_model, evaluation_loaders.as_dict(), device, oracle_model=oracle_model)
        history.append(_epoch_summary(epoch, losses, metrics))
        print(
            f"  [intervened-ft]  epoch {epoch}/{config.epochs}"
            f"  loss={losses['train_loss']:.4f}"
            f"  acc={losses['train_accuracy']:.3f}",
            flush=True,
        )

    return ft_model, history
