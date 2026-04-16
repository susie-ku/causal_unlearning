from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import MNIST

from causal_unlearning.config import DataConfig

World = Literal["observational", "intervened"]
RED = 0
GREEN = 1


def _select_indices(total_size: int, subset_size: int, seed: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    permutation = torch.randperm(total_size, generator=generator)
    return permutation[:subset_size]


def _majority_color(labels: torch.Tensor) -> torch.Tensor:
    return (labels >= 5).long()


def _assign_colors(labels: torch.Tensor, world: World, correlation: float, seed: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    if world == "intervened":
        return torch.randint(0, 2, labels.shape, generator=generator)
    colors = _majority_color(labels)
    if correlation >= 1.0:
        return colors
    if correlation <= 0.0:
        return 1 - colors
    flips = torch.rand(labels.shape, generator=generator) > correlation
    colors = colors.clone()
    colors[flips] = 1 - colors[flips]
    return colors


def colorize_digit(grayscale: torch.Tensor, color_id: int) -> torch.Tensor:
    image = torch.zeros(3, grayscale.shape[-2], grayscale.shape[-1], dtype=grayscale.dtype)
    image[color_id] = grayscale.squeeze(0)
    return image


def counterfactual_color(color_id: int) -> int:
    return GREEN if color_id == RED else RED


@dataclass(slots=True)
class DatasetBundle:
    observational_train: DataLoader
    intervened_train: DataLoader
    observational_eval: DataLoader
    intervened_eval: DataLoader

    def as_dict(self) -> dict[str, DataLoader]:
        return {
            "observational_train": self.observational_train,
            "intervened_train": self.intervened_train,
            "observational_eval": self.observational_eval,
            "intervened_eval": self.intervened_eval,
        }


class ColoredMNISTDataset(Dataset):
    def __init__(
        self,
        *,
        root: str,
        split: Literal["train", "test"],
        world: World,
        size: int,
        correlation: float,
        seed: int,
        download: bool,
    ) -> None:
        train_split = split == "train"
        mnist = MNIST(root=root, train=train_split, download=download)
        indices = _select_indices(len(mnist), size, seed)
        self.images = mnist.data[indices].float().div(255.0).unsqueeze(1)
        self.labels = mnist.targets[indices].long()
        self.colors = _assign_colors(self.labels, world, correlation, seed + 97)

    def __len__(self) -> int:
        return self.labels.shape[0]

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        grayscale = self.images[index]
        label = self.labels[index]
        color = int(self.colors[index].item())
        factual = colorize_digit(grayscale, color)
        counterfactual = colorize_digit(grayscale, counterfactual_color(color))
        return {
            "image": factual,
            "counterfactual": counterfactual,
            "label": label,
            "color": torch.tensor(color, dtype=torch.long),
        }


def build_dataloaders(config: DataConfig) -> DatasetBundle:
    loader_kwargs = {
        "batch_size": config.batch_size,
        "num_workers": config.num_workers,
        "pin_memory": torch.cuda.is_available(),
    }
    observational_train = ColoredMNISTDataset(
        root=config.root,
        split="train",
        world="observational",
        size=config.train_size,
        correlation=config.observational_correlation,
        seed=config.seed,
        download=config.download,
    )
    intervened_train = ColoredMNISTDataset(
        root=config.root,
        split="train",
        world="intervened",
        size=config.train_size,
        correlation=config.observational_correlation,
        seed=config.seed,
        download=config.download,
    )
    observational_eval = ColoredMNISTDataset(
        root=config.root,
        split="test",
        world="observational",
        size=config.eval_size,
        correlation=config.observational_correlation,
        seed=config.seed + 1,
        download=config.download,
    )
    intervened_eval = ColoredMNISTDataset(
        root=config.root,
        split="test",
        world="intervened",
        size=config.eval_size,
        correlation=config.observational_correlation,
        seed=config.seed + 1,
        download=config.download,
    )
    return DatasetBundle(
        observational_train=DataLoader(observational_train, shuffle=True, **loader_kwargs),
        intervened_train=DataLoader(intervened_train, shuffle=True, **loader_kwargs),
        observational_eval=DataLoader(observational_eval, shuffle=False, **loader_kwargs),
        intervened_eval=DataLoader(intervened_eval, shuffle=False, **loader_kwargs),
    )
