from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Iterable

DEFAULT_LAMBDA_SWEEP = (0.0, 0.1, 0.5, 1.0)


def parse_float_list(raw: str | Iterable[float] | None) -> tuple[float, ...]:
    if raw is None:
        return DEFAULT_LAMBDA_SWEEP
    if isinstance(raw, str):
        items = [item.strip() for item in raw.split(",") if item.strip()]
        if not items:
            raise ValueError("Expected at least one float value.")
        return tuple(float(item) for item in items)
    values = tuple(float(item) for item in raw)
    if not values:
        raise ValueError("Expected at least one float value.")
    return values


def _validate_probability(name: str, value: float) -> None:
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be in [0, 1], got {value}.")


def _validate_positive_int(name: str, value: int) -> None:
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value}.")


def _validate_non_negative(name: str, value: float) -> None:
    if value < 0:
        raise ValueError(f"{name} must be non-negative, got {value}.")


@dataclass(slots=True)
class DataConfig:
    root: str = "data"
    train_size: int = 20_000
    eval_size: int = 5_000
    batch_size: int = 128
    num_workers: int = 2
    observational_correlation: float = 0.9
    seed: int = 42
    download: bool = False

    def __post_init__(self) -> None:
        _validate_positive_int("train_size", self.train_size)
        _validate_positive_int("eval_size", self.eval_size)
        _validate_positive_int("batch_size", self.batch_size)
        if self.num_workers < 0:
            raise ValueError("num_workers must be non-negative.")
        _validate_probability("observational_correlation", self.observational_correlation)


@dataclass(slots=True)
class TrainConfig:
    epochs: int = 5
    lr: float = 1e-3
    weight_decay: float = 1e-4
    device: str = "auto"

    def __post_init__(self) -> None:
        _validate_positive_int("epochs", self.epochs)
        _validate_non_negative("lr", self.lr)
        _validate_non_negative("weight_decay", self.weight_decay)


@dataclass(slots=True)
class UnlearningConfig(TrainConfig):
    lambda_ce: float = 1.0
    lambda_locality: float = 1e-3

    def __post_init__(self) -> None:
        # Explicitly call the base validator for compatibility with slotted
        # dataclasses on newer Python versions.
        TrainConfig.__post_init__(self)
        _validate_non_negative("lambda_ce", self.lambda_ce)
        _validate_non_negative("lambda_locality", self.lambda_locality)


@dataclass(slots=True)
class RunConfig:
    output_dir: str = "artifacts/default"
    data: DataConfig = field(default_factory=DataConfig)
    baseline: TrainConfig = field(default_factory=TrainConfig)
    oracle: TrainConfig = field(default_factory=TrainConfig)
    unlearning: UnlearningConfig = field(default_factory=lambda: UnlearningConfig(epochs=2, lr=5e-4, weight_decay=0.0))
    lambda_ce_values: tuple[float, ...] = DEFAULT_LAMBDA_SWEEP

    def __post_init__(self) -> None:
        if not self.lambda_ce_values:
            raise ValueError("lambda_ce_values cannot be empty.")
        for value in self.lambda_ce_values:
            _validate_non_negative("lambda_ce_values", float(value))

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["lambda_ce_values"] = list(self.lambda_ce_values)
        return payload
