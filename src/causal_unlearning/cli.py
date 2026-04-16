from __future__ import annotations

import argparse

from causal_unlearning.config import DataConfig, RunConfig, TrainConfig, UnlearningConfig, parse_float_list


def _add_shared_data_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--train-size", type=int, default=20_000)
    parser.add_argument("--eval-size", type=int, default=5_000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--observational-correlation", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--download", action="store_true")


def _build_data_config(args: argparse.Namespace) -> DataConfig:
    return DataConfig(
        root=args.data_root,
        train_size=args.train_size,
        eval_size=args.eval_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        observational_correlation=args.observational_correlation,
        seed=args.seed,
        download=args.download,
    )


def _build_train_config(args: argparse.Namespace, *, prefix: str = "") -> TrainConfig:
    return TrainConfig(
        epochs=getattr(args, f"{prefix}epochs"),
        lr=getattr(args, f"{prefix}lr"),
        weight_decay=getattr(args, f"{prefix}weight_decay"),
        device=args.device,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="causal-unlearning",
        description="Proof-of-concept implementation of causal machine unlearning on Colored MNIST.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="Run the full baseline/oracle/unlearning pipeline.")
    _add_shared_data_args(run_parser)
    run_parser.add_argument("--output-dir", default="artifacts/default")
    run_parser.add_argument("--baseline-epochs", type=int, default=5)
    run_parser.add_argument("--baseline-lr", type=float, default=1e-3)
    run_parser.add_argument("--baseline-weight-decay", type=float, default=1e-4)
    run_parser.add_argument("--oracle-epochs", type=int, default=5)
    run_parser.add_argument("--oracle-lr", type=float, default=1e-3)
    run_parser.add_argument("--oracle-weight-decay", type=float, default=1e-4)
    run_parser.add_argument("--unlearning-epochs", type=int, default=2)
    run_parser.add_argument("--unlearning-lr", type=float, default=5e-4)
    run_parser.add_argument("--unlearning-weight-decay", type=float, default=0.0)
    run_parser.add_argument("--lambda-ce-values", default="0.0,0.1,0.5,1.0")
    run_parser.add_argument("--lambda-locality", type=float, default=1e-3)
    run_parser.add_argument("--device", default="auto")
    run_parser.set_defaults(handler=_handle_run)

    train_parser = subparsers.add_parser("train", help="Train either the observational baseline or intervened oracle.")
    _add_shared_data_args(train_parser)
    train_parser.add_argument("--world", choices=["observational", "intervened"], required=True)
    train_parser.add_argument("--name", default=None)
    train_parser.add_argument("--output-dir", default="artifacts/train")
    train_parser.add_argument("--epochs", type=int, default=5)
    train_parser.add_argument("--lr", type=float, default=1e-3)
    train_parser.add_argument("--weight-decay", type=float, default=1e-4)
    train_parser.add_argument("--device", default="auto")
    train_parser.set_defaults(handler=_handle_train)

    unlearn_parser = subparsers.add_parser("unlearn", help="Fine-tune a baseline checkpoint with the proposal loss.")
    _add_shared_data_args(unlearn_parser)
    unlearn_parser.add_argument("--baseline-checkpoint", required=True)
    unlearn_parser.add_argument("--oracle-checkpoint")
    unlearn_parser.add_argument("--output-dir", default="artifacts/unlearning")
    unlearn_parser.add_argument("--name", default=None)
    unlearn_parser.add_argument("--epochs", type=int, default=2)
    unlearn_parser.add_argument("--lr", type=float, default=5e-4)
    unlearn_parser.add_argument("--weight-decay", type=float, default=0.0)
    unlearn_parser.add_argument("--lambda-ce", type=float, default=1.0)
    unlearn_parser.add_argument("--lambda-locality", type=float, default=1e-3)
    unlearn_parser.add_argument("--device", default="auto")
    unlearn_parser.set_defaults(handler=_handle_unlearn)

    evaluate_parser = subparsers.add_parser("evaluate", help="Evaluate one or more checkpoints on proposal metrics.")
    _add_shared_data_args(evaluate_parser)
    evaluate_parser.add_argument("--checkpoints", nargs="+", required=True)
    evaluate_parser.add_argument("--oracle-checkpoint")
    evaluate_parser.add_argument("--output-json")
    evaluate_parser.add_argument("--device", default="auto")
    evaluate_parser.set_defaults(handler=_handle_evaluate)
    return parser


def _handle_run(args: argparse.Namespace) -> dict:
    from causal_unlearning.experiments import run_full_pipeline

    config = RunConfig(
        output_dir=args.output_dir,
        data=_build_data_config(args),
        baseline=TrainConfig(
            epochs=args.baseline_epochs,
            lr=args.baseline_lr,
            weight_decay=args.baseline_weight_decay,
            device=args.device,
        ),
        oracle=TrainConfig(
            epochs=args.oracle_epochs,
            lr=args.oracle_lr,
            weight_decay=args.oracle_weight_decay,
            device=args.device,
        ),
        unlearning=UnlearningConfig(
            epochs=args.unlearning_epochs,
            lr=args.unlearning_lr,
            weight_decay=args.unlearning_weight_decay,
            device=args.device,
            lambda_ce=0.0,
            lambda_locality=args.lambda_locality,
        ),
        lambda_ce_values=parse_float_list(args.lambda_ce_values),
    )
    return run_full_pipeline(config)


def _handle_train(args: argparse.Namespace) -> dict:
    from causal_unlearning.experiments import train_single_model

    name = args.name or ("baseline" if args.world == "observational" else "oracle")
    return train_single_model(
        name=name,
        world=args.world,
        data_config=_build_data_config(args),
        train_config=_build_train_config(args),
        output_dir=args.output_dir,
    )


def _handle_unlearn(args: argparse.Namespace) -> dict:
    from causal_unlearning.experiments import unlearn_from_checkpoint

    name = args.name or f"unlearn_lambda_{str(args.lambda_ce).replace('.', 'p')}"
    return unlearn_from_checkpoint(
        name=name,
        baseline_checkpoint=args.baseline_checkpoint,
        oracle_checkpoint=args.oracle_checkpoint,
        data_config=_build_data_config(args),
        unlearning_config=UnlearningConfig(
            epochs=args.epochs,
            lr=args.lr,
            weight_decay=args.weight_decay,
            device=args.device,
            lambda_ce=args.lambda_ce,
            lambda_locality=args.lambda_locality,
        ),
        output_dir=args.output_dir,
    )


def _handle_evaluate(args: argparse.Namespace) -> dict:
    from causal_unlearning.experiments import evaluate_checkpoints

    return evaluate_checkpoints(
        checkpoint_paths=args.checkpoints,
        oracle_checkpoint=args.oracle_checkpoint,
        data_config=_build_data_config(args),
        device=args.device,
        output_json=args.output_json,
    )


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.handler(args)
    return 0

