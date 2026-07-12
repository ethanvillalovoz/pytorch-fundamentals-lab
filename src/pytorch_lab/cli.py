"""Command-line entry point for reproducible lab experiments."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from pytorch_lab.iris import IrisConfig, train_iris, write_iris_artifacts
from pytorch_lab.mnist import MnistConfig, train_mnist


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="torch-lab", description="Run the tested lab examples.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    iris = subparsers.add_parser("iris", help="train the deterministic Iris MLP")
    iris.add_argument("--seed", type=int, default=42)
    iris.add_argument("--epochs", type=int, default=250)
    iris.add_argument("--learning-rate", type=float, default=0.01)
    iris.add_argument("--output-dir", type=Path, default=Path("artifacts/iris"))
    iris.add_argument("--checkpoint", action="store_true")

    mnist = subparsers.add_parser("mnist", help="train the MNIST CNN")
    mnist.add_argument("--seed", type=int, default=42)
    mnist.add_argument("--epochs", type=int, default=3)
    mnist.add_argument("--learning-rate", type=float, default=0.001)
    mnist.add_argument("--batch-size", type=int, default=128)
    mnist.add_argument("--device", choices=("cpu", "cuda", "mps"))
    mnist.add_argument("--data-dir", type=Path, default=Path("data"))
    mnist.add_argument("--no-download", action="store_true")
    mnist.add_argument("--max-train-batches", type=int)
    mnist.add_argument("--max-test-batches", type=int)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "iris":
        config = IrisConfig(
            seed=args.seed,
            epochs=args.epochs,
            learning_rate=args.learning_rate,
        )
        run = train_iris(config)
        metrics_path, predictions_path = write_iris_artifacts(
            run,
            args.output_dir,
            include_checkpoint=args.checkpoint,
        )
        print(
            json.dumps(
                {
                    "test_accuracy": run.test_accuracy,
                    "test_loss": run.test_loss,
                    "metrics": str(metrics_path),
                    "predictions": str(predictions_path),
                },
                indent=2,
            )
        )
        return 0

    config = MnistConfig(
        seed=args.seed,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        device=args.device,
    )
    run = train_mnist(
        config,
        data_dir=args.data_dir,
        download=not args.no_download,
        max_train_batches=args.max_train_batches,
        max_test_batches=args.max_test_batches,
    )
    print(
        json.dumps(
            [
                {
                    "epoch": item.epoch,
                    "train": asdict(item.train),
                    "test": asdict(item.test),
                }
                for item in run.history
            ],
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
