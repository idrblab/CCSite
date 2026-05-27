#!/usr/bin/env python3
"""Inference script for covalent cysteine site prediction.

Loads a trained checkpoint, runs inference on the configured test split,
then persists ground-truth labels and prediction scores for downstream analysis.
"""

import argparse
import json
from pathlib import Path
from typing import Tuple, Optional

import warnings

import numpy as np
import pandas as pd
import torch
from time import perf_counter

from peft import set_peft_model_state_dict

warnings.filterwarnings(
    "ignore",
    message=r"You are using `torch\.load` with `weights_only=False`",
    category=FutureWarning,
)

from src.utils.builder import load_config, create_model, build_embedding_model
from src.models.model import Tester
from src.data.data_generator import dataSet, collate_fn
from src.utils.metrics import metrics as compute_metrics


def parse_args() -> argparse.Namespace:
    """Parse command line arguments for inference."""
    parser = argparse.ArgumentParser(description="Run inference on the test set")
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to configuration file used during training",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to the model checkpoint (.pth).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to the output CSV file for predictions. Defaults to <config.paths.output_dir>/predictions.csv.",
    )
    parser.add_argument(
        "--test-fasta",
        type=str,
        required=True,
        help="FASTA file containing test sequences and residue labels.",
    )
    parser.add_argument(
        "--metrics",
        type=str,
        default=None,
        help="Optional path to save evaluation metrics (JSON). Defaults to <output>_metrics.json.",
    )
    return parser.parse_args()


def _prepare_paths(checkpoint_arg: str, output_arg: str, metrics_arg: Optional[str]) -> Tuple[Path, Path, Path]:
    """Resolve checkpoint, output, and metrics paths, creating directories when needed."""
    checkpoint_path = Path(checkpoint_arg)
    output_path = Path(output_arg)

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    if metrics_arg:
        metrics_path = Path(metrics_arg)
    else:
        metrics_path = output_path.with_name(f"{output_path.stem}_metrics.json")

    metrics_path.parent.mkdir(parents=True, exist_ok=True)

    return checkpoint_path, output_path, metrics_path
    

def _build_test_loader(config: dict, fasta_path: str):
    """Construct a DataLoader for the test FASTA file."""

    dataset = dataSet(config["data"]["window_size"], fasta_path)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=16,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn,
        drop_last=False,
    )
    return dataset, loader

def run_inference():
    """Entry point for running inference."""
    args = parse_args()
    config = load_config(args.config)

    # Derive default output path from config if not provided on the command line.
    if args.output is None:
        output_dir = config.get("paths", {}).get("output_dir", ".")
        args.output = str(Path(output_dir) / "predictions.csv")

    checkpoint_path, output_path, metrics_path = _prepare_paths(args.checkpoint, args.output, args.metrics)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running inference on device: {device}")

    _test_dataset, test_loader = _build_test_loader(config, args.test_fasta)

    lora_cfg = config.get("lora", {}) or {}
    lora_enabled = lora_cfg.get("enable", True)

    esmc_model = build_embedding_model(config, device)
    esmc_model.eval()

    model = create_model(config, device)
    state_dict = torch.load(str(checkpoint_path), map_location=device, weights_only=False)
    predictor_state = state_dict.get("predictor", state_dict)
    model.load_state_dict(predictor_state, strict=False)
    lora_state = state_dict.get("embedding")
    if lora_state is not None:
        if lora_enabled and hasattr(esmc_model, "peft_config"):
            try:
                set_peft_model_state_dict(esmc_model, lora_state)
            except (ValueError, AttributeError):
                esmc_model.load_state_dict(lora_state, strict=False)
        else:
            esmc_model.load_state_dict(lora_state, strict=False)

    tester = Tester(model, esmc_model, config["data"]["window_size"])
    infer_start = perf_counter()
    correct_labels_test, predicted_labels_test, predicted_scores_test, metadata = tester.test(
        test_loader, device
    )
    infer_duration = perf_counter() - infer_start

    correct_labels = np.asarray(correct_labels_test)
    predicted_labels = np.asarray(predicted_labels_test)
    predicted_scores = np.asarray(predicted_scores_test)

    names = [entry.get("name", "") for entry in metadata]
    positions = [int(entry.get("position", idx)) for idx, entry in enumerate(metadata)]
    positions_1indexed = [pos + 1 for pos in positions]

    if len(names) != len(correct_labels):
        print(
            f"[Warning] Metadata length ({len(names)}) does not match predictions ({len(correct_labels)}); aligning by truncation."
        )

    rows = min(len(correct_labels), len(names))
    df = pd.DataFrame(
        {
            "name": names[:rows],
            "position": [int(pos) for pos in positions_1indexed[:rows]],
            "label": correct_labels[:rows].astype(int),
            "score": predicted_scores[:rows].astype(float),
        }
    )
    df.to_csv(output_path, index=False)
    print(f"Saved inference outputs to {output_path}")

    if rows == 0:
        print("No samples available for metric computation; skipping metrics output.")
    else:
        acc, auc, recall, precision, f1, mcc, prc_auc = compute_metrics(
            correct_labels[:rows], predicted_labels[:rows], predicted_scores[:rows]
        )

        metrics_summary = {
            "accuracy": float(acc),
            "auc": float(auc),
            "recall": float(recall),
            "precision": float(precision),
            "f1": float(f1),
            "mcc": float(mcc),
            "prc_auc": float(prc_auc),
        }
        with open(metrics_path, "w", encoding="utf-8") as fp:
            json.dump(metrics_summary, fp, indent=2)

        print("Evaluation metrics:")
        for key, value in metrics_summary.items():
            print(f"  {key}: {value:.4f}")
        print(f"Metrics saved to {metrics_path}")

    print(f"Inference finished in {infer_duration:.2f} seconds")


if __name__ == "__main__":
    run_inference()
