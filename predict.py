#!/usr/bin/env python3
"""Lightweight prediction script for unlabeled FASTA inputs.

This utility mirrors the evaluate.py workflow but assumes the FASTA
contains only sequence headers and raw amino-acid sequences (no residue labels).
It produces a CSV with model scores for each cysteine residue detected in the
input proteins.
"""

import argparse
from pathlib import Path
from time import perf_counter
from typing import Optional, Tuple

import warnings

import numpy as np
import pandas as pd
import torch
from peft import set_peft_model_state_dict

warnings.filterwarnings(
    "ignore",
    message=r"You are using `torch\.load` with `weights_only=False`",
    category=FutureWarning,
)

from src.utils.builder import load_config, create_model, build_embedding_model
from src.models.model import Tester
from src.data.data_generator import dataSet, collate_fn


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for cysteine-site prediction."""
    parser = argparse.ArgumentParser(description="Run inference on unlabeled FASTA sequences")
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to the configuration file used during training.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to the trained model checkpoint (.pth).",
    )
    parser.add_argument(
        "--fasta",
        type=str,
        required=True,
        help="FASTA file where each protein has only a header and a sequence line.",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Destination CSV file for prediction results (no ground-truth labels).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for inference dataloader (default: 32).",
    )
    return parser.parse_args()


def _prepare_paths(checkpoint_arg: str, fasta_arg: str, output_arg: str) -> Tuple[Path, Path, Path]:
    """Validate and materialise filesystem paths."""
    checkpoint_path = Path(checkpoint_arg).expanduser().resolve()
    fasta_path = Path(fasta_arg).expanduser().resolve()
    output_path = Path(output_arg).expanduser().resolve()

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")
    if not fasta_path.exists():
        raise FileNotFoundError(f"FASTA file not found at {fasta_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    return checkpoint_path, fasta_path, output_path


def _build_loader(config: dict, fasta_path: Path, batch_size: int):
    """Construct a DataLoader that treats the FASTA as unlabeled input."""
    dataset = dataSet(config["data"]["window_size"], str(fasta_path))
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=config["data"].get("num_workers", 0),
        collate_fn=collate_fn,
        drop_last=False,
    )
    return dataset, loader


def main() -> None:
    args = parse_args()
    config = load_config(args.config)

    checkpoint_path, fasta_path, output_path = _prepare_paths(args.checkpoint, args.fasta, args.output)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running predictions on device: {device}")

    _dataset, loader = _build_loader(config, fasta_path, args.batch_size)

    esmc_model = build_embedding_model(config, device)
    esmc_model.eval()

    lora_cfg = config.get("lora", {}) or {}
    lora_enabled = bool(lora_cfg.get("enable", True))

    predictor = create_model(config, device)
    checkpoint = torch.load(str(checkpoint_path), map_location=device, weights_only=False)
    predictor_state = checkpoint.get("predictor", checkpoint)
    predictor.load_state_dict(predictor_state, strict=False)

    embedding_state = checkpoint.get("embedding")
    if embedding_state is not None:
        if lora_enabled and hasattr(esmc_model, "peft_config"):
            try:
                set_peft_model_state_dict(esmc_model, embedding_state)
            except (ValueError, AttributeError):
                esmc_model.load_state_dict(embedding_state, strict=False)
        else:
            esmc_model.load_state_dict(embedding_state, strict=False)

    tester = Tester(predictor, esmc_model, config["data"]["window_size"])

    infer_start = perf_counter()
    _, predicted_labels, predicted_scores, metadata = tester.test(loader, device)
    infer_time = perf_counter() - infer_start

    names = [entry.get("name", "") for entry in metadata]
    positions = [int(entry.get("position", idx)) for idx, entry in enumerate(metadata)]
    positions_1indexed = [pos + 1 for pos in positions]

    rows = min(len(names), len(predicted_scores))
    df = pd.DataFrame(
        {
            "name": names[:rows],
            "position": [int(pos) for pos in positions_1indexed[:rows]],
            "score": np.asarray(predicted_scores[:rows], dtype=float),
            "prediction": np.asarray(predicted_labels[:rows], dtype=int),
        }
    )
    df.to_csv(output_path, index=False)
    print(f"Saved predictions to {output_path}")

    print(f"Inference completed in {infer_time:.2f} seconds")


if __name__ == "__main__":
    main()
