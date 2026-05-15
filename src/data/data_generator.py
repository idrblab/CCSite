"""Data generator for cysteine-reactivity prediction.

Loads raw FASTA files and yields per-residue samples that are
encoded on-the-fly with ESMC during training/testing. Only cysteine residues are
included as individual samples to match the binary classification task.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import torch
from torch.utils import data


@dataclass
class SampleRecord:
    protein_idx: int
    position: int
    label: int
    name: str


class dataSet(data.Dataset):
    """Dataset that provides per-cysteine samples from FASTA files."""

    def __init__(self, window_size: int, fasta_file: str):
        super().__init__()
        self.window_size = int(window_size)
        self.fasta_file = fasta_file

        self.proteins: List[Dict[str, object]] = []
        self.samples: List[SampleRecord] = []
        self.sample_labels: List[int] = []

        self._load_fasta(fasta_file)
        if not self.samples:
            raise ValueError("Dataset contains no cysteine samples.")

        print(
            f"Loaded {len(self.samples)} cysteine samples from {len(self.proteins)} proteins"
        )

    def _load_fasta(self, fasta_file: str) -> None:
        lines: List[str] = []
        with open(fasta_file, "r", encoding="utf-8") as handle:
            for raw in handle:
                line = raw.strip()
                if line:
                    lines.append(line)

        i = 0
        while i < len(lines):
            header = lines[i]
            if not header.startswith(">"):
                i += 1
                continue

            name = header[1:].strip()
            if i + 1 >= len(lines):
                break
            sequence = lines[i + 1].strip()
            labels_line = ""
            if i + 2 < len(lines) and not lines[i + 2].startswith(">"):
                labels_line = lines[i + 2].strip()
                i += 3
            else:
                i += 2

            labels: List[int] = [1 if ch == "1" else 0 for ch in labels_line]
            if len(labels) < len(sequence):
                labels.extend([0] * (len(sequence) - len(labels)))

            protein_idx = len(self.proteins)
            self.proteins.append(
                {
                    "name": name,
                    "sequence": sequence,
                    "labels": labels,
                }
            )

            for pos, residue in enumerate(sequence):
                if residue != "C":
                    continue
                label = labels[pos] if pos < len(labels) else 0
                record = SampleRecord(
                    protein_idx=protein_idx,
                    position=pos,
                    label=int(label),
                    name=name,
                )
                self.samples.append(record)
                self.sample_labels.append(int(label))

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.samples)

    def __getitem__(self, index: int) -> Dict[str, object]:  # type: ignore[override]
        record = self.samples[index]
        protein = self.proteins[record.protein_idx]
        return {
            "sequence": protein["sequence"],
            "position": record.position,
            "label": record.label,
            "protein_idx": record.protein_idx,
            "name": record.name,
        }

    def get_metadata(self) -> List[SampleRecord]:
        """Return a copy of per-sample metadata in dataset order."""
        return list(self.samples)


def collate_fn(batch: List[Dict[str, object]]):
    """Custom collate function that preserves raw sequences for online encoding."""

    sequences = [item["sequence"] for item in batch]
    positions = torch.tensor([int(item["position"]) for item in batch], dtype=torch.long)
    labels = torch.tensor([int(item["label"]) for item in batch], dtype=torch.long)
    protein_indices = torch.tensor([int(item["protein_idx"]) for item in batch], dtype=torch.long)
    names = [str(item["name"]) for item in batch]

    return {
        "sequences": sequences,
        "positions": positions,
        "labels": labels,
        "protein_indices": protein_indices,
        "names": names,
    }
