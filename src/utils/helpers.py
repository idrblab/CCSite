"""
Helper functions for training and data processing.
"""

import torch
import numpy as np
import random
from typing import List, Sequence
from torch.nn.utils.rnn import pad_sequence


def init_seeds(seed: int = 42):
    """Initialize random seeds for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def prepare_batch_embeddings(
    esmc_model: torch.nn.Module,
    sequences: Sequence[str],
    positions: torch.Tensor,
    labels: torch.Tensor,
    protein_indices: torch.Tensor,
    window_size: int,
    device: torch.device,
    requires_grad: bool,
):
    """Convert raw sequences into local/global embeddings using ESMC."""

    if len(sequences) == 0:
        raise ValueError("Empty batch provided to prepare_batch_embeddings")

    esmc_model = esmc_model.to(device)

    protein_ids = protein_indices.tolist()
    unique_sequences: List[str] = []
    unique_lookup: dict[int, int] = {}
    sample_to_unique: List[int] = []

    for seq, prot_idx in zip(sequences, protein_ids):
        mapped = unique_lookup.get(prot_idx)
        if mapped is None:
            mapped = len(unique_sequences)
            unique_lookup[prot_idx] = mapped
            unique_sequences.append(seq)
        sample_to_unique.append(mapped)

    if not unique_sequences:
        raise ValueError("Unable to map batch sequences to unique proteins")

    tokenize_fn = getattr(esmc_model, "_tokenize", None)
    if tokenize_fn is None:
        raise AttributeError("ESMC model does not expose a _tokenize method")

    context = torch.enable_grad() if requires_grad else torch.no_grad()
    # Use autocast with bfloat16 for memory efficiency;
    # embeddings are cast back to float32 after this block.
    with context, torch.autocast(device_type=device.type, dtype=torch.bfloat16):
        token_batch = tokenize_fn(unique_sequences)
        if token_batch.device != device:
            token_batch = token_batch.to(device)
        outputs = esmc_model(sequence_tokens=token_batch)

    raw_embeddings = outputs.embeddings.to(torch.float32)

    unique_embeddings: List[torch.Tensor] = []
    for idx, seq in enumerate(unique_sequences):
        seq_len = len(seq)
        embed = raw_embeddings[idx, 1 : seq_len + 1, :].contiguous()
        unique_embeddings.append(embed)

    batch_size = len(sequences)
    positions_list = positions.tolist()

    protein_lengths: List[int] = []
    local_lengths: List[int] = []
    protein_segments: List[torch.Tensor] = []
    local_segments: List[torch.Tensor] = []

    for sample_idx in range(batch_size):
        u_idx = sample_to_unique[sample_idx]
        embedding = unique_embeddings[u_idx]
        seq_len = int(embedding.size(0))
        protein_lengths.append(seq_len)

        pos = int(positions_list[sample_idx])
        if seq_len == 0:
            start = 0
            end = 1
        else:
            pos = max(0, min(pos, seq_len - 1))
            radius = max(0, int(window_size))
            start = max(0, pos - radius)
            end = min(seq_len, pos + radius + 1)
            if end <= start:
                end = start + 1
                if end > seq_len:
                    start = max(0, seq_len - 1)
                    end = seq_len

        local_len = end - start
        local_lengths.append(local_len)

        protein_segments.append(embedding[:seq_len] if seq_len > 0 else embedding.new_zeros((1, embedding.size(-1))))
        local_slice = embedding[start:end]
        if local_slice.numel() == 0:
            local_slice = embedding.new_zeros((1, embedding.size(-1)))
        local_segments.append(local_slice)

    protein_tensor = pad_sequence(protein_segments, batch_first=True)
    local_tensor = pad_sequence(local_segments, batch_first=True)

    labels_device = labels.to(device)

    return (
        local_tensor,
        protein_tensor,
        labels_device,
        [int(x) for x in local_lengths],
        [int(x) for x in protein_lengths],
    )
