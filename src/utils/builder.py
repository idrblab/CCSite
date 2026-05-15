"""
Factory functions for loading config, building the embedding model, and
instantiating the Predictor. Shared by train.py, evaluate.py, and predict.py.
"""

import os
import yaml
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from peft import LoraConfig, get_peft_model

from esm.models.esmc import ESMC

from src.models.model import (
    Predictor, Encoder, Decoder, DecoderLayer,
    SelfAttention, PositionwiseFeedforward,
)


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from a YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def build_embedding_model(config: Dict[str, Any], device: torch.device) -> torch.nn.Module:
    """Load ESMC and optionally wrap it with LoRA adapters.

    If ``config['data']['esmc_weights_dir']`` is set, weights are loaded from
    that local directory (offline / air-gapped servers).  The ESM library
    expects the weight file at a fixed relative path under that directory::

        <esmc_weights_dir>/data/weights/esmc_600m_2024_12_v0.pth

    Example: if ``esmc_weights_dir`` is ``"esmc"``, the file must be at
    ``esmc/data/weights/esmc_600m_2024_12_v0.pth``.

    If the key is absent or ``None``, weights are downloaded from HuggingFace
    Hub (requires network access).
    """
    esmc_model_name = config['data'].get('esmc_model')
    esmc_weights_dir = config['data'].get('esmc_weights_dir')
    print(f"Loading ESMC model: {esmc_model_name}")

    if esmc_weights_dir:
        os.environ["INFRA_PROVIDER"] = "local"
        prev_cwd = os.getcwd()
        os.chdir(esmc_weights_dir)
        try:
            base_model = ESMC.from_pretrained(esmc_model_name, device=device)
        finally:
            os.chdir(prev_cwd)
            del os.environ["INFRA_PROVIDER"]
    else:
        # Download from HuggingFace Hub (requires network access).
        base_model = ESMC.from_pretrained(esmc_model_name, device=device)

    lora_cfg = config.get('lora', {}) or {}
    lora_enabled = lora_cfg.get('enable', True)

    for param in base_model.parameters():
        param.requires_grad = False

    if lora_enabled:
        base_targets = lora_cfg.get('target_modules', [
            "layernorm_qkv.1",
            "out_proj",
        ])
        target_blocks = lora_cfg.get('target_blocks')
        if target_blocks:
            resolved_blocks: List[int] = []
            seen = set()
            for entry in target_blocks if isinstance(target_blocks, list) else [target_blocks]:
                if isinstance(entry, dict):
                    start = int(entry['start'])
                    end = int(entry.get('end', start))
                    for idx in range(start, end + 1):
                        if idx not in seen:
                            resolved_blocks.append(idx)
                            seen.add(idx)
                else:
                    idx = int(entry)
                    if idx not in seen:
                        resolved_blocks.append(idx)
                        seen.add(idx)
            target_blocks = resolved_blocks

        if target_blocks:
            expanded_targets: List[str] = []
            for block_idx in target_blocks:
                for name in base_targets:
                    expanded_targets.append(f"transformer.blocks.{block_idx}.attn.{name}")
            target_modules = expanded_targets
        else:
            target_modules = base_targets

        lora_config = LoraConfig(
            r=lora_cfg.get('rank', 8),
            lora_alpha=lora_cfg.get('alpha', 16),
            lora_dropout=lora_cfg.get('dropout', 0.25),
            bias=lora_cfg.get('bias', 'none'),
            target_modules=target_modules,
        )
        lora_model = get_peft_model(base_model, lora_config)
        lora_model = lora_model.to(device)

        if hasattr(base_model, "_tokenize"):
            setattr(lora_model, "_tokenize", base_model._tokenize)
        if hasattr(base_model, "tokenizer"):
            lora_model.tokenizer = base_model.tokenizer

        return lora_model

    base_model = base_model.to(device)
    return base_model


def create_model(
    config: Dict[str, Any],
    device: torch.device,
    focal_alpha: Optional[float] = None,
) -> Predictor:
    """Instantiate the covalent cysteine-site Predictor from config."""
    print("Creating model...")

    protein_dim = config['model']['protein_dim']
    local_dim = config['model']['local_dim']
    hid_dim = config['model']['hidden_dim']
    n_layers = config['model']['n_layers']
    n_heads = config['model']['n_heads']
    pf_dim = config['model']['pf_dim']
    dropout = config['model']['dropout']
    kernel_size = config['model']['kernel_size']

    if focal_alpha is None:
        focal_alpha = config.get('training', {}).get('focal_alpha', 0.5)

    encoder = Encoder(protein_dim, hid_dim, n_layers, kernel_size, dropout, device)
    decoder = Decoder(local_dim, hid_dim, n_layers, n_heads, pf_dim, DecoderLayer,
                      SelfAttention, PositionwiseFeedforward, dropout, device)

    model = Predictor(encoder, decoder, device, focal_alpha)
    model.to(device)
    return model
