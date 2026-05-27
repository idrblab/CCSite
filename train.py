#!/usr/bin/env python3
"""
Training script for covalent cysteine site prediction.

This script handles:
- Data loading and preprocessing
- Model training with validation
- Checkpointing and early stopping
- Comprehensive logging and metrics
"""

import argparse
import warnings
import timeit
from pathlib import Path
from typing import Any, Tuple, List

import torch
import numpy as np

from torch.utils.data import Subset
from peft import set_peft_model_state_dict

from src.models.model import Trainer, Tester
from src.data.data_generator import dataSet, collate_fn
from src.utils.helpers import init_seeds
from src.utils.metrics import metrics
from src.utils.builder import load_config, build_embedding_model, create_model


# Suppress FutureWarning from torch.load(weights_only=False) in dependencies.
warnings.filterwarnings(
    "ignore",
    message=r"You are using `torch\.load` with `weights_only=False`",
    category=FutureWarning,
)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train covalent cysteine-site predictor")
    parser.add_argument("--config", type=str, default="config.yaml", 
                       help="Path to configuration file")
    parser.add_argument("--resume", type=str, default=None,
                       help="Path to checkpoint to resume from")
    parser.add_argument("--debug", action="store_true",
                       help="Enable debug mode with smaller dataset")
    return parser.parse_args()


def load_data(config: dict, debug: bool = False) -> Tuple[dataSet, dataSet]:
    """Load training and validation datasets."""
    print("Loading datasets...")

    window_size = config['data']['window_size']
    train_fasta = config['data']['train_fasta']
    valid_fasta = config['data']['valid_fasta']

    train_dataset = dataSet(window_size, train_fasta)
    valid_dataset = dataSet(window_size, valid_fasta)

    if debug:
        max_train = min(len(train_dataset), 512)
        max_valid = min(len(valid_dataset), 256)
        train_dataset = Subset(train_dataset, list(range(max_train)))
        valid_dataset = Subset(valid_dataset, list(range(max_valid)))
        print(f"Debug mode: using {max_train} train and {max_valid} valid samples")

    print(
        "Loaded datasets - Train: {0}, Valid: {1}".format(
            len(train_dataset), len(valid_dataset)
        )
    )

    return train_dataset, valid_dataset


def compute_loss_weights(dataset: dataSet) -> Tuple[List[float], int, int]:
    """Compute class-balanced loss weights from dataset."""
    if isinstance(dataset, Subset):
        labels_iter = [dataset.dataset.sample_labels[idx] for idx in dataset.indices]
    else:
        labels_iter = getattr(dataset, "sample_labels", [])

    pos_count = sum(1 for label in labels_iter if int(label) == 1)
    neg_count = len(labels_iter) - pos_count
    total = pos_count + neg_count
    if total == 0 or pos_count == 0 or neg_count == 0:
        print("Warning: Unable to compute balanced loss weights; falling back to [1.0, 1.0].")
        return [1.0, 1.0], pos_count, neg_count

    pos_weight = float(neg_count) / float(pos_count)
    loss_weights = [1.0, pos_weight]
    return loss_weights, pos_count, neg_count


def create_data_loaders(train_dataset: dataSet, valid_dataset: dataSet,
                       config: dict) -> Tuple:
    """Create data loaders for training and validation splits."""

    batch_size = config['training']['batch_size']
    num_workers = config['data'].get('num_workers', 0)
    pin_memory = bool(config['data'].get('pin_memory', False))

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=pin_memory,
        collate_fn=collate_fn, drop_last=False
    )
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory,
        collate_fn=collate_fn, drop_last=False
    )
    return train_loader, valid_loader

def log_combined_parameter_counts(predictor_model: torch.nn.Module, embedding_model: torch.nn.Module) -> None:
    """Log parameter counts for Predictor and LoRA (trainable) separately."""

    predictor_trainable = sum(p.numel() for p in predictor_model.parameters() if p.requires_grad)
    predictor_total = sum(p.numel() for p in predictor_model.parameters())

    lora_trainable = sum(p.numel() for p in embedding_model.parameters() if p.requires_grad)
    esmc_total = sum(p.numel() for p in embedding_model.parameters())

    print(
        f"Predictor params: total={predictor_total:,}, trainable={predictor_trainable:,} | "
        f"ESMC params: total={esmc_total:,}, LoRA trainable={lora_trainable:,} | "
        f"Total trainable={predictor_trainable + lora_trainable:,}"
    )


def setup_output_directories(config: dict) -> Tuple[Path, Path, str]:
    """Setup output directories for models and results."""

    model_dir = Path(config['paths']['model_dir'])
    result_dir = Path(config['paths']['result_dir'])

    model_dir.mkdir(parents=True, exist_ok=True)
    result_dir.mkdir(parents=True, exist_ok=True)

    experiment_name = "test"
    file_AUCs = result_dir / f"output-{experiment_name}.txt"
    file_model = model_dir / f"{experiment_name}-best.pth"
    checkpoint_pattern = str(model_dir / f"{experiment_name}-epoch{{epoch}}.pth")

    return file_AUCs, file_model, checkpoint_pattern


def train_epoch(trainer: Trainer, train_loader, device: torch.device) -> None:
    """Train model for one epoch."""
    trainer.train(train_loader, device)


def evaluate_epoch(tester: Tester, data_loader, device: torch.device,
                  compute_loss: bool = False) -> Tuple:
    """Run evaluation on given data loader."""
    eval_outputs = tester.test(data_loader, device, return_loss=compute_loss)

    if compute_loss:
        correct_labels, predicted_labels, predicted_scores, loss_total, sample_count, _ = eval_outputs
    else:
        correct_labels, predicted_labels, predicted_scores, _ = eval_outputs
        loss_total, sample_count = None, None

    correct_labels_arr = np.array(correct_labels)
    predicted_labels_arr = np.array(predicted_labels)
    predicted_scores_arr = np.array(predicted_scores)

    if compute_loss and sample_count:
        mean_loss = float(loss_total) / float(sample_count)
    elif compute_loss:
        mean_loss = None
    else:
        mean_loss = None

    return correct_labels_arr, predicted_labels_arr, predicted_scores_arr, mean_loss


def format_epoch_console_line(epoch_metrics: List[Any]) -> str:
    """Format a clean one-line epoch summary for terminal output."""
    (
        epoch,
        train_time,
        eval_time,
        loss_train,
        loss_valid,
        acc_train,
        auc_train,
        rec_train,
        pre_train,
        f1_train,
        mcc_train,
        prc_train,
        acc_valid,
        auc_valid,
        rec_valid,
        pre_valid,
        f1_valid,
        mcc_valid,
        prc_valid,
    ) = epoch_metrics

    return (
        f"Epoch {int(epoch):03d} | "
        f"time(train={float(train_time):.1f}s, eval={float(eval_time):.1f}s) | "
        f"train: loss={float(loss_train):.4f}, ACC={float(acc_train):.4f}, AUC={float(auc_train):.4f}, "
        f"Rec={float(rec_train):.4f}, Pre={float(pre_train):.4f}, F1={float(f1_train):.4f}, "
        f"MCC={float(mcc_train):.4f}, PRC={float(prc_train):.4f} | "
        f"valid: loss={float(loss_valid):.4f}, ACC={float(acc_valid):.4f}, AUC={float(auc_valid):.4f}, "
        f"Rec={float(rec_valid):.4f}, Pre={float(pre_valid):.4f}, F1={float(f1_valid):.4f}, "
        f"MCC={float(mcc_valid):.4f}, PRC={float(prc_valid):.4f}"
    )


def main():
    """Main training function."""
    args = parse_args()
    config = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Single process training on device: {device}")
    
    # Initialize seeds for reproducibility
    init_seeds(config['training']['seed'])
    
    # Load datasets
    train_dataset, valid_dataset = load_data(config, args.debug)
    
    # Create data loaders
    train_loader, valid_loader = create_data_loaders(
        train_dataset, valid_dataset, config
    )

    # Derive class statistics from data distribution
    loss_weights, pos_count, neg_count = compute_loss_weights(train_dataset)

    if pos_count == 0 or neg_count == 0:
        focal_alpha = 0.5
    else:
        total_samples = pos_count + neg_count
        focal_alpha = neg_count / total_samples

    ratio = loss_weights[1]
    print(
        f"Class distribution -> negatives: {neg_count}, positives: {pos_count}, "
        f"loss_weights: {loss_weights} (pos_weight≈{ratio:.4f})"
    )
    print(f"Focal alpha (positive class weight) set to: {focal_alpha:.4f}")
    
    esmc_model = build_embedding_model(config, device)
    lora_cfg = config.get('lora', {}) or {}
    lora_enabled = bool(lora_cfg.get('enable', True))

    # Create model
    model = create_model(config, device, focal_alpha=focal_alpha)
    log_combined_parameter_counts(model, esmc_model)

    if args.resume:
        print(f"Resuming from checkpoint: {args.resume}")
        resume_state = torch.load(args.resume, map_location=device)
        predictor_state = resume_state.get('predictor', resume_state)
        model.load_state_dict(predictor_state, strict=False)
        lora_state = resume_state.get('embedding')
        if lora_state is not None:
            try:
                set_peft_model_state_dict(esmc_model, lora_state)
            except ValueError:
                esmc_model.load_state_dict(lora_state, strict=False)
    
    # Create trainer and tester
    window_size = config['data']['window_size']
    trainer = Trainer(
        model,
        config['training']['learning_rate'],
        config['training']['weight_decay'],
        esmc_model,
        window_size,
        embedding_lr=config['training'].get('lora_learning_rate') if lora_enabled else None,
        lora_enabled=lora_enabled,
        initialize_model=(args.resume is None),
    )
    tester = Tester(model, esmc_model, window_size)
    
    # Setup output directories and files
    file_AUCs, file_model, checkpoint_pattern = setup_output_directories(config)
    
    # Initialize metrics logging
    header = (
        'Epoch\tTime1(sec)\tTime2(sec)\tLoss_train_mean\tLoss_valid_mean\t'
        'ACC_train\tAUC_train\tRec_train\tPre_train\tF1_train\tMCC_train\tPRC_train\t'
        'ACC_valid\tAUC_valid\tRec_valid\tPre_valid\tF1_valid\tMCC_valid\tPRC_valid'
    )
    with open(file_AUCs, 'w') as f:
        f.write(header + '\n')
    print('Training started...')
    print('Terminal summary columns: epoch | time | train(all metrics) | valid(all metrics)')
    
    # Training parameters
    max_PRC_valid = 0
    last_improve = 0
    epochs = config['training']['max_epochs']
    early_stopping = config['training']['early_stopping']
    decay_interval = config['training']['decay_interval']
    lr_decay = config['training']['lr_decay']
    
    # Training loop
    for epoch in range(1, epochs + 1):
        # Learning rate decay
        if epoch % decay_interval == 0:
            for param_group in trainer.optimizer.param_groups:
                param_group['lr'] *= lr_decay
        
        # Training phase
        start_time = timeit.default_timer()
        train_epoch(trainer, train_loader, device)
        train_time = timeit.default_timer() - start_time
        
        # Training metrics evaluation
        train_eval_start = timeit.default_timer()
        train_correct, train_predicted, train_scores, loss_train = evaluate_epoch(
            tester, train_loader, device, compute_loss=True
        )
        train_eval_time = timeit.default_timer() - train_eval_start

        ACC_train, AUC_train, Rec_train, Pre_train, F1_train, MCC_train, PRC_train = metrics(
            train_correct, train_predicted, train_scores
        )

        # Validation phase
        valid_eval_start = timeit.default_timer()

        correct_labels_valid, predicted_labels_valid, predicted_scores_valid, loss_valid = evaluate_epoch(
            tester, valid_loader, device, compute_loss=True
        )
        valid_eval_time = timeit.default_timer() - valid_eval_start
        
        ACC_valid, AUC_valid, Rec_valid, Pre_valid, F1_valid, MCC_valid, PRC_valid = metrics(
            correct_labels_valid, predicted_labels_valid, predicted_scores_valid
        )

        eval_time = train_eval_time + valid_eval_time
        loss_train_value = loss_train if loss_train is not None else float('nan')
        loss_valid_value = loss_valid if loss_valid is not None else float('nan')
        
        # Prepare metrics for logging
        epoch_metrics = [
            epoch, train_time, eval_time, loss_train_value, loss_valid_value,
            ACC_train, AUC_train, Rec_train, Pre_train, F1_train, MCC_train, PRC_train,
            ACC_valid, AUC_valid, Rec_valid, Pre_valid, F1_valid, MCC_valid, PRC_valid
        ]
        
        tester.save_AUCs(epoch_metrics, file_AUCs)
        print(format_epoch_console_line(epoch_metrics))
        epoch_checkpoint_path = Path(checkpoint_pattern.format(epoch=epoch))
        tester.save_model(model, epoch_checkpoint_path)
        
        # Model checkpointing based on validation PRC
        if PRC_valid > max_PRC_valid:
            last_improve = epoch
            max_PRC_valid = PRC_valid

            print(f'Validation improved at epoch {last_improve}, saving model...')
            tester.save_model(model, file_model)
        
        
        # Early stopping check
        if epoch - last_improve >= early_stopping:
            print(f'Early stopping at epoch {epoch} (no improvement for {early_stopping} epochs)')
            break
    
    # Training completed
    print('Training completed!')
    print(f'Best validation PRC: {max_PRC_valid:.4f}')
    print(f'Model saved to: {file_model}')


if __name__ == "__main__":
    main()
