# CCSite

A deep learning model for predicting covalent cysteine sites in proteins. CCSite fine-tunes ESM Cambrian via LoRA and combines it with a cysteine-centered local-global attention network to classify each cysteine residue as covalently reactive or not.

## Model Architecture

- **Adapted Sequence Embedding**: ESM C-600M (frozen) with LoRA adapters on the last 6 attention blocks
- **Contextual Pattern Encoding**: 1D convolutional encoder with GLU activation and residual connections
- **Cysteine-specific Decoding**: Transformer decoder with cross-attention between the target cysteine and the full protein
- **Covalent Ligandability Scoring**: 3-layer MLP (128 → 256 → 32 → 2) 
## Installation

```
# Python
python 3.10

# Deep learning
torch 2.4.0+cu118 
esm 3.2.1.post1
peft 0.6.0
transformers 4.48.1

# Data / scientific
numpy 1.26.4
pandas 2.3.0
scikit-learn 1.7.0

# Environment to reproduce
conda env create -f environment.yml
```

## ESM C Weights

CCSite uses [ESM C](https://github.com/Biohub/esm#esm-c-) to generate protein representations.

**With internet access** — weights are downloaded and cached automatically from HuggingFace Hub on first run. No extra setup needed.

**Without internet access (offline servers)** — download [ESMC-600M weights](https://huggingface.co/EvolutionaryScale/esmc-600m-2024-12) on a connected machine first:

Copy the downloaded directory to your server and place the weight file at: esmc/data/weights/esmc_600m_2024_12_v0.pth

Then uncomment and set `esmc_weights_dir` in `config.yaml`:

```yaml
data:
  esmc_weights_dir: "esmc"
```

## Data Format

Training data uses a 3-line FASTA format:

```
>protein_name
SEQUENCE
LABEL_STRING
```

Where `LABEL_STRING` is a binary string of the same length as the sequence — `1` marks a covalently reactive cysteine, `0` marks all other residues. For example:

```
>P12345
MACDEFCGHIK
00000010000
```

For prediction on unlabeled sequences, only the header and sequence lines are required (no label string).

## Project Structure

```
CCSite/
├── train.py          # Training script
├── evaluate.py       # Evaluation on labeled test set
├── predict.py        # Prediction on unlabeled sequences
├── config.yaml       # Configuration file
├── environment.yml   # Conda environment
├── dataset/
│   ├── all.fasta     # Full dataset (all proteins)
│   ├── random_split/ # Random train/valid/test split
│   └── five-fold_cross-validation_split/  # 5-fold CV splits (5 repeats)
├── example/          # Example input/output
├── ckpt/             # Model checkpoints
├── results/          # Training metrics logs
├── outputs/          # Evaluation outputs
└── src/
    ├── models/
    │   ├── model.py      # Model architecture (Encoder, Decoder, Predictor, Trainer, Tester)
    │   ├── radam.py      # RAdam optimizer
    │   └── lookahead.py  # Lookahead optimizer wrapper
    ├── data/
    │   └── data_generator.py  # Dataset and DataLoader
    └── utils/
        ├── builder.py    # Factory functions (load_config, build_embedding_model, create_model)
        ├── helpers.py    # Embedding preparation utilities
        └── metrics.py    # Evaluation metrics (AUROC, AUPRC, MCC, etc.)
```

## Usage

### Prediction on unlabeled sequences

```bash
python predict.py \
    --checkpoint ckpt/CCSite_model.pth \
    --fasta example/5QIO_A.fasta \
    --output example/5QIO_A_predictions.csv
```

Output CSV contains columns `name`, `position` (1-indexed residue position of the cysteine in sequence), `score`, `prediction`.

Optional arguments:
- `--config <path>` — path to config file (default: `config.yaml`)
- `--batch-size <int>` — batch size for inference (default: 32)

### Training

```bash
python train.py --config config.yaml
```

Training saves per-epoch checkpoints to `ckpt/` and the best model (by validation AUPRC) to `ckpt/test-best.pth`. Metrics are logged to `results/output-test.txt`.

> **Note:** Training results may vary across GPU architectures due to non-deterministic floating-point behavior. The provided pre-trained checkpoint and reported results were obtained on NVIDIA H800.

### Evaluation on a labeled test set

```bash
python evaluate.py \
    --checkpoint ckpt/CCSite_model.pth \
    --test-fasta dataset/random_split/test.fasta
```

Output is saved to `outputs/predictions.csv` (default from `config.yaml`) with columns `name`, `position`, `label`, `score`. Evaluation metrics (accuracy, AUROC, AUPRC, MCC, etc.) are saved to `outputs/predictions_metrics.json`.

Optional arguments:
- `--config <path>` — path to config file (default: `config.yaml`)
- `--output <path>` — override the output CSV path
- `--metrics <path>` — override the metrics JSON path
