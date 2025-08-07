from __future__ import annotations

import json
import os
import random
from typing import Any

import numpy as np
import polars as pl
import torch


def set_seed(seed: int = 42):
    """Set random seeds for reproducibility.

    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_sequence_statistics(sequences: list) -> dict[str, Any]:
    """Get statistics about sequence lengths and data.

    Args:
        sequences: List of sequence dictionaries

    Returns:
        Dictionary with sequence statistics
    """
    if "enhanced_data" in sequences[0]:
        lengths = [seq["enhanced_data"]["sequence_length"] for seq in sequences]
    else:
        lengths = [len(seq["data"]) for seq in sequences]

    return {
        "total_sequences": len(sequences),
        "min_length": min(lengths),
        "max_length": max(lengths),
        "mean_length": np.mean(lengths),
        "median_length": np.median(lengths),
        "std_length": np.std(lengths),
        "percentile_95": np.percentile(lengths, 95),
        "percentile_99": np.percentile(lengths, 99),
    }


def print_data_info(train_df: pl.DataFrame, demographics_df: pl.DataFrame = None):
    """Print information about the datasets.

    Args:
        train_df: Training dataframe
        demographics_df: Demographics dataframe (optional)
    """
    print("=== Data Information ===")
    print(f"Training data shape: {train_df.shape}")
    print(f"Columns in training data: {len(train_df.columns)}")
    print(f"Unique sequences: {train_df['sequence_id'].n_unique()}")
    print(f"Unique subjects: {train_df['subject'].n_unique()}")
    print(f"Unique gestures: {train_df['gesture'].n_unique()}")

    if demographics_df is not None:
        print(f"Demographics data shape: {demographics_df.shape}")
        print(f"Demographics columns: {demographics_df.columns}")

    print("\nGesture distribution:")
    gesture_counts = (
        train_df.group_by("gesture")
        .agg(
            pl.count().alias("count"),
        )
        .sort("count", descending=True)
    )
    print(gesture_counts)


def print_model_info(model: torch.nn.Module):
    """Print model information.

    Args:
        model: PyTorch model
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("=== Model Information ===")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Model size (MB): {total_params * 4 / 1024**2:.2f}")


def save_config(config: dict[str, Any], save_path: str):
    """Save configuration to JSON file.

    Args:
        config: Configuration dictionary
        save_path: Path to save the config
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    with open(save_path, "w") as f:
        json.dump(config, f, indent=2)

    print(f"Configuration saved to: {save_path}")


def load_config(config_path: str) -> dict[str, Any]:
    """Load configuration from JSON file.

    Args:
        config_path: Path to config file

    Returns:
        Configuration dictionary
    """
    with open(config_path) as f:
        config = json.load(f)

    print(f"Configuration loaded from: {config_path}")
    return config


def check_gpu_usage():
    """Print GPU usage information if available."""
    if torch.cuda.is_available():
        print("=== GPU Information ===")
        print(f"GPU available: {torch.cuda.is_available()}")
        print(f"GPU count: {torch.cuda.device_count()}")
        print(f"Current GPU: {torch.cuda.current_device()}")
        print(f"GPU name: {torch.cuda.get_device_name()}")

        # Memory usage
        allocated = torch.cuda.memory_allocated() / 1024**3
        cached = torch.cuda.memory_reserved() / 1024**3
        print(f"GPU memory allocated: {allocated:.2f} GB")
        print(f"GPU memory cached: {cached:.2f} GB")
    else:
        print("GPU not available, using CPU")


def create_experiment_dir(base_dir: str, experiment_name: str) -> str:
    """Create experiment directory with timestamp.

    Args:
        base_dir: Base directory for experiments
        experiment_name: Name of the experiment

    Returns:
        Path to created experiment directory
    """
    from datetime import datetime

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = os.path.join(base_dir, f"{experiment_name}_{timestamp}")

    os.makedirs(exp_dir, exist_ok=True)
    os.makedirs(os.path.join(exp_dir, "models"), exist_ok=True)
    os.makedirs(os.path.join(exp_dir, "plots"), exist_ok=True)
    os.makedirs(os.path.join(exp_dir, "logs"), exist_ok=True)

    print(f"Experiment directory created: {exp_dir}")
    return exp_dir


def log_experiment_results(results: dict[str, Any], log_path: str):
    """Log experiment results to file.

    Args:
        results: Results dictionary
        log_path: Path to log file
    """
    from datetime import datetime

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    os.makedirs(os.path.dirname(log_path), exist_ok=True)

    with open(log_path, "a") as f:
        f.write(f"\n=== Experiment Results - {timestamp} ===\n")
        for key, value in results.items():
            if isinstance(value, (list, dict)):
                f.write(f"{key}: {json.dumps(value, indent=2)}\n")
            else:
                f.write(f"{key}: {value}\n")
        f.write("=" * 50 + "\n")

    print(f"Results logged to: {log_path}")


def calculate_optimal_max_length(
    sequence_lengths: list,
    percentile: float = 95.0,
) -> int:
    """Calculate optimal maximum sequence length based on percentile.

    Args:
        sequence_lengths: List of sequence lengths
        percentile: Percentile to use for max length

    Returns:
        Optimal maximum length
    """
    optimal_length = int(np.percentile(sequence_lengths, percentile))

    coverage = sum(1 for length in sequence_lengths if length <= optimal_length) / len(
        sequence_lengths,
    )

    print(f"Optimal max length: {optimal_length} ({percentile}th percentile)")
    print(f"Coverage: {coverage*100:.1f}% of sequences")
    print(
        f"Sequences requiring truncation: {sum(1 for length in sequence_lengths if length > optimal_length)}",
    )
    print(
        f"Sequences requiring padding: {sum(1 for length in sequence_lengths if length < optimal_length)}",
    )

    return optimal_length


class EarlyStopping:
    """Early stopping utility class."""

    def __init__(self, patience: int = 7, min_delta: float = 0.0, mode: str = "max"):
        """Initialize early stopping.

        Args:
            patience: Number of epochs to wait before stopping
            min_delta: Minimum change to qualify as improvement
            mode: 'max' for metrics to maximize, 'min' for metrics to minimize
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, score: float) -> bool:
        """Check if should stop early.

        Args:
            score: Current metric value

        Returns:
            True if should stop early
        """
        if self.best_score is None:
            self.best_score = score
        elif self.mode == "max":
            if score > self.best_score + self.min_delta:
                self.best_score = score
                self.counter = 0
            else:
                self.counter += 1
        elif score < self.best_score - self.min_delta:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1

        if self.counter >= self.patience:
            self.early_stop = True

        return self.early_stop


def format_time(seconds: float) -> str:
    """Format seconds into readable time string.

    Args:
        seconds: Time in seconds

    Returns:
        Formatted time string
    """
    if seconds < 60:
        return f"{seconds:.1f}s"
    if seconds < 3600:
        minutes = seconds // 60
        seconds = seconds % 60
        return f"{minutes:.0f}m {seconds:.1f}s"
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    seconds = seconds % 60
    return f"{hours:.0f}h {minutes:.0f}m {seconds:.1f}s"
