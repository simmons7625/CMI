from __future__ import annotations

import os
from typing import TYPE_CHECKING

import torch
from torch import nn, optim
from tqdm import tqdm

from .focal_loss import FocalLoss, calculate_class_weights, create_focal_loss
from .model import create_model

try:
    import wandb
except ImportError:
    wandb = None

if TYPE_CHECKING:
    from torch.utils.data import DataLoader


class CMITrainer:
    """Trainer class for CMI gesture recognition model."""

    def __init__(
        self,
        model_config: dict,
        training_config: dict,
        device: str = "auto",
        use_wandb: bool = False,
        wandb_config: dict | None = None,
        train_sequences: list[dict] | None = None,
    ):
        """Initialize trainer.

        Args:
            model_config: Model configuration parameters
            training_config: Training configuration parameters
            device: Device to use ('auto', 'cuda', 'cpu')
            use_wandb: Whether to use wandb for logging
            wandb_config: wandb configuration dictionary
            train_sequences: Training sequences for focal loss class weighting
        """
        # Set device
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # Store configs
        self.model_config = model_config
        self.training_config = training_config
        self.use_wandb = use_wandb and wandb is not None

        # Initialize wandb if requested
        if self.use_wandb:
            wandb_config = wandb_config or {}
            wandb.init(
                project=wandb_config.get("project", "cmi-gesture-recognition"),
                entity=wandb_config.get("entity"),
                tags=wandb_config.get("tags", []),
                notes=wandb_config.get("notes", ""),
                config={**model_config, **training_config},
            )

        # Initialize model
        self.model = create_model(**model_config).to(self.device)

        # Initialize loss function
        loss_config = training_config.get("loss", {})
        label_smoothing = loss_config.get("label_smoothing", 0.0)

        if loss_config.get("type", "cross_entropy") == "focal_loss":
            if train_sequences is not None:
                class_counts = calculate_class_weights(
                    train_sequences,
                    model_config["num_classes"],
                )
                self.criterion = create_focal_loss(
                    num_classes=model_config["num_classes"],
                    class_counts=class_counts,
                    gamma=loss_config.get("gamma", 2.0),
                    reduction=loss_config.get("reduction", "mean"),
                    label_smoothing=label_smoothing,
                ).to(self.device)
            else:
                self.criterion = FocalLoss(
                    gamma=loss_config.get("gamma", 2.0),
                    reduction=loss_config.get("reduction", "mean"),
                    label_smoothing=label_smoothing,
                ).to(self.device)
        else:
            self.criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

        # Initialize optimizer and scheduler
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=training_config.get("learning_rate", 1e-3),
            weight_decay=training_config.get("weight_decay", 1e-4),
        )
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode="max",
            factor=training_config.get("lr_factor", 0.5),
            patience=training_config.get("lr_patience", 5),
        )

        # Training state
        self.best_val_acc = 0.0
        self.current_epoch = 0

        # History tracking
        self.train_losses = []
        self.train_accuracies = []
        self.val_losses = []
        self.val_accuracies = []

    def train_epoch(self, train_loader: DataLoader) -> tuple[float, float]:
        """Train for one epoch with chunked data from dataset.

        Args:
            train_loader: Training data loader with chunks

        Returns:
            Tuple of (average_loss, accuracy)
        """
        self.model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        train_pbar = tqdm(train_loader, desc=f"Epoch {self.current_epoch+1} [Train]")

        for batch in train_pbar:
            tof_data = batch["tof"].to(self.device)  # (batch_size, chunk_len, 320)
            acc_data = batch["acc"].to(self.device)  # (batch_size, chunk_len, acc_dim)
            rot_data = batch["rot"].to(self.device)  # (batch_size, chunk_len, rot_dim)
            thm_data = batch["thm"].to(self.device)  # (batch_size, chunk_len, thm_dim)
            labels = batch["label"].to(self.device)  # (batch_size,)

            self.optimizer.zero_grad()

            # Simple forward pass through model
            outputs = self.model(
                tof_data,
                acc_data,
                rot_data,
                thm_data,
            )  # (batch_size, num_classes)
            loss = self.criterion(outputs, labels)

            loss.backward()

            # Gradient clipping
            if self.training_config.get("gradient_clip_norm"):
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    max_norm=self.training_config["gradient_clip_norm"],
                )

            self.optimizer.step()

            # Statistics
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()

            # Update progress bar
            train_acc = 100.0 * train_correct / train_total
            train_pbar.set_postfix(
                {
                    "Loss": f"{loss.item():.4f}",
                    "Acc": f"{train_acc:.2f}%",
                },
            )

        avg_loss = train_loss / len(train_loader) if len(train_loader) > 0 else 0.0
        accuracy = 100.0 * train_correct / train_total if train_total > 0 else 0.0

        # Log training metrics to wandb
        if self.use_wandb:
            wandb.log(
                {
                    "epoch": self.current_epoch + 1,
                    "train_loss": avg_loss,
                    "train_accuracy": accuracy,
                    "learning_rate": self.optimizer.param_groups[0]["lr"],
                },
            )

        return avg_loss, accuracy

    def validate_epoch(self, val_loader: DataLoader) -> tuple[float, float]:
        """Validate for one epoch with chunked data from dataset.

        Args:
            val_loader: Validation data loader with chunks

        Returns:
            Tuple of (average_loss, accuracy)
        """
        self.model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            val_pbar = tqdm(val_loader, desc=f"Epoch {self.current_epoch+1} [Val]")

            for batch in val_pbar:
                tof_data = batch["tof"].to(self.device)  # (batch_size, chunk_len, 320)
                acc_data = batch["acc"].to(
                    self.device,
                )  # (batch_size, chunk_len, acc_dim)
                rot_data = batch["rot"].to(
                    self.device,
                )  # (batch_size, chunk_len, rot_dim)
                thm_data = batch["thm"].to(
                    self.device,
                )  # (batch_size, chunk_len, thm_dim)
                labels = batch["label"].to(self.device)  # (batch_size,)

                # Simple forward pass through model
                outputs = self.model(
                    tof_data,
                    acc_data,
                    rot_data,
                    thm_data,
                )  # (batch_size, num_classes)
                loss = self.criterion(outputs, labels)

                # Statistics
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

                # Update progress bar
                val_acc = 100.0 * val_correct / val_total
                val_pbar.set_postfix(
                    {
                        "Loss": f"{loss.item():.4f}",
                        "Acc": f"{val_acc:.2f}%",
                    },
                )

        avg_loss = val_loss / len(val_loader) if len(val_loader) > 0 else 0.0
        accuracy = 100.0 * val_correct / val_total if val_total > 0 else 0.0

        # Log validation metrics to wandb
        if self.use_wandb:
            wandb.log(
                {
                    "val_loss": avg_loss,
                    "val_accuracy": accuracy,
                },
            )

        return avg_loss, accuracy

    def finish(self):
        """Clean up wandb run."""
        if self.use_wandb:
            wandb.finish()

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        num_epochs: int,
        model_save_path: str = "models/best_model.pt",
    ) -> dict:
        """Main training loop.

        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            num_epochs: Number of epochs to train
            model_save_path: Path to save best model

        Returns:
            Dictionary with training history and final metrics
        """

        # Create save directory
        os.makedirs(os.path.dirname(model_save_path), exist_ok=True)

        for epoch in range(num_epochs):
            self.current_epoch = epoch

            # Training phase
            train_loss, train_acc = self.train_epoch(train_loader)

            # Validation phase
            val_loss, val_acc = self.validate_epoch(val_loader)

            # Store metrics
            self.train_losses.append(train_loss)
            self.train_accuracies.append(train_acc)
            self.val_losses.append(val_loss)
            self.val_accuracies.append(val_acc)

            # Learning rate scheduling
            self.scheduler.step(val_acc)

            # Save best model
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.save_checkpoint(model_save_path, epoch, val_acc)


        return {
            "train_losses": self.train_losses,
            "train_accuracies": self.train_accuracies,
            "val_losses": self.val_losses,
            "val_accuracies": self.val_accuracies,
            "best_val_accuracy": self.best_val_acc,
            "final_train_accuracy": self.train_accuracies[-1],
            "final_val_accuracy": self.val_accuracies[-1],
        }

    def save_checkpoint(
        self,
        path: str,
        epoch: int,
        val_acc: float,
        label_encoder=None,
        additional_data: dict | None = None,
    ):
        """Save model checkpoint.

        Args:
            path: Path to save checkpoint
            epoch: Current epoch
            val_acc: Validation accuracy
            label_encoder: Label encoder instance
            additional_data: Additional data to save
        """
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "best_val_acc": val_acc,
            "model_config": self.model_config,
            "training_config": self.training_config,
        }

        if label_encoder is not None:
            checkpoint["label_encoder"] = label_encoder

        if additional_data is not None:
            checkpoint.update(additional_data)

        torch.save(checkpoint, path)

    def load_checkpoint(self, path: str, load_optimizer: bool = True) -> dict:
        """Load model checkpoint.

        Args:
            path: Path to checkpoint
            load_optimizer: Whether to load optimizer state

        Returns:
            Checkpoint dictionary
        """
        checkpoint = torch.load(path, map_location=self.device)

        self.model.load_state_dict(checkpoint["model_state_dict"])

        if load_optimizer:
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            if "scheduler_state_dict" in checkpoint:
                self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        self.best_val_acc = checkpoint.get("best_val_acc", 0.0)
        self.current_epoch = checkpoint.get("epoch", 0)

        return checkpoint

    def count_parameters(self) -> dict[str, int]:
        """Count model parameters."""
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(
            p.numel() for p in self.model.parameters() if p.requires_grad
        )

        return {
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
        }


def create_trainer_config(
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
    lr_factor: float = 0.5,
    lr_patience: int = 5,
    gradient_clip_norm: float | None = 1.0,
    verbose: bool = True,
) -> dict:
    """Create training configuration dictionary.

    Args:
        learning_rate: Initial learning rate
        weight_decay: Weight decay for optimizer
        lr_factor: Factor to reduce learning rate
        lr_patience: Patience for learning rate scheduler
        gradient_clip_norm: Gradient clipping norm (None to disable)
        verbose: Whether to print verbose output

    Returns:
        Training configuration dictionary
    """
    return {
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "lr_factor": lr_factor,
        "lr_patience": lr_patience,
        "gradient_clip_norm": gradient_clip_norm,
        "verbose": verbose,
    }


def create_model_config(
    num_classes: int = 18,
    d_model: int = 128,
    d_reduced: int = 128,
    num_heads: int = 8,
    num_layers: int = 1,
    acc_dim: int = 4,
    rot_dim: int = 8,
    thm_dim: int = 5,
    dropout: float = 0.1,
    max_seq_length: int = 5000,
    sequence_processor: str = "transformer",
    tof_backbone: str = "b0",
) -> dict:
    """Create model configuration dictionary.

    Args:
        num_classes: Number of gesture classes
        d_model: Model dimension
        d_reduced: Reduced feature dimension after feature selection
        num_heads: Number of attention heads
        num_layers: Number of transformer layers
        acc_dim: Accelerometer feature dimension
        rot_dim: Rotation feature dimension
        thm_dim: Thermal feature dimension
        dropout: Dropout rate
        max_seq_length: Maximum sequence length for positional encoding
        sequence_processor: Sequence processor type ("transformer" or "gru")
        tof_backbone: TOF backbone architecture ("b0" or "b3")

    Returns:
        Model configuration dictionary
    """
    return {
        "num_classes": num_classes,
        "d_model": d_model,
        "d_reduced": d_reduced,
        "num_heads": num_heads,
        "num_layers": num_layers,
        "acc_dim": acc_dim,
        "rot_dim": rot_dim,
        "thm_dim": thm_dim,
        "dropout": dropout,
        "max_seq_length": max_seq_length,
        "sequence_processor": sequence_processor,
        "tof_backbone": tof_backbone,
    }
