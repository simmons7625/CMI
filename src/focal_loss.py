"""Focal Loss implementation for addressing class imbalance in gesture recognition."""

import torch
import torch.nn.functional as F
from torch import nn


class FocalLoss(nn.Module):
    """Focal Loss implementation for multi-class classification.

    Focal Loss addresses class imbalance by down-weighting easy examples
    and focusing on hard examples during training.

    Reference: "Focal Loss for Dense Object Detection" - Lin et al. (2017)
    """

    def __init__(
        self,
        alpha=None,
        gamma=2.0,
        reduction="mean",
        ignore_index=-100,
        label_smoothing=0.0,
    ):
        """Initialize Focal Loss.

        Args:
            alpha: Class weights (tensor of size num_classes or None for uniform)
            gamma: Focusing parameter (higher gamma puts more focus on hard examples)
            reduction: Reduction method ('mean', 'sum', or 'none')
            ignore_index: Index to ignore in loss calculation
            label_smoothing: Label smoothing factor (0.0 = no smoothing, 0.1 = 10% smoothing)
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.ignore_index = ignore_index
        self.label_smoothing = label_smoothing

    def forward(self, inputs, targets):
        """Forward pass for Focal Loss.

        Args:
            inputs: Predictions (batch_size, num_classes) - logits
            targets: Ground truth labels (batch_size,) - class indices

        Returns:
            Focal loss value
        """
        # Apply label smoothing if specified
        if self.label_smoothing > 0.0:
            num_classes = inputs.size(-1)
            # Create smoothed targets
            # Convert targets to one-hot and apply smoothing
            one_hot = torch.zeros_like(inputs).scatter_(1, targets.unsqueeze(1), 1)
            smooth_targets = (
                one_hot * (1.0 - self.label_smoothing)
                + self.label_smoothing / num_classes
            )

            # Compute cross entropy with smoothed targets
            log_probs = F.log_softmax(inputs, dim=-1)
            ce_loss = -(smooth_targets * log_probs).sum(dim=-1)

            # For focal loss, we need p_t for the true class
            # Use the original hard targets to get p_t
            pt = torch.exp(
                -F.cross_entropy(
                    inputs, targets, reduction="none", ignore_index=self.ignore_index
                )
            )
        else:
            # Standard cross entropy without smoothing
            ce_loss = F.cross_entropy(
                inputs, targets, reduction="none", ignore_index=self.ignore_index
            )
            pt = torch.exp(-ce_loss)

        # Compute alpha_t if alpha is provided
        if self.alpha is not None:
            if self.alpha.device != targets.device:
                self.alpha = self.alpha.to(targets.device)
            alpha_t = self.alpha[targets]
        else:
            alpha_t = 1.0

        # Compute focal loss
        focal_loss = alpha_t * (1 - pt) ** self.gamma * ce_loss

        # Apply reduction
        if self.reduction == "mean":
            return focal_loss.mean()
        if self.reduction == "sum":
            return focal_loss.sum()
        return focal_loss


def create_focal_loss(
    num_classes: int,
    class_counts: list[int] = None,
    gamma: float = 2.0,
    reduction: str = "mean",
    label_smoothing: float = 0.0,
) -> FocalLoss:
    """Create Focal Loss with automatic alpha calculation from class counts.

    Args:
        num_classes: Number of classes
        class_counts: List of class counts for calculating alpha weights
        gamma: Focusing parameter
        reduction: Reduction method
        label_smoothing: Label smoothing factor

    Returns:
        FocalLoss instance
    """
    alpha = None

    if class_counts is not None:
        # Calculate inverse frequency weights
        total_samples = sum(class_counts)
        alpha_weights = []

        for count in class_counts:
            if count > 0:
                # Inverse frequency weighting
                weight = total_samples / (num_classes * count)
                alpha_weights.append(weight)
            else:
                # Handle zero counts
                alpha_weights.append(1.0)

        alpha = torch.FloatTensor(alpha_weights)
        print(f"📊 Focal Loss alpha weights: {alpha.tolist()}")

    if label_smoothing > 0.0:
        print(f"🎯 Focal Loss with label smoothing: {label_smoothing}")

    return FocalLoss(
        alpha=alpha, gamma=gamma, reduction=reduction, label_smoothing=label_smoothing
    )


def calculate_class_weights(sequences: list[dict], num_classes: int) -> list[int]:
    """Calculate class counts from sequences for focal loss weighting.

    Args:
        sequences: List of sequence dictionaries
        num_classes: Total number of classes

    Returns:
        List of class counts
    """
    class_counts = [0] * num_classes

    for seq in sequences:
        label = seq.get("label", -1)
        if 0 <= label < num_classes:
            class_counts[label] += 1

    return class_counts
