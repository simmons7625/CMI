from __future__ import annotations

from typing import TYPE_CHECKING, Any

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import seaborn as sns
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from tqdm import tqdm

if TYPE_CHECKING:
    from torch.utils.data import DataLoader


class CMIEvaluator:
    """Evaluator class for CMI gesture recognition model."""

    def __init__(self, model: torch.nn.Module, device: str = "auto"):
        """Initialize evaluator.

        Args:
            model: Trained model to evaluate
            device: Device to use ('auto', 'cuda', 'cpu')
        """
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.model = model.to(self.device)
        self.model.eval()

    def predict(
        self,
        data_loader: DataLoader,
    ) -> tuple[list[int], list[int], list[list[float]]]:
        """Generate predictions for a dataset.

        Args:
            data_loader: DataLoader for the dataset

        Returns:
            Tuple of (predictions, true_labels, prediction_probabilities)
        """
        all_predictions = []
        all_labels = []
        all_probabilities = []

        with torch.no_grad():
            for batch in tqdm(data_loader, desc="Generating predictions"):
                # Move to device
                tof_data = batch["tof"].to(self.device)
                acc_data = batch["acc"].to(self.device)
                rot_data = batch["rot"].to(self.device)
                thm_data = batch["thm"].to(self.device)
                labels = batch["label"].to(self.device)

                # Forward pass
                outputs = self.model(tof_data, acc_data, rot_data, thm_data)
                probabilities = torch.softmax(outputs, dim=1)
                _, predicted = torch.max(outputs, 1)

                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())

        return all_predictions, all_labels, all_probabilities

    def evaluate(self, data_loader: DataLoader, label_encoder=None) -> dict[str, Any]:
        """Comprehensive model evaluation.

        Args:
            data_loader: DataLoader for evaluation dataset
            label_encoder: Label encoder for class names

        Returns:
            Dictionary with evaluation metrics
        """
        predictions, labels, probabilities = self.predict(data_loader)

        # Basic metrics
        accuracy = accuracy_score(labels, predictions)

        # Per-class metrics
        precision, recall, f1, support = precision_recall_fscore_support(
            labels,
            predictions,
            average=None,
        )

        # Macro averages
        macro_precision = np.mean(precision)
        macro_recall = np.mean(recall)
        macro_f1 = np.mean(f1)

        # Weighted averages
        weighted_precision = np.average(precision, weights=support)
        weighted_recall = np.average(recall, weights=support)
        weighted_f1 = np.average(f1, weights=support)

        # Class names
        if label_encoder is not None:
            class_names = label_encoder.classes_
        else:
            class_names = [f"Class_{i}" for i in range(len(precision))]

        # Per-class results
        per_class_results = []
        for i, class_name in enumerate(class_names):
            per_class_results.append(
                {
                    "class": class_name,
                    "precision": precision[i],
                    "recall": recall[i],
                    "f1_score": f1[i],
                    "support": support[i],
                },
            )

        return {
            "accuracy": accuracy,
            "macro_precision": macro_precision,
            "macro_recall": macro_recall,
            "macro_f1": macro_f1,
            "weighted_precision": weighted_precision,
            "weighted_recall": weighted_recall,
            "weighted_f1": weighted_f1,
            "per_class_results": per_class_results,
            "predictions": predictions,
            "true_labels": labels,
            "probabilities": probabilities,
            "class_names": class_names,
        }

    def print_classification_report(
        self,
        data_loader: DataLoader,
        label_encoder=None,
        digits: int = 3,
    ):
        """Print detailed classification report.

        Args:
            data_loader: DataLoader for evaluation dataset
            label_encoder: Label encoder for class names
            digits: Number of digits to display
        """
        predictions, labels, _ = self.predict(data_loader)

        target_names = label_encoder.classes_ if label_encoder is not None else None

        report = classification_report(
            labels,
            predictions,
            target_names=target_names,
            digits=digits,
        )
        print("Classification Report:")
        print(report)

    def plot_confusion_matrix(
        self,
        data_loader: DataLoader,
        label_encoder=None,
        figsize: tuple[int, int] = (16, 12),
        save_path: str | None = None,
    ):
        """Plot confusion matrix.

        Args:
            data_loader: DataLoader for evaluation dataset
            label_encoder: Label encoder for class names
            figsize: Figure size
            save_path: Path to save the plot (optional)
        """
        predictions, labels, _ = self.predict(data_loader)
        cm = confusion_matrix(labels, predictions)

        if label_encoder is not None:
            class_names = label_encoder.classes_
        else:
            class_names = [f"Class_{i}" for i in range(len(cm))]

        plt.figure(figsize=figsize)
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=class_names,
            yticklabels=class_names,
        )
        plt.title("Confusion Matrix")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.xticks(rotation=45, ha="right")
        plt.yticks(rotation=0)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.show()

    def analyze_per_class_performance(
        self,
        results: dict[str, Any],
        target_gestures: list[str] | None = None,
        non_target_gestures: list[str] | None = None,
    ) -> pl.DataFrame:
        """Analyze performance by gesture category.

        Args:
            results: Results from evaluate() method
            target_gestures: List of target gesture names
            non_target_gestures: List of non-target gesture names

        Returns:
            Polars DataFrame with performance analysis
        """
        # Create DataFrame from per-class results
        per_class_data = []
        for result in results["per_class_results"]:
            per_class_data.append(
                {
                    "gesture": result["class"],
                    "precision": result["precision"],
                    "recall": result["recall"],
                    "f1_score": result["f1_score"],
                    "support": result["support"],
                },
            )

        df = pl.DataFrame(per_class_data)

        # Add category classification if provided
        if target_gestures is not None and non_target_gestures is not None:
            df = df.with_columns(
                pl.when(pl.col("gesture").is_in(target_gestures))
                .then(pl.lit("Target"))
                .otherwise(pl.lit("Non-target"))
                .alias("category"),
            )

            # Calculate category averages
            category_stats = df.group_by("category").agg(
                [
                    pl.col("precision").mean().alias("avg_precision"),
                    pl.col("recall").mean().alias("avg_recall"),
                    pl.col("f1_score").mean().alias("avg_f1_score"),
                    pl.col("support").sum().alias("total_support"),
                ],
            )

            print("Performance by Category:")
            print(category_stats)

        # Sort by F1 score
        return df.sort("f1_score", descending=True)

    def plot_training_curves(
        self,
        training_history: dict[str, list[float]],
        figsize: tuple[int, int] = (15, 5),
        save_path: str | None = None,
    ):
        """Plot training and validation curves.

        Args:
            training_history: Training history dictionary
            figsize: Figure size
            save_path: Path to save the plot (optional)
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

        epochs = range(1, len(training_history["train_losses"]) + 1)

        # Loss curves
        ax1.plot(epochs, training_history["train_losses"], "b-", label="Training Loss")
        ax1.plot(epochs, training_history["val_losses"], "r-", label="Validation Loss")
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Loss")
        ax1.set_title("Training and Validation Loss")
        ax1.legend()
        ax1.grid(True)

        # Accuracy curves
        ax2.plot(
            epochs,
            training_history["train_accuracies"],
            "b-",
            label="Training Accuracy",
        )
        ax2.plot(
            epochs,
            training_history["val_accuracies"],
            "r-",
            label="Validation Accuracy",
        )
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Accuracy (%)")
        ax2.set_title("Training and Validation Accuracy")
        ax2.legend()
        ax2.grid(True)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.show()

        # Print final results
        print("Final Training Results:")
        print(
            f"  Final Train Accuracy: {training_history['train_accuracies'][-1]:.2f}%",
        )
        print(
            f"  Final Validation Accuracy: {training_history['val_accuracies'][-1]:.2f}%",
        )
        print(
            f"  Best Validation Accuracy: {max(training_history['val_accuracies']):.2f}%",
        )

    def calculate_confidence_metrics(self, data_loader: DataLoader) -> dict[str, float]:
        """Calculate prediction confidence metrics.

        Args:
            data_loader: DataLoader for evaluation dataset

        Returns:
            Dictionary with confidence metrics
        """
        predictions, labels, probabilities = self.predict(data_loader)
        probabilities = np.array(probabilities)

        # Max probabilities (confidence scores)
        max_probs = np.max(probabilities, axis=1)

        # Entropy (uncertainty measure)
        entropy = -np.sum(probabilities * np.log(probabilities + 1e-8), axis=1)

        # Confidence for correct vs incorrect predictions
        correct_mask = np.array(predictions) == np.array(labels)

        return {
            "mean_confidence": np.mean(max_probs),
            "mean_confidence_correct": np.mean(max_probs[correct_mask]),
            "mean_confidence_incorrect": np.mean(max_probs[~correct_mask]),
            "mean_entropy": np.mean(entropy),
            "mean_entropy_correct": np.mean(entropy[correct_mask]),
            "mean_entropy_incorrect": np.mean(entropy[~correct_mask]),
        }

    def _make_json_serializable(self, obj):
        """Recursively convert numpy types to JSON serializable types."""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, dict):
            return {
                key: self._make_json_serializable(value) for key, value in obj.items()
            }
        if isinstance(obj, list):
            return [self._make_json_serializable(item) for item in obj]
        return obj

    def save_results(self, results: dict[str, Any], save_path: str):
        """Save evaluation results to file.

        Args:
            results: Results dictionary from evaluate()
            save_path: Path to save results
        """
        # Recursively convert all numpy types to JSON serializable types
        serializable_results = self._make_json_serializable(results)

        import json

        with open(save_path, "w") as f:
            json.dump(serializable_results, f, indent=2)

        print(f"Results saved to: {save_path}")


def load_model_for_evaluation(
    checkpoint_path: str,
    device: str = "auto",
) -> tuple[torch.nn.Module, dict]:
    """Load trained model from checkpoint for evaluation.

    Args:
        checkpoint_path: Path to model checkpoint
        device: Device to use

    Returns:
        Tuple of (model, checkpoint_data)
    """
    if device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Import here to avoid circular imports
    from .model import create_model

    # Create model from saved config
    model_config = checkpoint.get("model_config", {})
    model = create_model(**model_config)

    # Load model weights
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    print(f"Model loaded from: {checkpoint_path}")
    print(f"Best validation accuracy: {checkpoint.get('best_val_acc', 'Unknown'):.2f}%")

    return model, checkpoint
