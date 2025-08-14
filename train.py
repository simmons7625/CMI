#!/usr/bin/env python3
"""CMI Gesture Recognition Training Script with Config File Support."""

import argparse
import json
import sys
import time
import traceback
from pathlib import Path

import polars as pl
import torch
import yaml
from torch.utils.data import DataLoader

from src.dataset import (
    CMIDataset,
    SequenceProcessor,
    get_enhanced_feature_dims,
    prepare_gesture_labels,
)
from src.evaluator import CMIEvaluator
from src.trainer import CMITrainer, create_model_config, create_trainer_config
from src.utils import create_experiment_dir, set_seed


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train CMI Gesture Recognition Model")
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to YAML configuration file",
    )
    return parser.parse_args()


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    config_path = Path(config_path)
    if not config_path.exists():
        msg = f"Config file not found: {config_path}"
        raise FileNotFoundError(msg)

    print(f"✅ Loading configuration from: {config_path}")
    with open(config_path) as f:
        return yaml.safe_load(f)


def load_and_process_data(config: dict):
    """Load and process training data."""
    print("Loading training data...")

    # Load data
    train_df = pl.read_csv(config["data"]["train_path"])
    train_df = train_df.fill_null(-1.0).fill_nan(-1.0)

    # Prepare labels
    train_df, label_encoder, target_gestures, non_target_gestures = (
        prepare_gesture_labels(train_df)
    )

    print("\nGesture ID mapping:")
    for i, gesture in enumerate(label_encoder.classes_):
        print(f"  {i}: {gesture}")

    # Process sequences
    processor = SequenceProcessor()
    num_samples = config["data"].get("num_samples")
    chunk_length = config["data"].get(
        "max_seq_length",
        100,
    )  # For chunking during training
    sequences = processor.process_dataframe(
        train_df,
        num_samples=num_samples,
        max_seq_length=chunk_length,
    )

    # Get feature dimensions
    feature_dims = get_enhanced_feature_dims(sequences)

    # Create train/val split
    train_sequences, val_sequences = processor.create_train_val_split(
        sequences,
        test_size=config["data"]["val_split"],
        random_state=config["training"]["random_seed"],
    )

    print(f"Training sequences: {len(train_sequences)}")
    print(f"Validation sequences: {len(val_sequences)}")

    return {
        "train_sequences": train_sequences,
        "val_sequences": val_sequences,
        "label_encoder": label_encoder,
        "max_seq_length": chunk_length,  # Return the chunk length used for training
        "feature_dims": feature_dims,
        "target_gestures": target_gestures,
        "non_target_gestures": non_target_gestures,
    }


def create_data_loaders(data_info: dict, config: dict):
    """Create training and validation data loaders."""
    print("Creating data loaders...")

    # Create datasets
    train_dataset = CMIDataset(
        data_info["train_sequences"],
        max_length=data_info["max_seq_length"],
    )

    val_dataset = CMIDataset(
        data_info["val_sequences"],
        max_length=data_info["max_seq_length"],
    )

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=True,
        num_workers=config["training"]["num_workers"],
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=False,
        num_workers=config["training"]["num_workers"],
    )

    print(f"Train batches: {len(train_loader)}")
    print(f"Validation batches: {len(val_loader)}")

    return train_loader, val_loader


def main():
    """Main training function."""
    # Parse arguments
    args = parse_arguments()

    # Load configuration
    try:
        config = load_config(args.config)
    except FileNotFoundError as e:
        print(f"❌ {e}")
        return 1

    # Set seed for reproducibility
    set_seed(config["training"]["random_seed"])

    # Check GPU availability
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create experiment directory
    exp_name = config.get("experiment_name", "cmi_training")
    exp_dir = create_experiment_dir(base_dir="experiments", experiment_name=exp_name)
    print(f"Experiment directory: {exp_dir}")

    try:
        start_time = time.time()

        # Load and process data
        data_info = load_and_process_data(config)

        # Create data loaders
        train_loader, val_loader = create_data_loaders(data_info, config)

        # Create model configuration
        model_config = create_model_config(
            num_classes=config["model"]["num_classes"],
            d_model=config["model"]["d_model"],
            d_reduced=config["model"]["d_reduced"],
            num_heads=config["model"]["num_heads"],
            num_layers=config["model"].get("num_layers", 1),
            acc_dim=data_info["feature_dims"]["acc"],
            rot_dim=data_info["feature_dims"]["rot"],
            thm_dim=data_info["feature_dims"]["thm"],
            dropout=config["model"].get("dropout", 0.1),
            max_seq_length=config["model"].get(
                "max_seq_length",
                5000,
            ),  # For positional encoding
        )

        # Create trainer configuration
        trainer_config = create_trainer_config(
            learning_rate=config["training"]["learning_rate"],
            weight_decay=config["training"]["weight_decay"],
            lr_factor=config["training"]["scheduler"]["decay_factor"],
            lr_patience=config["training"]["scheduler"]["patience"],
            gradient_clip_norm=config["training"]["gradient_clip_norm"],
        )

        # Create trainer with wandb config
        use_wandb = config.get("logging", {}).get("use_wandb", False)
        wandb_config = config.get("logging", {}).get("wandb", {})
        trainer = CMITrainer(
            model_config,
            trainer_config,
            device="auto",
            use_wandb=use_wandb,
            wandb_config=wandb_config,
        )

        # Print model info
        param_info = trainer.count_parameters()
        print(f"Model parameters: {param_info['trainable_parameters']:,}")

        # Save configurations
        config_dir = Path(exp_dir) / "configs"
        config_dir.mkdir(exist_ok=True)

        with open(config_dir / "config.yaml", "w") as f:
            yaml.dump(config, f, default_flow_style=False, indent=2)

        with open(config_dir / "model_config.json", "w") as f:
            json.dump(model_config, f, indent=2)

        with open(config_dir / "trainer_config.json", "w") as f:
            json.dump(trainer_config, f, indent=2)

        # Train model
        print(f"{'='*60}")
        print("🚀 Starting training...")
        print(f"{'='*60}")

        model_save_path = Path(exp_dir) / "models" / "best_model.pt"
        model_save_path.parent.mkdir(exist_ok=True)

        training_history = trainer.train(
            train_loader=train_loader,
            val_loader=val_loader,
            num_epochs=config["training"]["num_epochs"],
            model_save_path=str(model_save_path),
        )

        # Load best model for evaluation
        trainer.load_checkpoint(str(model_save_path), load_optimizer=False)
        evaluator = CMIEvaluator(trainer.model, device=device)

        # Evaluate model
        print(f"{'='*60}")
        print("📊 Evaluating model...")
        print(f"{'='*60}")

        results = evaluator.evaluate(val_loader, data_info["label_encoder"])

        # Print results
        print("\n🎯 Final Results:")
        print(f"  Accuracy: {results['accuracy']:.4f} ({results['accuracy']*100:.2f}%)")
        print(f"  Macro F1: {results['macro_f1']:.4f}")
        print(f"  Weighted F1: {results['weighted_f1']:.4f}")

        # Print classification report
        evaluator.print_classification_report(val_loader, data_info["label_encoder"])

        # Save visualizations
        plots_dir = Path(exp_dir) / "plots"
        plots_dir.mkdir(exist_ok=True)

        # Training curves
        evaluator.plot_training_curves(
            training_history,
            save_path=plots_dir / "training_curves.png",
        )
        print(f"📈 Training curves saved to: {plots_dir / 'training_curves.png'}")

        # Confusion matrix
        evaluator.plot_confusion_matrix(
            val_loader,
            data_info["label_encoder"],
            save_path=plots_dir / "confusion_matrix.png",
        )
        print(f"🔥 Confusion matrix saved to: {plots_dir / 'confusion_matrix.png'}")

        # Save results
        results_file = Path(exp_dir) / "results.json"
        evaluator.save_results(results, str(results_file))

        # Save final model with metadata
        final_model_path = Path(exp_dir) / "models" / "final_model.pt"
        trainer.save_checkpoint(
            str(final_model_path),
            epoch=config["training"]["num_epochs"] - 1,
            val_acc=results["accuracy"] * 100,
            label_encoder=data_info["label_encoder"],
            additional_data={
                "config": config,
                "feature_dims": data_info["feature_dims"],
                "max_seq_length": data_info["max_seq_length"],
                "training_history": training_history,
                "target_gestures": data_info["target_gestures"],
                "non_target_gestures": data_info["non_target_gestures"],
            },
        )

        # Final summary
        training_time = time.time() - start_time
        final_accuracy = results["accuracy"] * 100

        print(f"\n{'='*60}")
        print("🎉 TRAINING COMPLETED SUCCESSFULLY!")
        print(f"{'='*60}")
        print(
            f"⏱️  Training time: {training_time:.1f} seconds ({training_time/60:.1f} minutes)",
        )
        print(f"🎯 Best validation accuracy: {trainer.best_val_acc:.2f}%")
        print(f"📊 Final validation accuracy: {final_accuracy:.2f}%")
        print(f"💾 Results saved to: {exp_dir}")
        print(f"🤖 Final model saved to: {final_model_path}")
        print(f"{'='*60}")

        return 0

    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        traceback.print_exc()
        return 1

    finally:
        # Clean up wandb run
        if "trainer" in locals():
            trainer.finish()


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
