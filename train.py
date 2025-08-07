#!/usr/bin/env python3
"""CMI Gesture Recognition Training Script.

This script trains the four-branch gesture recognition model for the CMI competition.
"""

import argparse
import os
import sys

import polars as pl
from torch.utils.data import DataLoader

# Import modularized components
from src.dataset import (
    CMIDataset,
    SequenceProcessor,
    get_enhanced_feature_dims,
    prepare_gesture_labels,
)
from src.evaluator import CMIEvaluator
from src.trainer import CMITrainer, create_model_config, create_trainer_config
from src.utils import (
    calculate_optimal_max_length,
    check_gpu_usage,
    create_experiment_dir,
    format_time,
    get_sequence_statistics,
    log_experiment_results,
    print_data_info,
    print_model_info,
    save_config,
    set_seed,
)


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train CMI Gesture Recognition Model")

    # Data arguments
    parser.add_argument(
        "--data-dir",
        type=str,
        default="dataset",
        help="Directory containing training data",
    )
    parser.add_argument(
        "--enhanced-features",
        action="store_true",
        default=True,
        help="Use enhanced feature processing",
    )
    parser.add_argument(
        "--max-seq-length",
        type=int,
        default=None,
        help="Maximum sequence length (auto-calculate if None)",
    )
    parser.add_argument(
        "--percentile-cutoff",
        type=float,
        default=95.0,
        help="Percentile for auto max sequence length calculation",
    )

    # Model arguments
    parser.add_argument(
        "--d-model",
        type=int,
        default=128,
        help="Model embedding dimension",
    )
    parser.add_argument(
        "--num-heads",
        type=int,
        default=8,
        help="Number of attention heads",
    )
    parser.add_argument(
        "--num-classes",
        type=int,
        default=18,
        help="Number of gesture classes",
    )

    # Training arguments
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Training batch size",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-3,
        help="Initial learning rate",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=1e-4,
        help="Weight decay for optimizer",
    )
    parser.add_argument(
        "--gradient-clip",
        type=float,
        default=1.0,
        help="Gradient clipping norm (0 to disable)",
    )

    # Other arguments
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=2,
        help="Number of data loading workers",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cuda", "cpu"],
        help="Device to use for training",
    )
    parser.add_argument(
        "--experiment-name",
        type=str,
        default="cmi_training",
        help="Name for the experiment",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="experiments",
        help="Output directory for experiments",
    )

    return parser.parse_args()


def load_data(args):
    """Load and prepare training data."""
    print("Loading training data...")

    # Load raw data
    train_df = pl.read_csv(f"{args.data_dir}/train.csv")
    demographics_df = pl.read_csv(f"{args.data_dir}/train_demographics.csv")

    # Print data info
    print_data_info(train_df, demographics_df)

    # Prepare gesture labels
    train_df, label_encoder, target_gestures, non_target_gestures = (
        prepare_gesture_labels(train_df)
    )

    print("\nGesture ID mapping:")
    for i, gesture in enumerate(label_encoder.classes_):
        print(f"{i}: {gesture}")

    return (
        train_df,
        demographics_df,
        label_encoder,
        target_gestures,
        non_target_gestures,
    )


def process_sequences(args, train_df, label_encoder):
    """Process sequences with enhanced features."""
    print("\nProcessing sequences...")

    # Initialize sequence processor
    processor = SequenceProcessor(use_enhanced_features=args.enhanced_features)

    # Process dataframe into sequences
    sequences, sequence_lengths = processor.process_dataframe(train_df, label_encoder)

    # Print sequence statistics
    stats = get_sequence_statistics(sequences)
    print("\nSequence Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")

    # Calculate optimal max sequence length
    if args.max_seq_length is None:
        max_seq_length = calculate_optimal_max_length(
            sequence_lengths,
            percentile=args.percentile_cutoff,
        )
    else:
        max_seq_length = args.max_seq_length
        print(f"\nUsing specified max sequence length: {max_seq_length}")

    # Create train/val split
    train_sequences, val_sequences = processor.create_train_val_split(sequences)

    print("\nDataset split:")
    print(f"  Training sequences: {len(train_sequences)}")
    print(f"  Validation sequences: {len(val_sequences)}")

    # Get feature dimensions if using enhanced features
    feature_dims = get_enhanced_feature_dims(sequences)
    print("\nFeature dimensions:")
    for key, dim in feature_dims.items():
        print(f"  {key}: {dim}")

    return train_sequences, val_sequences, max_seq_length, feature_dims


def create_dataloaders(args, train_sequences, val_sequences, max_seq_length):
    """Create training and validation dataloaders."""
    print("\nCreating dataloaders...")

    # Create datasets
    train_dataset = CMIDataset(
        train_sequences,
        max_length=max_seq_length,
        use_enhanced_features=args.enhanced_features,
    )
    val_dataset = CMIDataset(
        val_sequences,
        max_length=max_seq_length,
        use_enhanced_features=args.enhanced_features,
    )

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=args.device != "cpu",
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.device != "cpu",
    )

    print(f"  Train batches: {len(train_loader)}")
    print(f"  Validation batches: {len(val_loader)}")

    return train_loader, val_loader


def setup_training(args, feature_dims, exp_dir):
    """Setup model and trainer."""
    print("\nSetting up training...")

    # Create model configuration
    model_config = create_model_config(
        num_classes=args.num_classes,
        d_model=args.d_model,
        num_heads=args.num_heads,
        acc_dim=feature_dims["acc"],
        rot_dim=feature_dims["rot"],
        thm_dim=feature_dims["thm"],
    )

    # Create training configuration
    training_config = create_trainer_config(
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        gradient_clip_norm=args.gradient_clip if args.gradient_clip > 0 else None,
    )

    # Save configurations
    save_config(model_config, os.path.join(exp_dir, "model_config.json"))
    save_config(training_config, os.path.join(exp_dir, "training_config.json"))
    save_config(vars(args), os.path.join(exp_dir, "args.json"))

    # Create trainer
    trainer = CMITrainer(model_config, training_config, device=args.device)

    # Print model info
    print_model_info(trainer.model)

    return trainer, model_config


def main():
    """Main training function."""
    import time

    start_time = time.time()

    # Parse arguments
    args = parse_arguments()

    # Set seed for reproducibility
    set_seed(args.seed)

    # Check GPU
    check_gpu_usage()

    # Create experiment directory
    exp_dir = create_experiment_dir(args.output_dir, args.experiment_name)

    try:
        # Load data
        (
            train_df,
            demographics_df,
            label_encoder,
            target_gestures,
            non_target_gestures,
        ) = load_data(args)

        # Process sequences
        train_sequences, val_sequences, max_seq_length, feature_dims = (
            process_sequences(
                args,
                train_df,
                label_encoder,
            )
        )

        # Create dataloaders
        train_loader, val_loader = create_dataloaders(
            args,
            train_sequences,
            val_sequences,
            max_seq_length,
        )

        # Setup training
        trainer, model_config = setup_training(args, feature_dims, exp_dir)

        # Train model
        print(f"\n{'='*60}")
        print("Starting training...")
        print(f"{'='*60}")

        model_save_path = os.path.join(exp_dir, "models", "best_model.pt")
        training_history = trainer.train(
            train_loader=train_loader,
            val_loader=val_loader,
            num_epochs=args.epochs,
            model_save_path=model_save_path,
        )

        # Evaluate model
        print(f"\n{'='*60}")
        print("Evaluating model...")
        print(f"{'='*60}")

        # Load best model
        trainer.load_checkpoint(model_save_path, load_optimizer=False)
        evaluator = CMIEvaluator(trainer.model, device=args.device)

        # Run evaluation
        results = evaluator.evaluate(val_loader, label_encoder)

        # Print results
        print("\nFinal Results:")
        print(f"  Validation Accuracy: {results['accuracy']:.4f}")
        print(f"  Macro F1: {results['macro_f1']:.4f}")
        print(f"  Weighted F1: {results['weighted_f1']:.4f}")

        # Analyze per-class performance
        per_class_df = evaluator.analyze_per_class_performance(
            results,
            target_gestures,
            non_target_gestures,
        )
        print("\nPer-class Performance (top 5):")
        print(per_class_df.head())

        # Plot training curves
        evaluator.plot_training_curves(
            training_history,
            save_path=os.path.join(exp_dir, "plots", "training_curves.png"),
        )

        # Plot confusion matrix
        evaluator.plot_confusion_matrix(
            val_loader,
            label_encoder,
            save_path=os.path.join(exp_dir, "plots", "confusion_matrix.png"),
        )

        # Print classification report
        evaluator.print_classification_report(val_loader, label_encoder)

        # Save final results
        final_results = {
            "training_history": training_history,
            "evaluation_results": {
                "accuracy": results["accuracy"],
                "macro_f1": results["macro_f1"],
                "weighted_f1": results["weighted_f1"],
                "per_class_f1": [r["f1_score"] for r in results["per_class_results"]],
            },
            "model_config": model_config,
            "feature_dims": feature_dims,
            "max_seq_length": max_seq_length,
            "total_parameters": trainer.count_parameters()["total_parameters"],
        }

        # Save results
        evaluator.save_results(
            results,
            os.path.join(exp_dir, "evaluation_results.json"),
        )
        log_experiment_results(
            final_results,
            os.path.join(exp_dir, "experiment_log.txt"),
        )

        # Save final model with metadata
        final_model_path = os.path.join(exp_dir, "models", "final_model.pt")
        trainer.save_checkpoint(
            final_model_path,
            epoch=args.epochs - 1,
            val_acc=results["accuracy"] * 100,
            label_encoder=label_encoder,
            additional_data={
                "training_history": training_history,
                "final_metrics": final_results["evaluation_results"],
                "feature_dims": feature_dims,
                "max_seq_length": max_seq_length,
                "target_gestures": target_gestures,
                "non_target_gestures": non_target_gestures,
            },
        )

        # Print final summary
        end_time = time.time()
        total_time = end_time - start_time

        print(f"\n{'='*60}")
        print("TRAINING COMPLETED SUCCESSFULLY!")
        print(f"{'='*60}")
        print(f"Total training time: {format_time(total_time)}")
        print(f"Best validation accuracy: {training_history['best_val_accuracy']:.2f}%")
        print(f"Final validation accuracy: {results['accuracy']*100:.2f}%")
        print(f"Experiment directory: {exp_dir}")
        print(f"Model saved: {final_model_path}")

    except Exception as e:
        print(f"\nERROR: Training failed with exception: {e}")
        import traceback

        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
