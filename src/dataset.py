from __future__ import annotations

import numpy as np
import polars as pl
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset
from tqdm import tqdm

from .feature_processor import FeatureProcessor


class CMIDataset(Dataset):
    """Dataset class for CMI gesture recognition data."""

    def __init__(
        self,
        sequences: list[dict],
        max_length: int | None = None,
        use_enhanced_features: bool = True,
    ):
        """Initialize CMI dataset.

        Args:
            sequences: List of sequence dictionaries containing data and labels
            max_length: Maximum sequence length for padding/truncation
            use_enhanced_features: Whether to use enhanced feature processing
        """
        self.sequences = sequences
        self.use_enhanced_features = use_enhanced_features

        # Calculate max length if not provided
        if max_length is None:
            if use_enhanced_features:
                self.max_length = max(
                    len(seq["enhanced_data"]["tof"])
                    if "enhanced_data" in seq
                    else len(seq["data"])
                    for seq in sequences
                )
            else:
                self.max_length = max(len(seq["data"]) for seq in sequences)
        else:
            self.max_length = max_length

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        sequence = self.sequences[idx]
        label = sequence["label"]

        if self.use_enhanced_features and "enhanced_data" in sequence:
            # Use enhanced features
            tof_data = sequence["enhanced_data"]["tof"]
            acc_data = sequence["enhanced_data"]["acc"]
            rot_data = sequence["enhanced_data"]["rot"]
            thm_data = sequence["enhanced_data"]["thm"]
        else:
            # Use original features
            data = sequence["data"]

            # Pad or truncate sequence
            seq_len = len(data)
            if seq_len < self.max_length:
                # Pad with zeros
                padding = np.zeros((self.max_length - seq_len, data.shape[1]))
                data = np.vstack([data, padding])
            elif seq_len > self.max_length:
                # Truncate
                data = data[: self.max_length]

            # Split into sensor modalities
            tof_data = data[:, :320]  # ToF features (320)
            acc_data = data[:, 320:323]  # Accelerometer (3)
            rot_data = data[:, 323:327]  # Rotation (4)
            thm_data = data[:, 327:332]  # Thermal (5)

            # Handle missing values (-1.0) by replacing with 0
            tof_data = np.where(tof_data == -1.0, 0.0, tof_data)
            acc_data = np.where(acc_data == -1.0, 0.0, acc_data)
            rot_data = np.where(rot_data == -1.0, 0.0, rot_data)
            thm_data = np.where(thm_data == -1.0, 0.0, thm_data)

        # Ensure consistent length for enhanced features
        if self.use_enhanced_features:
            seq_len = len(tof_data)
            if seq_len < self.max_length:
                # Pad enhanced features
                tof_padding = np.zeros((self.max_length - seq_len, tof_data.shape[1]))
                acc_padding = np.zeros((self.max_length - seq_len, acc_data.shape[1]))
                rot_padding = np.zeros((self.max_length - seq_len, rot_data.shape[1]))
                thm_padding = np.zeros((self.max_length - seq_len, thm_data.shape[1]))

                tof_data = np.vstack([tof_data, tof_padding])
                acc_data = np.vstack([acc_data, acc_padding])
                rot_data = np.vstack([rot_data, rot_padding])
                thm_data = np.vstack([thm_data, thm_padding])
            elif seq_len > self.max_length:
                # Truncate
                tof_data = tof_data[: self.max_length]
                acc_data = acc_data[: self.max_length]
                rot_data = rot_data[: self.max_length]
                thm_data = thm_data[: self.max_length]

        return {
            "tof": torch.FloatTensor(tof_data),
            "acc": torch.FloatTensor(acc_data),
            "rot": torch.FloatTensor(rot_data),
            "thm": torch.FloatTensor(thm_data),
            "label": torch.LongTensor([label])[0],
        }


class SequenceProcessor:
    """Process raw data into sequences for training."""

    def __init__(self, use_enhanced_features: bool = True):
        self.use_enhanced_features = use_enhanced_features
        self.feature_processor = FeatureProcessor() if use_enhanced_features else None

        # Define feature columns
        self.acc_cols = ["acc_x", "acc_y", "acc_z"]
        self.rot_cols = ["rot_w", "rot_x", "rot_y", "rot_z"]
        self.thm_cols = [f"thm_{i}" for i in range(1, 6)]
        self.tof_cols = [f"tof_{i}_v{j}" for i in range(1, 6) for j in range(64)]

    def process_dataframe(
        self,
        train_df: pl.DataFrame,
        label_encoder: LabelEncoder,
    ) -> list[dict]:
        """Process polars dataframe into sequences.

        Args:
            train_df: Training dataframe with gesture sequences
            label_encoder: Fitted label encoder for gesture classes

        Returns:
            List of sequence dictionaries
        """
        sequences = []
        sequence_lengths = []

        # Group by sequence_id and process each sequence
        grouped = train_df.group_by("sequence_id")

        for seq_id, group in tqdm(grouped, desc="Processing sequences"):
            try:
                gesture_id = group["gesture_id"][0]

                if self.use_enhanced_features:
                    # Create enhanced features using FeatureProcessor
                    enhanced_features = self.feature_processor.create_sequence_features(
                        group,
                    )

                    sequences.append(
                        {
                            "sequence_id": seq_id[0],
                            "enhanced_data": enhanced_features,
                            "label": gesture_id,
                        },
                    )
                    sequence_lengths.append(enhanced_features["sequence_length"])
                else:
                    # Use original features
                    seq_data = group.select(
                        self.acc_cols + self.rot_cols + self.thm_cols + self.tof_cols,
                    ).to_numpy()

                    sequences.append(
                        {
                            "sequence_id": seq_id[0],
                            "data": seq_data,
                            "label": gesture_id,
                        },
                    )
                    sequence_lengths.append(len(seq_data))

            except Exception as e:
                print(f"Error processing sequence {seq_id[0]}: {e}")
                # Fallback to original processing
                seq_data = group.select(
                    self.acc_cols + self.rot_cols + self.thm_cols + self.tof_cols,
                ).to_numpy()
                sequences.append(
                    {
                        "sequence_id": seq_id[0],
                        "data": seq_data,
                        "label": gesture_id,
                    },
                )
                sequence_lengths.append(len(seq_data))

        return sequences, sequence_lengths

    def create_train_val_split(
        self,
        sequences: list[dict],
        test_size: float = 0.2,
        random_state: int = 42,
    ) -> tuple:
        """Create stratified train/validation split.

        Args:
            sequences: List of processed sequences
            test_size: Fraction of data for validation
            random_state: Random seed for reproducibility

        Returns:
            Tuple of (train_sequences, val_sequences)
        """
        labels = [seq["label"] for seq in sequences]

        # Stratified train/validation split
        train_indices, val_indices = train_test_split(
            range(len(sequences)),
            test_size=test_size,
            random_state=random_state,
            stratify=labels,
        )

        train_sequences = [sequences[i] for i in train_indices]
        val_sequences = [sequences[i] for i in val_indices]

        return train_sequences, val_sequences


def prepare_gesture_labels(train_df: pl.DataFrame) -> tuple:
    """Prepare gesture labels and label encoder.

    Args:
        train_df: Training dataframe

    Returns:
        Tuple of (updated_dataframe, label_encoder, target_gestures, non_target_gestures)
    """
    # Define gesture classes
    target_gestures = [
        "Above ear - pull hair",
        "Cheek - pinch skin",
        "Eyebrow - pull hair",
        "Eyelash - pull hair",
        "Forehead - pull hairline",
        "Forehead - scratch",
        "Neck - pinch skin",
        "Neck - scratch",
    ]

    non_target_gestures = [
        "Text on phone",
        "Wave hello",
        "Write name in air",
        "Pull air toward your face",
        "Feel around in tray and pull out an object",
        "Glasses on/off",
        "Drink from bottle/cup",
        "Scratch knee/leg skin",
        "Write name on leg",
        "Pinch knee/leg skin",
    ]

    # Create label encoder
    label_encoder = LabelEncoder()
    train_df = train_df.with_columns(
        pl.Series(label_encoder.fit_transform(train_df["gesture"].to_numpy())).alias(
            "gesture_id",
        ),
    )

    return train_df, label_encoder, target_gestures, non_target_gestures


def get_enhanced_feature_dims(sequences: list[dict]) -> dict[str, int]:
    """Get feature dimensions from enhanced sequences.

    Args:
        sequences: List of sequences with enhanced features

    Returns:
        Dictionary of feature dimensions
    """
    if not sequences or "enhanced_data" not in sequences[0]:
        # Default dimensions for original features
        return {
            "tof": 320,
            "acc": 3,
            "rot": 4,
            "thm": 5,
        }

    sample_features = sequences[0]["enhanced_data"]
    return {
        "tof": sample_features["tof"].shape[1],
        "acc": sample_features["acc"].shape[1],
        "rot": sample_features["rot"].shape[1],
        "thm": sample_features["thm"].shape[1],
    }
