from __future__ import annotations

import numpy as np
import polars as pl
import torch
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
    ):
        """Initialize CMI dataset.

        Args:
            sequences: List of sequence dictionaries containing data and labels
            max_length: Maximum sequence length for padding/truncation
            use_enhanced_features: Whether to use enhanced feature processing
        """
        self.sequences = sequences

        # Calculate max length if not provided
        if max_length is None:
            self.max_length = max(
                len(seq["enhanced_data"]["tof"])
                if "enhanced_data" in seq
                else len(seq["data"])
                for seq in sequences
            )
        else:
            self.max_length = max_length

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        sequence = self.sequences[idx]
        label = sequence.get("label", -1)  # Default to -1 if label is missing

        if "enhanced_data" in sequence:
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

        # Get chunk start index for positional encoding
        chunk_start_idx = sequence.get("chunk_start_idx", 0)

        return {
            "sequence_id": sequence["sequence_id"],
            "tof": torch.FloatTensor(tof_data),  # (seq_len, 320)
            "acc": torch.FloatTensor(acc_data),  # (seq_len, 3)
            "rot": torch.FloatTensor(rot_data),  # (seq_len, 4)
            "thm": torch.FloatTensor(thm_data),  # (seq_len, 5)
            "label": torch.LongTensor([label])[0],  # scalar
            "chunk_start_idx": torch.LongTensor([chunk_start_idx])[0],  # scalar
        }


class SequenceProcessor:
    """Process raw data into sequences for training."""

    def __init__(self):
        self.feature_processor = FeatureProcessor()

        # Define feature columns
        self.acc_cols = ["acc_x", "acc_y", "acc_z"]
        self.rot_cols = ["rot_w", "rot_x", "rot_y", "rot_z"]
        self.thm_cols = [f"thm_{i}" for i in range(1, 6)]
        self.tof_cols = [f"tof_{i}_v{j}" for i in range(1, 6) for j in range(64)]

    def process_dataframe(
        self,
        df: pl.DataFrame,
        num_samples: int | None = None,
        max_seq_length: int | None = None,
    ) -> list[dict]:
        """Process polars dataframe into sequences.

        Args:
            df: Training dataframe with gesture sequences
            num_samples: Maximum number of samples to process (None for all)
            max_seq_length: Maximum sequence length for chunking (None for no chunking)

        Returns:
            List of sequence dictionaries
        """
        sequences = []

        # Group by sequence_id and process each sequence
        grouped = df.group_by("sequence_id")

        # Limit number of sequences if num_samples is specified
        if num_samples is not None:
            grouped = list(grouped)[:num_samples]

        for seq_id, group in tqdm(grouped, desc="Processing sequences"):
            try:
                if "gesture_id" not in group.columns:
                    gesture_id = None
                else:
                    gesture_id = group["gesture_id"][0]

                # Create enhanced features using FeatureProcessor
                enhanced_features = self.feature_processor.create_sequence_features(
                    group,
                )

                # Apply chunking if max_seq_length is specified
                if max_seq_length is not None:
                    chunked_sequences = self._chunk_sequence(
                        enhanced_features,
                        gesture_id,
                        seq_id[0],
                        max_seq_length,
                    )
                    sequences.extend(chunked_sequences)
                else:
                    sequences.append(
                        {
                            "sequence_id": seq_id[0],
                            "enhanced_data": enhanced_features,
                            "label": gesture_id,
                        },
                    )

            except Exception as e:
                print(f"Error processing sequence {seq_id[0]}: {e}")
                # Fallback to original processing
                seq_data = group.select(
                    self.acc_cols + self.rot_cols + self.thm_cols + self.tof_cols,
                ).to_numpy()

                # Apply chunking to fallback data if max_seq_length is specified
                if max_seq_length is not None:
                    chunked_sequences = self._chunk_original_sequence(
                        seq_data,
                        gesture_id,
                        seq_id[0],
                        max_seq_length,
                    )
                    sequences.extend(chunked_sequences)
                else:
                    sequences.append(
                        {
                            "sequence_id": seq_id[0],
                            "data": seq_data,
                            "label": gesture_id,
                        },
                    )

        return sequences

    def _chunk_sequence(
        self,
        enhanced_features: dict,
        gesture_id: int,
        sequence_id: str,
        max_seq_length: int,
    ) -> list[dict]:
        """Chunk enhanced features into smaller sequences."""
        tof_data = enhanced_features["tof"]
        acc_data = enhanced_features["acc"]
        rot_data = enhanced_features["rot"]
        thm_data = enhanced_features["thm"]

        seq_length = len(tof_data)
        chunks = []

        if seq_length <= max_seq_length:
            # No chunking needed
            chunks.append(
                {
                    "sequence_id": sequence_id,
                    "enhanced_data": enhanced_features,
                    "label": gesture_id,
                    "chunk_start_idx": 0,
                },
            )
        else:
            # Split into chunks
            num_chunks = (seq_length + max_seq_length - 1) // max_seq_length
            for i in range(num_chunks):
                start_idx = i * max_seq_length
                end_idx = min((i + 1) * max_seq_length, seq_length)

                chunk_features = {
                    "tof": tof_data[start_idx:end_idx],
                    "acc": acc_data[start_idx:end_idx],
                    "rot": rot_data[start_idx:end_idx],
                    "thm": thm_data[start_idx:end_idx],
                }

                chunks.append(
                    {
                        "sequence_id": f"{sequence_id}",
                        "enhanced_data": chunk_features,
                        "label": gesture_id,
                        "chunk_start_idx": start_idx,
                    },
                )

        return chunks

    def _chunk_original_sequence(
        self,
        seq_data: np.ndarray,
        gesture_id: int,
        sequence_id: str,
        max_seq_length: int,
    ) -> list[dict]:
        """Chunk original sequence data into smaller sequences."""
        seq_length = len(seq_data)
        chunks = []

        if seq_length <= max_seq_length:
            # No chunking needed
            chunks.append(
                {
                    "sequence_id": sequence_id,
                    "data": seq_data,
                    "label": gesture_id,
                    "chunk_start_idx": 0,
                },
            )
        else:
            # Split into chunks
            num_chunks = (seq_length + max_seq_length - 1) // max_seq_length
            for i in range(num_chunks):
                start_idx = i * max_seq_length
                end_idx = min((i + 1) * max_seq_length, seq_length)

                chunks.append(
                    {
                        "sequence_id": f"{sequence_id}",
                        "data": seq_data[start_idx:end_idx],
                        "label": gesture_id,
                        "chunk_start_idx": start_idx,
                    },
                )

        return chunks

    def create_train_val_split(
        self,
        sequences: list[dict],
        test_size: float = 0.2,
        random_state: int = 42,
    ) -> tuple:
        """Create stratified train/validation split with equal split ratios per label.

        Args:
            sequences: List of processed sequences
            test_size: Fraction of data for validation
            random_state: Random seed for reproducibility

        Returns:
            Tuple of (train_sequences, val_sequences)
        """
        return stratified_split_equal_ratio(sequences, test_size, random_state)


def stratified_split_equal_ratio(
    sequences: list[dict],
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[list[dict], list[dict]]:
    """Create stratified train/validation split with equal split ratios per label.

    This function ensures that each label is split with exactly the same ratio,
    avoiding the imbalanced splits that can occur with sklearn's train_test_split.

    Args:
        sequences: List of processed sequences
        test_size: Fraction of data for validation (0.0 to 1.0)
        random_state: Random seed for reproducibility

    Returns:
        Tuple of (train_sequences, val_sequences)
    """
    np.random.seed(random_state)

    # Group sequences by label
    label_groups = {}
    for i, seq in enumerate(sequences):
        label = seq["label"]
        if label not in label_groups:
            label_groups[label] = []
        label_groups[label].append(i)

    train_indices = []
    val_indices = []

    # Split each label group with the exact same ratio
    for label, indices in label_groups.items():
        # Shuffle indices for this label
        shuffled_indices = np.array(indices)
        np.random.shuffle(shuffled_indices)

        # Calculate split point
        n_samples = len(shuffled_indices)
        n_val = int(np.round(n_samples * test_size))
        n_train = n_samples - n_val

        # Ensure we have at least one sample in each split if possible
        if n_samples >= 2:
            n_val = max(1, n_val)
            n_train = n_samples - n_val

        # Split indices
        val_indices.extend(shuffled_indices[:n_val].tolist())
        train_indices.extend(shuffled_indices[n_val:].tolist())

        print(
            f"Label {label}: {n_train} train, {n_val} val ({n_val/(n_train+n_val)*100:.1f}% val)",
        )

    # Create train and validation sequences
    train_sequences = [sequences[i] for i in train_indices]
    val_sequences = [sequences[i] for i in val_indices]

    return train_sequences, val_sequences


def prepare_gesture_labels(df: pl.DataFrame) -> tuple:
    """Prepare gesture labels and label encoder.

    Args:
        df: Training dataframe

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
    df = df.with_columns(
        pl.Series(label_encoder.fit_transform(df["gesture"].to_numpy())).alias(
            "gesture_id",
        ),
    )

    return df, label_encoder, target_gestures, non_target_gestures


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
