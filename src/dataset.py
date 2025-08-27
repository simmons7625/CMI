from __future__ import annotations

import numpy as np
import polars as pl
import torch
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset

from .feature_processor import FeatureProcessor


class CMIDataset(Dataset):
    """Dataset class for CMI gesture recognition data with chunking and augmentation support."""

    def __init__(
        self,
        sequences: list[dict],
        chunk_size: int = 100,
        use_chunking: bool = True,
        augmentation_config: dict | None = None,
    ):
        """Initialize CMI dataset.

        Args:
            sequences: List of sequence dictionaries containing data and labels
            chunk_size: Size of chunks to create from sequences
            use_chunking: Whether to split sequences into chunks
            augmentation_config: Configuration for data augmentation
        """
        self.sequences = sequences
        self.chunk_size = chunk_size
        self.use_chunking = use_chunking
        self.augmentation_config = augmentation_config or {}

        # Create chunks if chunking is enabled
        if self.use_chunking:
            self.chunks = self._create_chunks()
            print(
                f"📦 Created {len(self.chunks)} chunks from {len(sequences)} sequences (chunk_size={chunk_size})",
            )
        else:
            # Calculate max_length for padding
            sequence_lengths = []
            for seq in sequences:
                if "enhanced_data" in seq:
                    seq_len = len(seq["enhanced_data"]["tof"])
                else:
                    seq_len = len(seq["data"])
                sequence_lengths.append(seq_len)

            self.max_length = max(sequence_lengths) if sequence_lengths else 100
            print(
                f"📏 Auto-calculated max_length: {self.max_length} (from {len(sequence_lengths)} sequences)",
            )

    def __len__(self) -> int:
        return len(self.chunks) if self.use_chunking else len(self.sequences)

    def _create_chunks(self) -> list[dict]:
        """Create chunks from sequences for chunk-wise training."""
        chunks = []

        for sequence in self.sequences:
            label = sequence.get("label", -1)
            sequence_id = sequence["sequence_id"]

            if "enhanced_data" in sequence:
                # Use enhanced features
                tof_data = sequence["enhanced_data"]["tof"]
                acc_data = sequence["enhanced_data"]["acc"]
                rot_data = sequence["enhanced_data"]["rot"]
                thm_data = sequence["enhanced_data"]["thm"]
            else:
                # Use original features
                data = sequence["data"]
                tof_data = data[:, :320]  # ToF features (320)
                acc_data = data[:, 320:323]  # Accelerometer (3)
                rot_data = data[:, 323:327]  # Rotation (4)
                thm_data = data[:, 327:332]  # Thermal (5)

            # Create overlapping chunks
            seq_len = len(tof_data)
            if seq_len <= self.chunk_size:
                # If sequence is shorter than chunk_size, use as single chunk
                chunks.append(
                    {
                        "sequence_id": sequence_id,
                        "tof": tof_data,
                        "acc": acc_data,
                        "rot": rot_data,
                        "thm": thm_data,
                        "label": label,
                        "chunk_idx": 0,
                        "total_chunks": 1,
                    },
                )
            else:
                # Create overlapping chunks with 50% overlap
                step = self.chunk_size // 2
                chunk_idx = 0
                for start in range(0, seq_len - self.chunk_size + 1, step):
                    end = start + self.chunk_size
                    chunks.append(
                        {
                            "sequence_id": sequence_id,
                            "tof": tof_data[start:end],
                            "acc": acc_data[start:end],
                            "rot": rot_data[start:end],
                            "thm": thm_data[start:end],
                            "label": label,
                            "chunk_idx": chunk_idx,
                            "total_chunks": (seq_len - self.chunk_size) // step + 1,
                        },
                    )
                    chunk_idx += 1

        return chunks

    def _apply_augmentation(
        self,
        tof_data: np.ndarray,
        acc_data: np.ndarray,
        rot_data: np.ndarray,
        thm_data: np.ndarray,
    ) -> tuple:
        """Apply Gaussian noise augmentation to sensor data.

        Gaussian noise with mu = source_value, sigma = hyperparameter
        """
        if not self.augmentation_config.get("enabled", False):
            return tof_data, acc_data, rot_data, thm_data

        # Gaussian noise augmentation
        if self.augmentation_config.get("gaussian_noise", {}).get("enabled", False):
            noise_sigma = self.augmentation_config["gaussian_noise"].get("sigma", 0.01)
            noise_prob = self.augmentation_config["gaussian_noise"].get("prob", 0.5)

            # Apply noise to each modality with specified probability
            if np.random.rand() < noise_prob:
                # mu = source_value, sigma = hyperparameter
                tof_noise = np.random.normal(loc=tof_data, scale=noise_sigma)
                tof_data = tof_noise

            if np.random.rand() < noise_prob:
                acc_noise = np.random.normal(loc=acc_data, scale=noise_sigma)
                acc_data = acc_noise

            if np.random.rand() < noise_prob:
                rot_noise = np.random.normal(loc=rot_data, scale=noise_sigma)
                rot_data = rot_noise

            if np.random.rand() < noise_prob:
                thm_noise = np.random.normal(loc=thm_data, scale=noise_sigma)
                thm_data = thm_noise

        return tof_data, acc_data, rot_data, thm_data

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        if self.use_chunking:
            # Return chunk
            chunk = self.chunks[idx]

            tof_data = chunk["tof"]
            acc_data = chunk["acc"]
            rot_data = chunk["rot"]
            thm_data = chunk["thm"]

            # Apply augmentation
            tof_data, acc_data, rot_data, thm_data = self._apply_augmentation(
                tof_data,
                acc_data,
                rot_data,
                thm_data,
            )

            return {
                "sequence_id": chunk["sequence_id"],
                "tof": torch.FloatTensor(tof_data),
                "acc": torch.FloatTensor(acc_data),
                "rot": torch.FloatTensor(rot_data),
                "thm": torch.FloatTensor(thm_data),
                "label": torch.LongTensor([chunk["label"]])[0],
                "chunk_idx": chunk["chunk_idx"],
                "total_chunks": chunk["total_chunks"],
            }
        # Return full sequence (legacy mode)
        sequence = self.sequences[idx]
        label = sequence.get("label", -1)
        chunk_size = sequence.get("chunk_size", 100)

        if "enhanced_data" in sequence:
            # Use enhanced features
            tof_data = sequence["enhanced_data"]["tof"]
            acc_data = sequence["enhanced_data"]["acc"]
            rot_data = sequence["enhanced_data"]["rot"]
            thm_data = sequence["enhanced_data"]["thm"]
        else:
            # Use original features
            data = sequence["data"]
            tof_data = data[:, :320]  # ToF features (320)
            acc_data = data[:, 320:323]  # Accelerometer (3)
            rot_data = data[:, 323:327]  # Rotation (4)
            thm_data = data[:, 327:332]  # Thermal (5)

        # Apply augmentation
        tof_data, acc_data, rot_data, thm_data = self._apply_augmentation(
            tof_data,
            acc_data,
            rot_data,
            thm_data,
        )

        # Pad sequences to max_length for standard batching
        seq_len = len(tof_data)
        if seq_len < self.max_length:
            # Pad with -1.0 (consistent with missing data convention)
            pad_shape_tof = (self.max_length - seq_len, tof_data.shape[1])
            pad_shape_acc = (self.max_length - seq_len, acc_data.shape[1])
            pad_shape_rot = (self.max_length - seq_len, rot_data.shape[1])
            pad_shape_thm = (self.max_length - seq_len, thm_data.shape[1])

            tof_padding = np.full(pad_shape_tof, -1.0)
            acc_padding = np.full(pad_shape_acc, -1.0)
            rot_padding = np.full(pad_shape_rot, -1.0)
            thm_padding = np.full(pad_shape_thm, -1.0)

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
            seq_len = self.max_length

        return {
            "sequence_id": sequence["sequence_id"],
            "tof": torch.FloatTensor(tof_data),
            "acc": torch.FloatTensor(acc_data),
            "rot": torch.FloatTensor(rot_data),
            "thm": torch.FloatTensor(thm_data),
            "label": torch.LongTensor([label])[0],
            "chunk_size": chunk_size,
            "actual_length": seq_len,
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
        chunk_size: int | None = None,
    ) -> list[dict]:
        """Process polars dataframe into sequences.

        Args:
            df: Training dataframe with gesture sequences
            num_samples: Maximum number of samples to process (None for all)
            chunk_size: Chunk size for model internal chunking (not used for dataset chunking)

        Returns:
            List of sequence dictionaries (full sequences, not chunked)
        """
        sequences = []

        # Group by sequence_id and process each sequence
        grouped = df.group_by("sequence_id")

        # Limit number of sequences if num_samples is specified
        if num_samples is not None:
            grouped = list(grouped)[:num_samples]

        for seq_id, group in grouped:
            try:
                if "gesture_id" not in group.columns:
                    gesture_id = -1
                else:
                    gesture_id = group["gesture_id"][0]

                # Create enhanced features using FeatureProcessor
                enhanced_features = self.feature_processor.create_sequence_features(
                    group,
                )

                # Store full sequence (no dataset-level chunking)
                sequences.append(
                    {
                        "sequence_id": seq_id[0],
                        "enhanced_data": enhanced_features,
                        "label": gesture_id,
                        "chunk_size": chunk_size
                        or 100,  # Store chunk size for model use
                    },
                )

            except Exception as e:
                print(f"Error processing sequence {seq_id[0]}: {e}")
                # Fallback to original processing
                seq_data = group.select(
                    self.acc_cols + self.rot_cols + self.thm_cols + self.tof_cols,
                ).to_numpy()

                # Store full sequence (no dataset-level chunking)
                sequences.append(
                    {
                        "sequence_id": seq_id[0],
                        "data": seq_data,
                        "label": gesture_id,
                        "chunk_size": chunk_size
                        or 100,  # Store chunk size for model use
                    },
                )

        return sequences

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


def get_dataset_stats(sequences: list[dict]) -> dict:
    """Get dataset statistics including feature dimensions and sequence lengths.

    Args:
        sequences: List of sequences with enhanced features

    Returns:
        Dictionary containing feature dimensions and sequence length statistics
    """
    if not sequences:
        return {
            "feature_dims": {"tof": 320, "acc": 3, "rot": 4, "thm": 5},
            "max_length": 100,
            "min_length": 1,
            "avg_length": 50,
        }

    # Get feature dimensions
    if "enhanced_data" in sequences[0]:
        sample_features = sequences[0]["enhanced_data"]
        feature_dims = {
            "tof": sample_features["tof"].shape[1],
            "acc": sample_features["acc"].shape[1],
            "rot": sample_features["rot"].shape[1],
            "thm": sample_features["thm"].shape[1],
        }
    else:
        feature_dims = {"tof": 320, "acc": 3, "rot": 4, "thm": 5}

    # Calculate sequence length statistics
    sequence_lengths = []
    for seq in sequences:
        if "enhanced_data" in seq:
            seq_len = len(seq["enhanced_data"]["tof"])
        else:
            seq_len = len(seq["data"])
        sequence_lengths.append(seq_len)

    return {
        "feature_dims": feature_dims,
        "max_length": max(sequence_lengths),
        "min_length": min(sequence_lengths),
        "avg_length": sum(sequence_lengths) / len(sequence_lengths),
        "total_sequences": len(sequences),
    }


def get_enhanced_feature_dims(sequences: list[dict]) -> dict[str, int]:
    """Get feature dimensions from enhanced sequences (backward compatibility).

    Args:
        sequences: List of sequences with enhanced features

    Returns:
        Dictionary of feature dimensions
    """
    stats = get_dataset_stats(sequences)
    return stats["feature_dims"]
