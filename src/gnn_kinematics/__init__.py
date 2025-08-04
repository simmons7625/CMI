"""
GNN Kinematics Module for CMI Gesture Classification

This module implements a novel Graph Neural Network approach for gesture classification
based on kinematic modeling using virtual shoulder-elbow-wrist chains.
"""

from .config import (
    TRAINING_CONFIG,
    DATA_CONFIG,
    KINEMATIC_CONFIG,
    GESTURE_CLASSES,
    DEMOGRAPHICS_FEATURES,
    DEVICE
)

from .model import VirtualKinematicChain

from .feature_engineering import (
    remove_gravity_from_acc,
    calculate_angular_velocity_from_quat,
    calculate_angular_distance,
    extract_sequence_features
)

from .data_utils import (
    load_data,
    create_sample_data_for_gnn,
    extract_demographics,
    prepare_sequence_data
)

# Optional imports (if modules exist)
try:
    from .visualization import (
        plot_training_history,
        visualize_gesture_generation,
        visualize_gesture_comparison,
        plot_gesture_distribution,
        plot_demographics_distribution
    )
except ImportError:
    plot_training_history = None
    visualize_gesture_generation = None
    visualize_gesture_comparison = None
    plot_gesture_distribution = None
    plot_demographics_distribution = None

try:
    from .classifier import GNNGestureClassifier
except ImportError:
    GNNGestureClassifier = None

__version__ = "1.0.0"
__author__ = "CMI Team"

__all__ = [
    # Core model
    'VirtualKinematicChain',
    'GNNGestureClassifier',
    
    # Configuration
    'TRAINING_CONFIG',
    'DATA_CONFIG', 
    'KINEMATIC_CONFIG',
    'GESTURE_CLASSES',
    'DEMOGRAPHICS_FEATURES',
    'DEVICE',
    
    # Feature engineering
    'remove_gravity_from_acc',
    'calculate_angular_velocity_from_quat',
    'calculate_angular_distance',
    'extract_sequence_features',
    
    # Data utilities
    'load_data',
    'create_sample_data_for_gnn',
    'extract_demographics',
    'prepare_sequence_data',
    
    # Visualization
    'plot_training_history',
    'visualize_gesture_generation',
    'visualize_gesture_comparison',
    'plot_gesture_distribution',
    'plot_demographics_distribution',
]