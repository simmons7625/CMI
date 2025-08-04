"""
Configuration and constants for GNN Kinematics Model
"""
from pathlib import Path
import torch
import yaml

def load_config():
    """Load configuration from YAML file"""
    config_path = Path(__file__).parent / "config.yaml"
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

# Load configuration
_config = load_config()

# Device configuration
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Training configuration
TRAINING_CONFIG = {
    'TRAIN_MODE': _config['training']['train_mode'],
    'USE_LOCAL_DATA': _config['training']['use_local_data'],
    'RANDOM_SEED': _config['training']['random_seed'],
    'BATCH_SIZE': _config['training']['batch_size'],
    'LEARNING_RATE': _config['training']['learning_rate'],
    'N_EPOCHS': _config['training']['n_epochs'],
    'PATIENCE': _config['training']['patience'],
    'VALIDATION_SPLIT': _config['training']['validation_split'],
}

# Data configuration
DATA_CONFIG = {
    'SAMPLE_RATE': _config['data']['sample_rate'],
    'MAX_SEQUENCE_LENGTH': _config['data']['max_sequence_length'],
    'MIN_SEQUENCE_LENGTH': _config['data']['min_sequence_length'],
}

# Data paths
DATA_DIR = Path(_config['paths']['data_dir'])
MODELS_DIR = Path(_config['paths']['models_dir'])
OUTPUT_DIR = Path(_config['paths']['output_dir'])
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DATA_PATHS = {
    'train_data': Path(_config['paths']['train_data']),
    'train_demographics': Path(_config['paths']['train_demographics']),
    'test_data': Path(_config['paths']['test_data']),
    'test_demographics': Path(_config['paths']['test_demographics'])
}

# Gesture classes
GESTURE_CLASSES = _config['gesture_classes']

# Demographics features
DEMOGRAPHICS_FEATURES = _config['demographics_features']

# Kinematic model parameters
KINEMATIC_CONFIG = {
    'n_joints': _config['kinematic']['n_joints'],
    'hidden_dim': _config['kinematic']['hidden_dim'],
    'gnn_layers': _config['kinematic']['gnn_layers'],
    'attention_heads': _config['kinematic']['attention_heads'],
    'dropout': _config['kinematic']['dropout'],
    'joint_dof': _config['kinematic']['joint_dof'],
    'physics_weight': _config['kinematic']['physics_weight'],
}

# Model architecture constants
MODEL_CONFIG = {
    'n_classes': len(GESTURE_CLASSES),
    'demo_features_dim': len(DEMOGRAPHICS_FEATURES),
}