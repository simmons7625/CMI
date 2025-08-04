"""
Data loading and processing utilities for GNN kinematics
"""
import numpy as np
import pandas as pd
import polars as pl
from pathlib import Path

from .config import DATA_PATHS, GESTURE_CLASSES, DEMOGRAPHICS_FEATURES


def load_data():
    """Load training and test data"""
    print("Loading data for GNN training...")
    
    data = {}
    
    for key, path in DATA_PATHS.items():
        if path.exists():
            try:
                df = pl.read_csv(str(path))
                data[key] = df
                print(f"   ✓ {key}: {df.shape[0]} rows, {df.shape[1]} columns")
            except Exception as e:
                print(f"   ✗ {key}: Failed to load - {e}")
                data[key] = None
        else:
            print(f"   ⚠ {key}: File not found at {path}")
            data[key] = None
    
    return data


def create_sample_data_for_gnn():
    """Create sample data optimized for GNN training"""
    print("Creating sample data for GNN testing...")
    
    # Create more realistic synthetic data
    n_samples_per_seq = 50  # Shorter sequences for faster training
    n_sequences = 200  # More sequences for better variety
    
    sample_data = []
    
    for seq_idx in range(n_sequences):
        # Random gesture
        gesture = np.random.choice(GESTURE_CLASSES)
        subject_id = f'SUBJ_{seq_idx % 50:06d}'  # Reuse subjects
        
        # Generate gesture-specific patterns
        t = np.linspace(0, 2, n_samples_per_seq)  # 2 seconds at 25Hz
        
        # Create gesture-specific motion patterns
        if 'pull hair' in gesture:
            # Sharp, quick movements
            base_freq = 2.0
            amplitude = 1.5
        elif 'scratch' in gesture:
            # Repetitive, oscillatory
            base_freq = 3.0
            amplitude = 1.0
        elif 'Text on phone' in gesture:
            # Small, precise movements
            base_freq = 5.0
            amplitude = 0.3
        else:
            # Default pattern
            base_freq = 1.0
            amplitude = 0.8
        
        # Generate quaternion-based rotation data
        angle_x = amplitude * np.sin(2 * np.pi * base_freq * t) + 0.1 * np.random.randn(n_samples_per_seq)
        angle_y = amplitude * np.cos(2 * np.pi * base_freq * t) + 0.1 * np.random.randn(n_samples_per_seq)
        angle_z = 0.5 * amplitude * np.sin(4 * np.pi * base_freq * t) + 0.1 * np.random.randn(n_samples_per_seq)
        
        # Convert to quaternions
        rot_w = np.cos(np.sqrt(angle_x**2 + angle_y**2 + angle_z**2) / 2)
        rot_x = angle_x / (2 * np.sqrt(angle_x**2 + angle_y**2 + angle_z**2 + 1e-8))
        rot_y = angle_y / (2 * np.sqrt(angle_x**2 + angle_y**2 + angle_z**2 + 1e-8))
        rot_z = angle_z / (2 * np.sqrt(angle_x**2 + angle_y**2 + angle_z**2 + 1e-8))
        
        # Generate corresponding accelerometer data
        acc_x = np.gradient(np.gradient(angle_x)) + np.random.randn(n_samples_per_seq) * 0.1
        acc_y = np.gradient(np.gradient(angle_y)) + np.random.randn(n_samples_per_seq) * 0.1
        acc_z = np.gradient(np.gradient(angle_z)) + 9.81 + np.random.randn(n_samples_per_seq) * 0.1
        
        for i in range(n_samples_per_seq):
            sample_data.append({
                'sequence_id': f'SEQ_{seq_idx:06d}',
                'subject': subject_id,
                'gesture': gesture,
                'acc_x': acc_x[i],
                'acc_y': acc_y[i],
                'acc_z': acc_z[i],
                'rot_w': rot_w[i],
                'rot_x': rot_x[i],
                'rot_y': rot_y[i],
                'rot_z': rot_z[i],
            })
            
            # Add thermal and TOF data (simplified)
            for thm_idx in range(1, 6):
                sample_data[-1][f'thm_{thm_idx}'] = np.random.uniform(25, 35)
            
            for tof_idx in range(1, 6):
                for pixel in range(64):
                    sample_data[-1][f'tof_{tof_idx}_v{pixel}'] = np.random.choice(
                        [-1, np.random.uniform(0, 1000)], p=[0.1, 0.9]
                    )
    
    train_data = pl.DataFrame(sample_data)
    
    # Create demographics
    unique_subjects = train_data['subject'].unique().to_list()
    demographics_data = []
    
    for subject in unique_subjects:
        demographics_data.append({
            'subject': subject,
            'adult_child': np.random.choice([0, 1]),
            'age': np.random.randint(8, 65),
            'sex': np.random.choice([0, 1]),
            'handedness': np.random.choice([0, 1]),
            'height_cm': np.random.uniform(120, 190),
            'shoulder_to_wrist_cm': np.random.uniform(50, 80),
            'elbow_to_wrist_cm': np.random.uniform(20, 35)
        })
    
    train_demographics = pl.DataFrame(demographics_data)
    
    # Create test set (smaller)
    test_data = train_data.sample(500, seed=42)
    test_demographics = train_demographics.sample(20, seed=42)
    
    print(f"   ✓ Sample train data: {train_data.shape}")
    print(f"   ✓ Sample train demographics: {train_demographics.shape}")
    print(f"   ✓ Sample test data: {test_data.shape}")
    print(f"   ✓ Sample test demographics: {test_demographics.shape}")
    
    return {
        'train_data': train_data,
        'train_demographics': train_demographics,
        'test_data': test_data,
        'test_demographics': test_demographics
    }


def extract_demographics(demographics_data):
    """Extract demographics features"""
    if demographics_data is None or len(demographics_data) == 0:
        # Default demographics
        return [1.0, 25.0, 1.0, 1.0, 170.0, 60.0, 25.0]
    
    try:
        if hasattr(demographics_data, 'to_pandas'):
            demo_df = demographics_data.to_pandas()
        else:
            demo_df = demographics_data
        
        if len(demo_df) == 0:
            return [1.0, 25.0, 1.0, 1.0, 170.0, 60.0, 25.0]
        
        demo_values = []
        for feature in DEMOGRAPHICS_FEATURES:
            if feature in demo_df.columns:
                value = demo_df[feature].iloc[0]
                demo_values.append(float(value) if pd.notna(value) else 0.0)
            else:
                demo_values.append(0.0)
        
        return demo_values
        
    except Exception as e:
        print(f"Demographics extraction failed: {e}")
        return [1.0, 25.0, 1.0, 1.0, 170.0, 60.0, 25.0]


def prepare_sequence_data(sequence_data, demographics_data):
    """Prepare sequence and demographics data for GNN input"""
    # Extract demographics
    demo_values = extract_demographics(demographics_data)
    
    # Convert sequence to pandas if needed
    if hasattr(sequence_data, 'to_pandas'):
        seq_df = sequence_data.to_pandas()
    else:
        seq_df = sequence_data.copy()
    
    return seq_df, demo_values