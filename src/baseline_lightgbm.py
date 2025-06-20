"""
LightGBM Baseline Model for CMI Detect Behavior with Sensor Data
Creates statistical features from time series sensor data for gesture classification
"""

import pandas as pd
import polars as pl
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')

class CMIFeatureExtractor:
    """Extract statistical features from time series sensor data"""
    
    def __init__(self):
        # Define sensor groups for feature extraction
        self.sensor_groups = {
            'acc': ['acc_x', 'acc_y', 'acc_z'],
            'rot': ['rot_w', 'rot_x', 'rot_y', 'rot_z'],
            'thm': [f'thm_{i}' for i in range(1, 6)],
            'tof': [f'tof_{i}_v{j}' for i in range(1, 6) for j in range(64)]
        }
        
    def extract_statistical_features(self, df_group):
        """Extract statistical features from a sequence group"""
        features = {}
        
        for sensor_type, columns in self.sensor_groups.items():
            # Get available columns (some might be missing)
            available_cols = [col for col in columns if col in df_group.columns]
            
            if not available_cols:
                continue
                
            # Get sensor data, replace -1.0 with NaN for proper statistics
            sensor_data = df_group[available_cols].replace(-1.0, np.nan)
            
            # Basic statistics
            features[f'{sensor_type}_mean'] = sensor_data.mean().mean()
            features[f'{sensor_type}_std'] = sensor_data.std().mean()
            features[f'{sensor_type}_min'] = sensor_data.min().min()
            features[f'{sensor_type}_max'] = sensor_data.max().max()
            features[f'{sensor_type}_median'] = sensor_data.median().mean()
            features[f'{sensor_type}_range'] = features[f'{sensor_type}_max'] - features[f'{sensor_type}_min']
            
            # Advanced statistics
            features[f'{sensor_type}_skew'] = sensor_data.skew().mean()
            features[f'{sensor_type}_kurtosis'] = sensor_data.kurtosis().mean()
            features[f'{sensor_type}_var'] = sensor_data.var().mean()
            
            # Percentiles
            features[f'{sensor_type}_q25'] = sensor_data.quantile(0.25).mean()
            features[f'{sensor_type}_q75'] = sensor_data.quantile(0.75).mean()
            features[f'{sensor_type}_iqr'] = features[f'{sensor_type}_q75'] - features[f'{sensor_type}_q25']
            
            # Signal properties
            features[f'{sensor_type}_missing_ratio'] = (sensor_data.isna().sum().sum() / sensor_data.size)
            features[f'{sensor_type}_zero_crossing'] = self._count_zero_crossings(sensor_data)
            
            # Energy and RMS
            features[f'{sensor_type}_energy'] = (sensor_data**2).sum().sum()
            features[f'{sensor_type}_rms'] = np.sqrt((sensor_data**2).mean().mean())
            
        return features
    
    def _count_zero_crossings(self, data):
        """Count zero crossings in the signal"""
        try:
            # For multivariate data, count zero crossings per column then average
            zero_crossings = []
            for col in data.columns:
                series = data[col].dropna()
                if len(series) > 1:
                    crossings = ((series[:-1] * series[1:].values) < 0).sum()
                    zero_crossings.append(crossings)
            return np.mean(zero_crossings) if zero_crossings else 0
        except:
            return 0
    
    def extract_sequence_features(self, df_sequence):
        """Extract sequence-level features"""
        features = {}
        
        # Sequence length and timing
        features['sequence_length'] = len(df_sequence)
        features['sequence_duration'] = df_sequence['sequence_counter'].max() - df_sequence['sequence_counter'].min()
        
        # Phase information
        if 'phase' in df_sequence.columns:
            features['num_phases'] = df_sequence['phase'].nunique()
        
        return features
    
    def transform(self, df):
        """Transform DataFrame to feature matrix"""
        features_list = []
        
        # Group by sequence_id to extract features per sequence
        for sequence_id, group in df.groupby('sequence_id'):
            # Extract statistical features
            stat_features = self.extract_statistical_features(group)
            
            # Extract sequence features
            seq_features = self.extract_sequence_features(group)
            
            # Combine all features
            all_features = {**stat_features, **seq_features}
            all_features['sequence_id'] = sequence_id
            
            # Add target and metadata
            all_features['gesture'] = group['gesture'].iloc[0]
            all_features['subject'] = group['subject'].iloc[0]
            
            features_list.append(all_features)
        
        return pd.DataFrame(features_list)

class CMILightGBMBaseline:
    """LightGBM baseline model for CMI competition"""
    
    def __init__(self):
        self.feature_extractor = CMIFeatureExtractor()
        self.label_encoder = LabelEncoder()
        self.models = []
        self.feature_names = None
        
        # LightGBM parameters
        self.lgb_params = {
            'objective': 'multiclass',
            'num_class': 18,
            'metric': 'multi_logloss',
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'feature_fraction': 0.9,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbose': -1,
            'random_state': 42
        }
    
    def load_data(self):
        """Load training and test data"""
        print("Loading data...")
        
        # Load training data
        self.train_df = pd.read_csv('data/raw/train.csv')
        self.train_demographics = pd.read_csv('data/raw/train_demographics.csv')
        
        # Load test data
        self.test_df = pd.read_csv('data/raw/test.csv')
        self.test_demographics = pd.read_csv('data/raw/test_demographics.csv')
        
        print(f"Training data shape: {self.train_df.shape}")
        print(f"Test data shape: {self.test_df.shape}")
        print(f"Unique gestures: {self.train_df['gesture'].nunique()}")
        print(f"Unique sequences in train: {self.train_df['sequence_id'].nunique()}")
        
    def prepare_features(self):
        """Extract features from raw data"""
        print("Extracting features from training data...")
        self.train_features = self.feature_extractor.transform(self.train_df)
        
        print("Extracting features from test data...")
        self.test_features = self.feature_extractor.transform(self.test_df)
        
        # Merge with demographics
        self.train_features = self.train_features.merge(
            self.train_demographics, on='subject', how='left'
        )
        self.test_features = self.test_features.merge(
            self.test_demographics, on='subject', how='left'
        )
        
        print(f"Training features shape: {self.train_features.shape}")
        print(f"Test features shape: {self.test_features.shape}")
        
        # Prepare feature matrix
        exclude_cols = ['sequence_id', 'gesture', 'subject']
        self.feature_names = [col for col in self.train_features.columns if col not in exclude_cols]
        
        self.X_train = self.train_features[self.feature_names].fillna(0)
        self.y_train = self.label_encoder.fit_transform(self.train_features['gesture'])
        self.X_test = self.test_features[self.feature_names].fillna(0)
        
        print(f"Feature matrix shape: {self.X_train.shape}")
        print(f"Number of classes: {len(self.label_encoder.classes_)}")
        
    def train_model(self, n_folds=5):
        """Train LightGBM model with cross-validation"""
        print(f"Training LightGBM model with {n_folds}-fold CV...")
        
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
        oof_predictions = np.zeros(len(self.X_train))
        test_predictions = np.zeros((len(self.X_test), len(self.label_encoder.classes_)))
        
        feature_importance = pd.DataFrame()
        
        for fold, (train_idx, val_idx) in enumerate(skf.split(self.X_train, self.y_train)):
            print(f"Training fold {fold + 1}/{n_folds}")
            
            X_fold_train, X_fold_val = self.X_train.iloc[train_idx], self.X_train.iloc[val_idx]
            y_fold_train, y_fold_val = self.y_train[train_idx], self.y_train[val_idx]
            
            # Create LightGBM datasets
            train_data = lgb.Dataset(X_fold_train, label=y_fold_train)
            val_data = lgb.Dataset(X_fold_val, label=y_fold_val, reference=train_data)
            
            # Train model
            model = lgb.train(
                self.lgb_params,
                train_data,
                valid_sets=[train_data, val_data],
                num_boost_round=1000,
                valid_names=['train', 'val'],
                callbacks=[lgb.early_stopping(50), lgb.log_evaluation(100)]
            )
            
            # Predictions
            oof_predictions[val_idx] = model.predict(X_fold_val).argmax(axis=1)
            test_predictions += model.predict(self.X_test) / n_folds
            
            # Feature importance
            fold_importance = pd.DataFrame()
            fold_importance['feature'] = self.feature_names
            fold_importance['importance'] = model.feature_importance()
            fold_importance['fold'] = fold
            feature_importance = pd.concat([feature_importance, fold_importance])
            
            self.models.append(model)
        
        # Calculate CV score
        cv_score = accuracy_score(self.y_train, oof_predictions)
        print(f"Cross-validation accuracy: {cv_score:.4f}")
        
        # Feature importance summary
        self.feature_importance = feature_importance.groupby('feature')['importance'].mean().sort_values(ascending=False)
        print(f"\nTop 10 important features:")
        print(self.feature_importance.head(10))
        
        # Final test predictions
        self.test_pred_classes = test_predictions.argmax(axis=1)
        self.test_pred_proba = test_predictions
        
        return cv_score
    
    def create_submission(self, filename='submission.csv'):
        """Create submission file"""
        # Convert predictions back to gesture names
        predicted_gestures = self.label_encoder.inverse_transform(self.test_pred_classes)
        
        # Create submission DataFrame
        submission = pd.DataFrame({
            'sequence_id': self.test_features['sequence_id'],
            'gesture': predicted_gestures
        })
        
        submission.to_csv(filename, index=False)
        print(f"Submission saved to {filename}")
        print(f"Submission shape: {submission.shape}")
        print("\nPrediction distribution:")
        print(submission['gesture'].value_counts())
        
        return submission
    
    def run_baseline(self):
        """Run complete baseline pipeline"""
        print("=" * 50)
        print("CMI LightGBM Baseline Model")
        print("=" * 50)
        
        # Load data
        self.load_data()
        
        # Prepare features
        self.prepare_features()
        
        # Train model
        cv_score = self.train_model()
        
        # Create submission
        submission = self.create_submission('results/lightgbm_baseline_submission.csv')
        
        print("=" * 50)
        print("Baseline Complete!")
        print(f"Final CV Score: {cv_score:.4f}")
        print("=" * 50)
        
        return cv_score, submission

if __name__ == "__main__":
    # Create results directory
    import os
    os.makedirs('results', exist_ok=True)
    
    # Run baseline model
    baseline = CMILightGBMBaseline()
    cv_score, submission = baseline.run_baseline()