"""
Feature engineering functions for sensor data processing
"""
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation as R
from .config import DATA_CONFIG


def remove_gravity_from_acc(acc_data, rot_data):
    """Remove gravity component from accelerometer data using quaternion rotation"""
    if isinstance(acc_data, pd.DataFrame):
        acc_values = acc_data[['acc_x', 'acc_y', 'acc_z']].values
    else:
        acc_values = acc_data

    if isinstance(rot_data, pd.DataFrame):
        quat_values = rot_data[['rot_x', 'rot_y', 'rot_z', 'rot_w']].values
    else:
        quat_values = rot_data

    num_samples = acc_values.shape[0]
    linear_accel = np.zeros_like(acc_values)
    gravity_world = np.array([0, 0, 9.81])

    for i in range(num_samples):
        if np.all(np.isnan(quat_values[i])) or np.all(np.isclose(quat_values[i], 0)):
            linear_accel[i, :] = acc_values[i, :] 
            continue
        try:
            rotation = R.from_quat(quat_values[i])
            gravity_sensor_frame = rotation.apply(gravity_world, inverse=True)
            linear_accel[i, :] = acc_values[i, :] - gravity_sensor_frame
        except ValueError:
             linear_accel[i, :] = acc_values[i, :]
             
    return linear_accel


def calculate_angular_velocity_from_quat(rot_data, time_delta=None):
    """Calculate angular velocity from quaternion data"""
    if time_delta is None:
        time_delta = 1.0 / DATA_CONFIG['SAMPLE_RATE']
    
    if isinstance(rot_data, pd.DataFrame):
        quat_values = rot_data[['rot_x', 'rot_y', 'rot_z', 'rot_w']].values
    else:
        quat_values = rot_data

    num_samples = quat_values.shape[0]
    angular_vel = np.zeros((num_samples, 3))

    for i in range(num_samples - 1):
        q_t = quat_values[i]
        q_t_plus_dt = quat_values[i+1]

        if np.all(np.isnan(q_t)) or np.all(np.isclose(q_t, 0)) or \
           np.all(np.isnan(q_t_plus_dt)) or np.all(np.isclose(q_t_plus_dt, 0)):
            continue

        try:
            rot_t = R.from_quat(q_t)
            rot_t_plus_dt = R.from_quat(q_t_plus_dt)
            delta_rot = rot_t.inv() * rot_t_plus_dt
            angular_vel[i, :] = delta_rot.as_rotvec() / time_delta
        except ValueError:
            pass
            
    return angular_vel


def calculate_angular_distance(rot_data):
    """Calculate angular distance between consecutive quaternions"""
    if isinstance(rot_data, pd.DataFrame):
        quat_values = rot_data[['rot_x', 'rot_y', 'rot_z', 'rot_w']].values
    else:
        quat_values = rot_data

    num_samples = quat_values.shape[0]
    angular_dist = np.zeros(num_samples)

    for i in range(num_samples - 1):
        q1 = quat_values[i]
        q2 = quat_values[i+1]

        if np.all(np.isnan(q1)) or np.all(np.isclose(q1, 0)) or \
           np.all(np.isnan(q2)) or np.all(np.isclose(q2, 0)):
            angular_dist[i] = 0
            continue
        try:
            r1 = R.from_quat(q1)
            r2 = R.from_quat(q2)
            relative_rotation = r1.inv() * r2
            angle = np.linalg.norm(relative_rotation.as_rotvec())
            angular_dist[i] = angle
        except ValueError:
            angular_dist[i] = 0
            pass
            
    return angular_dist


def extract_sequence_features(sequence_data):
    """Extract angular velocity and other kinematic features from sequence"""
    try:
        if hasattr(sequence_data, 'to_pandas'):
            df = sequence_data.to_pandas()
        else:
            df = sequence_data.copy()
        
        # Calculate angular velocity
        angular_velocity = calculate_angular_velocity_from_quat(df)
        
        # Calculate linear acceleration (gravity removed)
        linear_accel = remove_gravity_from_acc(df, df)
        
        # Calculate angular distance
        angular_distance = calculate_angular_distance(df)
        
        # Combine features
        features = {
            'angular_velocity': angular_velocity,
            'linear_acceleration': linear_accel,
            'angular_distance': angular_distance,
            'timestamp': np.arange(len(df)) / DATA_CONFIG['SAMPLE_RATE']
        }
        
        return features
        
    except Exception as e:
        print(f"Feature extraction failed: {e}")
        return None