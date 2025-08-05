import polars as pl
import numpy as np
from typing import Tuple, Dict, Any


class DataLoader:
    def __init__(self, train_path: str = 'data/raw/train.csv', 
                 demographics_path: str = 'data/raw/train_demographics.csv'):
        self.train_path = train_path
        self.demographics_path = demographics_path
        
    def load_data(self) -> Tuple[pl.DataFrame, pl.DataFrame]:
        train_data = pl.read_csv(self.train_path)
        demographics_data = pl.read_csv(self.demographics_path)
        return train_data, demographics_data
    
    def get_sequence_data(self, sequence_id: str, 
                         train_data: pl.DataFrame) -> pl.DataFrame:
        return train_data.filter(pl.col('sequence_id') == sequence_id)
    
    def get_demographics_features(self, subject: str, 
                                demographics_data: pl.DataFrame) -> Dict[str, Any]:
        subject_data = demographics_data.filter(pl.col('subject') == subject)
        if subject_data.height == 0:
            return {}
        
        features = {}
        for col in demographics_data.columns:
            if col != 'subject':
                features[col] = subject_data.select(col).item()
        return features
    
    def get_sequence_with_demographics(self, sequence_id: str) -> Tuple[pl.DataFrame, Dict[str, Any]]:
        train_data, demographics_data = self.load_data()
        
        sequence_data = self.get_sequence_data(sequence_id, train_data)
        if sequence_data.height == 0:
            return pl.DataFrame(), {}
        
        subject = sequence_data.select('subject').unique().item()
        demographics_features = self.get_demographics_features(subject, demographics_data)
        
        return sequence_data, demographics_features


class FeatureProcessor:
    def __init__(self):
        pass
    
    def remove_gravity_from_acc(acc_data, rot_data):

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

    def calculate_angular_velocity_from_quat(rot_data, time_delta=1/200): # Assuming 200Hz sampling rate
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

                # Calculate the relative rotation
                delta_rot = rot_t.inv() * rot_t_plus_dt
                
                # Convert delta rotation to angular velocity vector
                # The rotation vector (Euler axis * angle) scaled by 1/dt
                # is a good approximation for small delta_rot
                angular_vel[i, :] = delta_rot.as_rotvec() / time_delta
            except ValueError:
                # If quaternion is invalid, angular velocity remains zero
                pass
                
        return angular_vel
        
    def calculate_angular_distance(rot_data):
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
                angular_dist[i] = 0 # Или np.nan, в зависимости от желаемого поведения
                continue
            try:
                # Преобразование кватернионов в объекты Rotation
                r1 = R.from_quat(q1)
                r2 = R.from_quat(q2)

                # Вычисление углового расстояния: 2 * arccos(|real(p * q*)|)
                # где p* - сопряженный кватернион q
                # В scipy.spatial.transform.Rotation, r1.inv() * r2 дает относительное вращение.
                # Угол этого относительного вращения - это и есть угловое расстояние.
                relative_rotation = r1.inv() * r2
                
                # Угол rotation vector соответствует угловому расстоянию
                # Норма rotation vector - это угол в радианах
                angle = np.linalg.norm(relative_rotation.as_rotvec())
                angular_dist[i] = angle
            except ValueError:
                angular_dist[i] = 0 # В случае недействительных кватернионов
                pass
                
        return angular_dist