import numpy as np
import pandas as pd
import polars as pl
from scipy.spatial.transform import Rotation as R


class FeatureProcessor:
    def __init__(self):
        pass

    @staticmethod
    def remove_gravity_from_acc(acc_data, rot_data):
        if isinstance(acc_data, pd.DataFrame):
            acc_values = acc_data[["acc_x", "acc_y", "acc_z"]].values
        else:
            acc_values = acc_data

        if isinstance(rot_data, pd.DataFrame):
            quat_values = rot_data[["rot_x", "rot_y", "rot_z", "rot_w"]].values
        else:
            quat_values = rot_data

        num_samples = acc_values.shape[0]
        linear_accel = np.zeros_like(acc_values)

        gravity_world = np.array([0, 0, 9.81])

        for i in range(num_samples):
            if np.all(np.isnan(quat_values[i])) or np.all(
                np.isclose(quat_values[i], 0),
            ):
                linear_accel[i, :] = acc_values[i, :]
                continue

            try:
                rotation = R.from_quat(quat_values[i])
                gravity_sensor_frame = rotation.apply(gravity_world, inverse=True)
                linear_accel[i, :] = acc_values[i, :] - gravity_sensor_frame
            except ValueError:
                linear_accel[i, :] = acc_values[i, :]

        return linear_accel

    @staticmethod
    def calculate_angular_velocity_from_quat(
        rot_data,
        time_delta=1 / 200,
    ):  # Assuming 200Hz sampling rate
        if isinstance(rot_data, pd.DataFrame):
            quat_values = rot_data[["rot_x", "rot_y", "rot_z", "rot_w"]].values
        else:
            quat_values = rot_data

        num_samples = quat_values.shape[0]
        angular_vel = np.zeros((num_samples, 3))

        for i in range(num_samples - 1):
            q_t = quat_values[i]
            q_t_plus_dt = quat_values[i + 1]

            if (
                np.all(np.isnan(q_t))
                or np.all(np.isclose(q_t, 0))
                or np.all(np.isnan(q_t_plus_dt))
                or np.all(np.isclose(q_t_plus_dt, 0))
            ):
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

    @staticmethod
    def calculate_angular_distance(rot_data):
        if isinstance(rot_data, pd.DataFrame):
            quat_values = rot_data[["rot_x", "rot_y", "rot_z", "rot_w"]].values
        else:
            quat_values = rot_data

        num_samples = quat_values.shape[0]
        angular_dist = np.zeros(num_samples)

        for i in range(num_samples - 1):
            q1 = quat_values[i]
            q2 = quat_values[i + 1]

            if (
                np.all(np.isnan(q1))
                or np.all(np.isclose(q1, 0))
                or np.all(np.isnan(q2))
                or np.all(np.isclose(q2, 0))
            ):
                angular_dist[i] = 0  # Or np.nan, depending on desired behavior
                continue
            try:
                # Convert quaternions to Rotation objects
                r1 = R.from_quat(q1)
                r2 = R.from_quat(q2)

                # Calculate angular distance: 2 * arccos(|real(p * q*)|)
                # where p* is the conjugate quaternion of q
                # In scipy.spatial.transform.Rotation, r1.inv() * r2 gives relative rotation.
                # The angle of this relative rotation is the angular distance.
                relative_rotation = r1.inv() * r2

                # The angle of rotation vector corresponds to angular distance
                # The norm of rotation vector is the angle in radians
                angle = np.linalg.norm(relative_rotation.as_rotvec())
                angular_dist[i] = angle
            except ValueError:
                angular_dist[i] = 0  # In case of invalid quaternions

        return angular_dist

    @staticmethod
    def process_thermal_features(thm_data):
        """Process thermal sensor data - simple cleaning only."""
        if isinstance(thm_data, pl.DataFrame):
            thm_cols = [f"thm_{i}" for i in range(1, 6)]
            thm_values = thm_data.select(thm_cols).to_numpy()
        else:
            thm_values = thm_data

        # Replace missing values (-1.0) with 0
        return np.where(thm_values == -1.0, 0.0, thm_values)

    @staticmethod
    def create_sequence_features(sequence_data: pl.DataFrame):
        """Create comprehensive features for a sequence."""
        # Extract sensor columns
        acc_cols = ["acc_x", "acc_y", "acc_z"]
        rot_cols = ["rot_w", "rot_x", "rot_y", "rot_z"]
        thm_cols = [f"thm_{i}" for i in range(1, 6)]
        tof_cols = [f"tof_{i}_v{j}" for i in range(1, 6) for j in range(64)]

        # Get sensor data
        acc_data = sequence_data.select(acc_cols).to_numpy()
        rot_data = sequence_data.select(rot_cols).to_numpy()
        thm_data = sequence_data.select(thm_cols).to_numpy()
        tof_data = sequence_data.select(tof_cols).to_numpy()

        # Handle missing values
        acc_data = np.where(acc_data == -1.0, 0.0, acc_data)
        rot_data = np.where(rot_data == -1.0, 0.0, rot_data)

        # Process features
        processor = FeatureProcessor()

        # Enhanced accelerometer features (remove gravity)
        try:
            linear_acc = processor.remove_gravity_from_acc(acc_data, rot_data)
            acc_magnitude = np.linalg.norm(linear_acc, axis=1, keepdims=True)
            enhanced_acc = np.concatenate(
                [linear_acc, acc_magnitude],
                axis=1,
            )  # 3 + 1 = 4 features
        except:
            # Fallback to original acceleration with magnitude
            acc_magnitude = np.linalg.norm(acc_data, axis=1, keepdims=True)
            enhanced_acc = np.concatenate(
                [acc_data, acc_magnitude],
                axis=1,
            )  # 3 + 1 = 4 features

        # Enhanced rotation features
        try:
            angular_vel = processor.calculate_angular_velocity_from_quat(rot_data)
            angular_dist = processor.calculate_angular_distance(rot_data)
            enhanced_rot = np.concatenate(
                [
                    rot_data,  # Original quaternion (4)
                    angular_vel,  # Angular velocity (3)
                    angular_dist.reshape(-1, 1),  # Angular distance (1)
                ],
                axis=1,
            )  # Total: 4 + 3 + 1 = 8 features
        except:
            # Fallback to original rotation data
            enhanced_rot = rot_data  # 4 features

        # Raw thermal features
        enhanced_thm = np.where(thm_data == -1.0, 0.0, thm_data)  # 5 features

        # ToF features with simple cleaning (cross-sensor correlation now handled in model)
        enhanced_tof = np.where(tof_data == -1.0, 0.0, tof_data)  # 320 features

        return {
            "tof": enhanced_tof,
            "acc": enhanced_acc,
            "rot": enhanced_rot,
            "thm": enhanced_thm,
            "sequence_length": len(sequence_data),
        }
