"""
Train GNN Kinematics Model using Reinforcement Learning

This script trains the GNN kinematics model to generate realistic motion patterns
for each gesture using reinforcement learning techniques.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
import random
from tqdm import tqdm
import matplotlib.pyplot as plt
from pathlib import Path

from gnn_kinematics import (
    VirtualKinematicChain,
    GESTURE_CLASSES,
    DEMOGRAPHICS_FEATURES,
    KINEMATIC_CONFIG,
    TRAINING_CONFIG,
    DEVICE,
    extract_sequence_features,
    load_data,
    create_sample_data_for_gnn
)


class GestureMotionEnvironment:
    """Custom environment for training gesture motion patterns"""
    
    def __init__(self, target_gesture, demographics, reference_motion=None):
        self.target_gesture = target_gesture
        self.demographics = demographics
        self.reference_motion = reference_motion  # Real motion data if available
        
        # Environment parameters
        self.max_sequence_length = 100
        self.current_step = 0
        self.generated_motion = []
        
        # Reward parameters
        self.smoothness_weight = 0.3
        self.realism_weight = 0.4
        self.similarity_weight = 0.3
        
    def reset(self):
        """Reset environment for new episode"""
        self.current_step = 0
        self.generated_motion = []
        return self._get_state()
    
    def step(self, action):
        """Take action and return new state, reward, done, info"""
        # Action is angular velocity [x, y, z]
        angular_velocity = action
        self.generated_motion.append(angular_velocity)
        self.current_step += 1
        
        # Calculate reward
        reward = self._calculate_reward(angular_velocity)
        
        # Check if episode is done
        done = self.current_step >= self.max_sequence_length
        
        # Get next state
        next_state = self._get_state()
        
        info = {
            'step': self.current_step,
            'cumulative_reward': sum([self._calculate_reward(motion) for motion in self.generated_motion])
        }
        
        return next_state, reward, done, info
    
    def _get_state(self):
        """Get current environment state"""
        # State includes: gesture embedding, demographics, current motion history
        state = {
            'gesture': self.target_gesture,
            'demographics': self.demographics,
            'motion_history': self.generated_motion[-10:] if len(self.generated_motion) > 0 else [],
            'step': self.current_step,
            'max_steps': self.max_sequence_length
        }
        return state
    
    def _calculate_reward(self, action):
        """Calculate reward for current action"""
        reward = 0.0
        
        # Smoothness reward - penalize sudden changes
        if len(self.generated_motion) > 1:
            prev_motion = self.generated_motion[-2]
            motion_diff = np.linalg.norm(action - prev_motion)
            smoothness_reward = -motion_diff * self.smoothness_weight
            reward += smoothness_reward
        
        # Realism reward - encourage human-like motion magnitudes
        motion_magnitude = np.linalg.norm(action)
        if 0.1 <= motion_magnitude <= 5.0:  # Reasonable angular velocity range
            realism_reward = self.realism_weight
        else:
            realism_reward = -abs(motion_magnitude - 2.5) * self.realism_weight
        reward += realism_reward
        
        # Similarity reward - compare with reference motion if available
        if self.reference_motion is not None and self.current_step < len(self.reference_motion):
            reference_motion = self.reference_motion[self.current_step]
            similarity = -np.linalg.norm(action - reference_motion)
            reward += similarity * self.similarity_weight
        
        # Gesture-specific rewards
        reward += self._gesture_specific_reward(action)
        
        return reward
    
    def _gesture_specific_reward(self, action):
        """Apply gesture-specific reward shaping"""
        motion_magnitude = np.linalg.norm(action)
        
        if 'pull hair' in self.target_gesture:
            # Sharp, quick movements
            if motion_magnitude > 2.0:
                return 0.5
            else:
                return -0.2
                
        elif 'scratch' in self.target_gesture:
            # Repetitive, oscillatory patterns
            if len(self.generated_motion) > 3:
                # Check for oscillatory behavior
                recent_motions = np.array(self.generated_motion[-4:])
                if self._is_oscillatory(recent_motions):
                    return 0.5
            return 0.0
            
        elif 'Text on phone' in self.target_gesture:
            # Small, precise movements
            if 0.1 <= motion_magnitude <= 0.5:
                return 0.5
            else:
                return -0.3
                
        else:
            # Default gesture
            return 0.0
    
    def _is_oscillatory(self, motions):
        """Check if motion pattern is oscillatory"""
        if len(motions) < 4:
            return False
        
        # Simple oscillation detection
        magnitudes = np.linalg.norm(motions, axis=1)
        peaks = []
        for i in range(1, len(magnitudes) - 1):
            if magnitudes[i] > magnitudes[i-1] and magnitudes[i] > magnitudes[i+1]:
                peaks.append(i)
        
        return len(peaks) >= 2


class PPOAgent:
    """Proximal Policy Optimization agent for gesture motion generation"""
    
    def __init__(self, kinematic_model, gesture_classes, demographics_dim=7):
        self.kinematic_model = kinematic_model
        self.gesture_classes = gesture_classes
        self.n_gestures = len(gesture_classes)
        self.gesture_to_idx = {gesture: idx for idx, gesture in enumerate(gesture_classes)}
        
        # RL parameters
        self.lr = 3e-4
        self.gamma = 0.99
        self.eps_clip = 0.2
        self.k_epochs = 4
        self.entropy_coef = 0.01
        
        # Memory
        self.memory = []
        
        # Optimizer
        self.optimizer = AdamW(self.kinematic_model.parameters(), lr=self.lr)
        
        print(f"✅ PPO Agent initialized for {self.n_gestures} gestures")
    
    def select_action(self, state, deterministic=False):
        """Select action using current policy"""
        with torch.no_grad():
            # Prepare state for model
            gesture_name = state['gesture']
            gesture_idx = torch.tensor([self.gesture_to_idx[gesture_name]], dtype=torch.long, device=DEVICE)
            demographics = torch.tensor([state['demographics']], dtype=torch.float32, device=DEVICE)
            
            # Generate motion using kinematic model
            # For RL training, we generate single timestep
            motion = self.kinematic_model(gesture_idx, demographics, sequence_length=1)
            action = motion.squeeze().cpu().numpy()  # [3] angular velocity
            
            # Add exploration noise if not deterministic
            if not deterministic:
                noise = np.random.normal(0, 0.1, size=action.shape)
                action = action + noise
            
            return action
    
    def update(self, episodes_data):
        """Update policy using collected episodes"""
        if len(episodes_data) == 0:
            return
        
        total_loss = 0.0
        n_updates = 0
        
        for episode in episodes_data:
            states = episode['states']
            actions = episode['actions']
            rewards = episode['rewards']
            
            # Calculate discounted rewards
            discounted_rewards = self._calculate_discounted_rewards(rewards)
            
            # Convert to tensors
            gesture_indices = []
            demographics_batch = []
            actions_batch = []
            
            for state, action in zip(states, actions):
                gesture_indices.append(self.gesture_to_idx[state['gesture']])
                demographics_batch.append(state['demographics'])
                actions_batch.append(action)
            
            gesture_indices = torch.tensor(gesture_indices, dtype=torch.long, device=DEVICE)
            demographics_batch = torch.tensor(demographics_batch, dtype=torch.float32, device=DEVICE)
            actions_batch = torch.tensor(actions_batch, dtype=torch.float32, device=DEVICE)
            discounted_rewards = torch.tensor(discounted_rewards, dtype=torch.float32, device=DEVICE)
            
            # Update policy
            for _ in range(self.k_epochs):
                # Generate current predictions
                predicted_motions = []
                for i in range(len(gesture_indices)):
                    motion = self.kinematic_model(
                        gesture_indices[i:i+1], 
                        demographics_batch[i:i+1], 
                        sequence_length=1
                    )
                    predicted_motions.append(motion.squeeze())
                
                predicted_motions = torch.stack(predicted_motions)
                
                # Calculate loss (simplified PPO)
                mse_loss = F.mse_loss(predicted_motions, actions_batch)
                
                # Reward-weighted loss
                weighted_loss = mse_loss * torch.mean(discounted_rewards)
                
                # Entropy bonus for exploration
                entropy_loss = -torch.mean(torch.sum(predicted_motions ** 2, dim=1))
                
                total_loss_step = weighted_loss + self.entropy_coef * entropy_loss
                
                # Backward pass
                self.optimizer.zero_grad()
                total_loss_step.backward()
                torch.nn.utils.clip_grad_norm_(self.kinematic_model.parameters(), 0.5)
                self.optimizer.step()
                
                total_loss += total_loss_step.item()
                n_updates += 1
        
        avg_loss = total_loss / max(n_updates, 1)
        return avg_loss
    
    def _calculate_discounted_rewards(self, rewards):
        """Calculate discounted cumulative rewards"""
        discounted = []
        cumulative = 0
        
        for reward in reversed(rewards):
            cumulative = reward + self.gamma * cumulative
            discounted.append(cumulative)
        
        discounted.reverse()
        return discounted


def train_gnn_with_rl(n_episodes=1000, save_interval=100, use_real_data=True):
    """Main training function using reinforcement learning"""
    print("=" * 60)
    print("TRAINING GNN KINEMATICS WITH REINFORCEMENT LEARNING")
    print("=" * 60)
    
    # Load or create data
    if use_real_data:
        data = load_data()
        if data['train_data'] is None:
            print("Real data not found, using sample data...")
            data = create_sample_data_for_gnn()
    else:
        data = create_sample_data_for_gnn()
    
    # Initialize kinematic model
    kinematic_model = VirtualKinematicChain(
        n_joints=KINEMATIC_CONFIG['n_joints'],
        hidden_dim=KINEMATIC_CONFIG['hidden_dim'],
        n_classes=len(GESTURE_CLASSES),
        gnn_layers=KINEMATIC_CONFIG['gnn_layers'],
        attention_heads=KINEMATIC_CONFIG['attention_heads'],
        dropout=KINEMATIC_CONFIG['dropout']
    ).to(DEVICE)
    
    # Initialize RL agent
    agent = PPOAgent(kinematic_model, GESTURE_CLASSES)
    
    # Training metrics
    episode_rewards = []
    episode_losses = []
    best_reward = -float('inf')
    
    # Create results directory
    results_dir = Path("../results/gnn_rl_training")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n🚀 Starting RL training for {n_episodes} episodes...")
    print(f"   Device: {DEVICE}")
    print(f"   Gestures: {len(GESTURE_CLASSES)}")
    print(f"   Model parameters: {sum(p.numel() for p in kinematic_model.parameters()):,}")
    
    # Training loop
    for episode in tqdm(range(n_episodes), desc="Training Episodes"):
        # Select random gesture and demographics for this episode
        gesture = random.choice(GESTURE_CLASSES)
        demographics = [
            random.choice([0, 1]),  # adult_child
            random.randint(8, 65),  # age
            random.choice([0, 1]),  # sex
            random.choice([0, 1]),  # handedness
            random.uniform(120, 190),  # height_cm
            random.uniform(50, 80),   # shoulder_to_wrist_cm
            random.uniform(20, 35)    # elbow_to_wrist_cm
        ]
        
        # Get reference motion if available
        reference_motion = None
        if use_real_data and data['train_data'] is not None:
            # Try to find real motion data for this gesture
            reference_motion = get_reference_motion(data['train_data'], gesture)
        
        # Create environment
        env = GestureMotionEnvironment(gesture, demographics, reference_motion)
        
        # Run episode
        state = env.reset()
        episode_reward = 0
        episode_data = {
            'states': [],
            'actions': [],
            'rewards': []
        }
        
        done = False
        while not done:
            # Select action
            action = agent.select_action(state, deterministic=False)
            
            # Take step
            next_state, reward, done, info = env.step(action)
            
            # Store experience
            episode_data['states'].append(state)
            episode_data['actions'].append(action)
            episode_data['rewards'].append(reward)
            
            episode_reward += reward
            state = next_state
        
        episode_rewards.append(episode_reward)
        
        # Update agent every few episodes
        if episode % 10 == 0:
            # Collect recent episodes for training
            recent_episodes = [episode_data]  # In practice, collect multiple episodes
            loss = agent.update(recent_episodes)
            if loss is not None:
                episode_losses.append(loss)
        
        # Save best model
        if episode_reward > best_reward:
            best_reward = episode_reward
            torch.save({
                'model_state_dict': kinematic_model.state_dict(),
                'episode': episode,
                'best_reward': best_reward,
                'gesture_classes': GESTURE_CLASSES
            }, results_dir / 'best_rl_model.pth')
        
        # Progress reporting
        if episode % 100 == 0:
            avg_reward = np.mean(episode_rewards[-100:])
            avg_loss = np.mean(episode_losses[-10:]) if episode_losses else 0
            print(f"\nEpisode {episode:4d}: Avg Reward = {avg_reward:.3f}, "
                  f"Best Reward = {best_reward:.3f}, Loss = {avg_loss:.4f}")
        
        # Save checkpoint
        if episode % save_interval == 0 and episode > 0:
            torch.save({
                'model_state_dict': kinematic_model.state_dict(),
                'episode': episode,
                'episode_rewards': episode_rewards,
                'episode_losses': episode_losses,
                'gesture_classes': GESTURE_CLASSES
            }, results_dir / f'checkpoint_episode_{episode}.pth')
    
    # Final save
    torch.save({
        'model_state_dict': kinematic_model.state_dict(),
        'episode': n_episodes,
        'episode_rewards': episode_rewards,
        'episode_losses': episode_losses,
        'gesture_classes': GESTURE_CLASSES,
        'final_model': True
    }, results_dir / 'final_rl_model.pth')
    
    # Plot training progress
    plot_training_progress(episode_rewards, episode_losses, results_dir)
    
    print(f"\n✅ RL Training complete!")
    print(f"   Total episodes: {n_episodes}")
    print(f"   Best reward: {best_reward:.3f}")
    print(f"   Final avg reward: {np.mean(episode_rewards[-100:]):.3f}")
    print(f"   Models saved to: {results_dir}")
    
    return kinematic_model, episode_rewards, episode_losses


def get_reference_motion(train_data, gesture):
    """Extract reference motion data for specific gesture"""
    try:
        # Filter data for specific gesture
        gesture_data = train_data.filter(train_data['gesture'] == gesture)
        
        if len(gesture_data) == 0:
            return None
        
        # Get first sequence as reference
        first_sequence = gesture_data.filter(
            gesture_data['sequence_id'] == gesture_data['sequence_id'][0]
        )
        
        # Extract features
        features = extract_sequence_features(first_sequence)
        if features is not None:
            return features['angular_velocity']
        
    except Exception as e:
        print(f"Failed to extract reference motion for {gesture}: {e}")
    
    return None


def plot_training_progress(episode_rewards, episode_losses, save_dir):
    """Plot and save training progress"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Plot rewards
    ax1.plot(episode_rewards, alpha=0.6)
    if len(episode_rewards) > 100:
        # Moving average
        moving_avg = []
        window = 100
        for i in range(window, len(episode_rewards)):
            moving_avg.append(np.mean(episode_rewards[i-window:i]))
        ax1.plot(range(window, len(episode_rewards)), moving_avg, 'r-', linewidth=2)
    
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Reward')
    ax1.set_title('Training Rewards')
    ax1.grid(True)
    
    # Plot losses
    if episode_losses:
        ax2.plot(episode_losses)
        ax2.set_xlabel('Update Step')
        ax2.set_ylabel('Loss')
        ax2.set_title('Training Losses')
        ax2.grid(True)
    else:
        ax2.text(0.5, 0.5, 'No loss data', ha='center', va='center', transform=ax2.transAxes)
    
    plt.tight_layout()
    plt.savefig(save_dir / 'training_progress.png', dpi=150, bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    # Set random seeds
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    
    # Train model
    model, rewards, losses = train_gnn_with_rl(
        n_episodes=500,  # Reduced for testing
        save_interval=50,
        use_real_data=True
    )
    
    print("\n🎉 Training completed successfully!")