"""
Virtual Kinematic Chain GNN Model
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from .config import KINEMATIC_CONFIG, DEMOGRAPHICS_FEATURES, DEVICE


class VirtualKinematicChain(nn.Module):
    """Virtual kinematic chain model using GNN for gesture simulation"""
    
    def __init__(self, n_joints=3, hidden_dim=128, n_classes=18, 
                 gnn_layers=3, attention_heads=4, dropout=0.1):
        super().__init__()
        
        self.n_joints = n_joints
        self.hidden_dim = hidden_dim
        self.n_classes = n_classes
        self.gnn_layers = gnn_layers
        
        # Gesture embeddings
        self.gesture_embeddings = nn.Embedding(n_classes, hidden_dim)
        
        # Demographics encoder
        self.demo_encoder = nn.Sequential(
            nn.Linear(len(DEMOGRAPHICS_FEATURES), 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, hidden_dim)
        )
        
        # Joint-specific initializers
        self.joint_initializers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ) for _ in range(n_joints)
        ])
        
        # Graph neural network layers
        self.gnn_convs = nn.ModuleList()
        self.gnn_norms = nn.ModuleList()
        
        for i in range(gnn_layers):
            # Use Graph Attention Network for better expressiveness
            self.gnn_convs.append(
                GATConv(hidden_dim, hidden_dim // attention_heads, 
                       heads=attention_heads, dropout=dropout, concat=True)
            )
            self.gnn_norms.append(nn.LayerNorm(hidden_dim))
        
        # Temporal modeling
        self.temporal_lstm = nn.LSTM(
            hidden_dim, hidden_dim, batch_first=True, dropout=dropout
        )
        
        # Physics-informed constraints
        self.physics_constraint = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 3)  # 3D angular velocity constraints
        )
        
        # Angular velocity prediction head (for wrist)
        self.angular_velocity_head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 3)  # 3D angular velocity
        )
        
        # Build kinematic graph (shoulder -> elbow -> wrist)
        self.register_buffer('edge_index', 
                           torch.tensor([[0, 1], [1, 2]], dtype=torch.long).t().contiguous())
        
        print(f"✅ VirtualKinematicChain initialized")
        print(f"   Joints: {n_joints}, Hidden dim: {hidden_dim}")
        print(f"   GNN layers: {gnn_layers}, Attention heads: {attention_heads}")
        
    def forward(self, gesture_idx, demographics, sequence_length, return_intermediates=False):
        """Generate angular velocity sequence for given gesture and demographics"""
        batch_size = gesture_idx.shape[0]
        device = gesture_idx.device
        
        # Encode gesture and demographics
        gesture_emb = self.gesture_embeddings(gesture_idx)  # (batch, hidden_dim)
        demo_emb = self.demo_encoder(demographics)  # (batch, hidden_dim)
        
        # Initialize joint features
        joint_features = []
        combined_emb = torch.cat([gesture_emb, demo_emb], dim=1)  # (batch, hidden_dim*2)
        
        for i in range(self.n_joints):
            joint_feat = self.joint_initializers[i](combined_emb)
            joint_features.append(joint_feat)
        
        joint_features = torch.stack(joint_features, dim=1)  # (batch, n_joints, hidden_dim)
        
        # Generate sequence of angular velocities
        angular_velocities = []
        intermediate_states = [] if return_intermediates else None
        
        # LSTM hidden state for temporal consistency
        lstm_hidden = None
        
        for t in range(sequence_length):
            # Flatten for GNN processing
            x = joint_features.view(-1, self.hidden_dim)  # (batch*n_joints, hidden_dim)
            
            # Create batch-aware edge index
            edge_index = self._create_batch_edge_index(batch_size, device)
            
            # Apply GNN layers
            for conv, norm in zip(self.gnn_convs, self.gnn_norms):
                x_new = conv(x, edge_index)
                x_new = norm(x_new)
                x = F.relu(x_new) + x  # Residual connection
            
            # Reshape back to (batch, n_joints, hidden_dim)
            x = x.view(batch_size, self.n_joints, self.hidden_dim)
            
            # Temporal modeling with LSTM
            wrist_features = x[:, -1:, :]  # Extract wrist joint (last joint)
            lstm_out, lstm_hidden = self.temporal_lstm(wrist_features, lstm_hidden)
            wrist_features = lstm_out.squeeze(1)  # (batch, hidden_dim)
            
            # Apply physics constraints
            physics_constraint = self.physics_constraint(wrist_features)
            
            # Predict angular velocity for this timestep
            angular_vel = self.angular_velocity_head(wrist_features)  # (batch, 3)
            
            # Apply physics constraints (soft constraints)
            angular_vel = angular_vel + KINEMATIC_CONFIG['physics_weight'] * physics_constraint
            
            angular_velocities.append(angular_vel)
            
            if return_intermediates:
                intermediate_states.append({
                    'joint_features': x.clone(),
                    'wrist_features': wrist_features.clone(),
                    'physics_constraint': physics_constraint.clone()
                })
            
            # Update joint features with some dynamics (learned evolution)
            # Add small learned perturbation for next timestep
            evolution_noise = 0.05 * torch.randn_like(x) * (t / sequence_length)
            joint_features = x + evolution_noise
        
        # Stack to create sequence
        angular_velocities = torch.stack(angular_velocities, dim=1)  # (batch, seq_len, 3)
        
        if return_intermediates:
            return angular_velocities, intermediate_states
        else:
            return angular_velocities
    
    def _create_batch_edge_index(self, batch_size, device):
        """Create edge index for batched graphs"""
        batch_edge_indices = []
        
        for b in range(batch_size):
            offset = b * self.n_joints
            batch_edges = self.edge_index + offset
            batch_edge_indices.append(batch_edges)
        
        return torch.cat(batch_edge_indices, dim=1).to(device)
    
    def visualize_kinematic_graph(self):
        """Visualize the kinematic graph structure"""
        # Convert to networkx for visualization
        edge_list = self.edge_index.t().cpu().numpy()
        G = nx.DiGraph()
        G.add_edges_from(edge_list)
        
        # Create layout
        pos = {0: (0, 0), 1: (1, 0), 2: (2, 0)}  # Linear layout for shoulder->elbow->wrist
        
        plt.figure(figsize=(8, 4))
        nx.draw(G, pos, with_labels=True, node_color='lightblue', 
                node_size=1500, arrowsize=20, font_size=12)
        
        # Add labels
        labels = {0: 'Shoulder', 1: 'Elbow', 2: 'Wrist'}
        nx.draw_networkx_labels(G, pos, labels, font_size=10)
        
        plt.title('Virtual Kinematic Chain: Shoulder → Elbow → Wrist')
        plt.axis('off')
        plt.tight_layout()
        plt.show()