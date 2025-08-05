import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class SEBlock2D(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class MBConv2D(nn.Module):
    def __init__(self, in_channels, out_channels, expand_ratio=6, kernel_size=3, stride=1, se_ratio=0.25, dropout_rate=0.2):
        super().__init__()
        self.stride = stride
        self.use_residual = stride == 1 and in_channels == out_channels
        
        hidden_dim = in_channels * expand_ratio
        
        layers = []
        
        # Expansion phase
        if expand_ratio != 1:
            layers.extend([
                nn.Conv2d(in_channels, hidden_dim, 1, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True)
            ])
        
        # Depthwise convolution
        layers.extend([
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size, stride, kernel_size//2, groups=hidden_dim, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU6(inplace=True)
        ])
        
        # Squeeze-and-Excitation
        if se_ratio > 0:
            layers.append(SEBlock2D(hidden_dim, int(1/se_ratio)))
        
        # Point-wise linear projection
        layers.extend([
            nn.Conv2d(hidden_dim, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels)
        ])
        
        self.conv = nn.Sequential(*layers)
        self.dropout = nn.Dropout2d(dropout_rate) if dropout_rate > 0 else nn.Identity()
        
    def forward(self, x):
        if self.use_residual:
            return x + self.dropout(self.conv(x))
        else:
            return self.conv(x)


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, query, key, value, mask=None):
        batch_size, seq_len = query.size(0), query.size(1)
        
        # Linear transformations and reshape
        Q = self.w_q(query).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = self.w_k(key).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V = self.w_v(value).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        
        # Attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        context = torch.matmul(attn_weights, V)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        
        output = self.w_o(context)
        return output


class MultiHeadAttention2D(nn.Module):
    def __init__(self, feature_dim, seq_len, num_heads, dropout=0.1):
        super().__init__()
        self.feature_dim = feature_dim
        self.seq_len = seq_len
        self.num_heads = num_heads
        
        assert feature_dim % num_heads == 0
        self.head_dim = feature_dim // num_heads
        
        self.w_q = nn.Linear(feature_dim, feature_dim)
        self.w_k = nn.Linear(feature_dim, feature_dim)
        self.w_v = nn.Linear(feature_dim, feature_dim)
        self.w_o = nn.Linear(feature_dim, feature_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        # x shape: (batch_size, feature_dim, seq_len)
        batch_size, feature_dim, seq_len = x.size()
        
        # Transpose to (batch_size, seq_len, feature_dim) for attention
        x = x.transpose(1, 2)  # (batch_size, seq_len, feature_dim)
        
        # Linear transformations
        Q = self.w_q(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.w_k(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.w_v(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Attention computation
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        context = torch.matmul(attn_weights, V)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, feature_dim)
        
        output = self.w_o(context)
        # Transpose back to (batch_size, feature_dim, seq_len)
        return output.transpose(1, 2)

class TOFBranch(nn.Module):
    def __init__(self, num_sensors=5, values_per_sensor=64, d_model=128, num_heads=8):
        super().__init__()
        self.num_sensors = num_sensors
        self.values_per_sensor = values_per_sensor
        self.d_model = d_model
        
        # EfficientNet-style backbone for spatial feature extraction
        self.stem = nn.Conv2d(1, 32, 3, stride=1, padding=1, bias=False)
        self.bn_stem = nn.BatchNorm2d(32)
        
        # MBConv blocks - EfficientNet B0 style
        self.mb_blocks = nn.ModuleList([
            MBConv2D(32, 16, expand_ratio=1, kernel_size=3, stride=1),   # Stage 1
            MBConv2D(16, 24, expand_ratio=6, kernel_size=3, stride=2),   # Stage 2: 8x8 -> 4x4
            MBConv2D(24, 24, expand_ratio=6, kernel_size=3, stride=1),   # Stage 2
            MBConv2D(24, 40, expand_ratio=6, kernel_size=5, stride=2),   # Stage 3: 4x4 -> 2x2
            MBConv2D(40, 40, expand_ratio=6, kernel_size=5, stride=1),   # Stage 3
            MBConv2D(40, 80, expand_ratio=6, kernel_size=3, stride=1),   # Stage 4
            MBConv2D(80, 112, expand_ratio=6, kernel_size=5, stride=1),  # Stage 5
            MBConv2D(112, 192, expand_ratio=6, kernel_size=5, stride=1), # Stage 6
        ])
        
        # Global average pooling and projection
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.feature_proj = nn.Linear(192, d_model)
        
    def forward(self, x):
        # x shape: (batch_size, seq_len, 320) # 5 sensors * 64 values
        batch_size, seq_len, _ = x.size()
        
        # Reshape to process all sensors and timesteps
        x = x.view(batch_size, seq_len, self.num_sensors, self.values_per_sensor)
        
        # Process each sensor-timestep combination
        outputs = []
        for t in range(seq_len):
            timestep_features = []
            
            for sensor in range(self.num_sensors):
                # Extract sensor data and reshape to 8x8 grid
                sensor_data = x[:, t, sensor, :].view(batch_size, 1, 8, 8)
                
                # Apply EfficientNet backbone
                out = F.relu(self.bn_stem(self.stem(sensor_data)))
                
                # Apply MBConv blocks
                for mb_block in self.mb_blocks:
                    out = mb_block(out)
                
                # Global pooling and projection
                out = self.global_pool(out)  # (batch_size, 192, 1, 1)
                out = out.view(batch_size, -1)  # (batch_size, 192)
                sensor_feat = self.feature_proj(out)  # (batch_size, d_model)
                timestep_features.append(sensor_feat)
            
            # Average sensor features for this timestep
            timestep_feat = torch.stack(timestep_features, dim=1).mean(dim=1)
            outputs.append(timestep_feat)
        
        # Stack to get (batch_size, seq_len, d_model), then transpose for 2D attention
        x = torch.stack(outputs, dim=1)  # (batch_size, seq_len, d_model)
        return x.transpose(1, 2)  # (batch_size, d_model, seq_len) for 2D attention

class OtherSensorsBranch(nn.Module):
    def __init__(self, input_dim=19, d_model=128, num_heads=8):  # acc(3) + rot(4) + thm(5) + demographics(7)
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.attention = MultiHeadAttention(d_model, num_heads)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, x):
        # x shape: (batch_size, seq_len, input_dim)
        x = self.input_proj(x)
        x = self.attention(x, x, x)
        x = self.norm(x + self.dropout(x))
        # Transpose to (batch_size, d_model, seq_len) for consistency with TOF branch
        return x.transpose(1, 2)

class GestureBranchedModel(nn.Module):
    def __init__(self, num_classes=18, d_model=128, num_heads=8, seq_len=None):
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        
        # Feature branches
        self.tof_branch = TOFBranch(d_model=d_model, num_heads=num_heads)
        self.acc_branch = OtherSensorsBranch(input_dim=3, d_model=d_model, num_heads=num_heads)
        self.rot_branch = OtherSensorsBranch(input_dim=4, d_model=d_model, num_heads=num_heads)
        self.thm_branch = OtherSensorsBranch(input_dim=5, d_model=d_model, num_heads=num_heads)
        
        # 2D Attention for feature fusion - operates on (feature_dim, seq_len)
        self.tof_attention = MultiHeadAttention2D(d_model, seq_len, num_heads) if seq_len else None
        self.other_attention = MultiHeadAttention2D(d_model, seq_len, num_heads) if seq_len else None
        
        # Feature fusion by concatenation
        self.fusion_norm = nn.BatchNorm1d(d_model * 2)
        
        # Global pooling and classification
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(d_model // 2, num_classes)
        )
        
    def forward(self, tof_features, other_features):
        # Process each branch - both return (batch_size, d_model, seq_len)
        tof_out = self.tof_branch(tof_features)
        other_out = self.other_branch(other_features)
        
        # Apply 2D attention if seq_len is fixed
        if self.tof_attention is not None:
            tof_out = self.tof_attention(tof_out)
        if self.other_attention is not None:
            other_out = self.other_attention(other_out)
        
        # Concatenate features along feature dimension
        fused = torch.cat([tof_out, other_out], dim=1)  # (batch_size, d_model*2, seq_len)
        
        # Apply normalization
        fused = self.fusion_norm(fused)
        
        # Global pooling over sequence dimension
        pooled = self.global_pool(fused).squeeze(-1)  # (batch_size, d_model*2)
        
        # Classification to 18 classes
        logits = self.classifier(pooled)
        return logits


def create_model(num_classes=18, d_model=128, num_heads=8, seq_len=None):
    return GestureBranchedModel(num_classes=num_classes, d_model=d_model, num_heads=num_heads, seq_len=seq_len)