import math

import torch
import torch.nn.functional as F
from torch import nn


class PositionalEncoding(nn.Module):
    """Positional encoding that supports chunk start indices for proper sequence positioning."""

    def __init__(self, d_model, max_seq_length=5000):
        super().__init__()
        self.d_model = d_model

        # Create positional encoding matrix
        pe = torch.zeros(max_seq_length, d_model)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model),
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # Add batch dimension

        # Register as buffer so it moves with the model but isn't a parameter
        self.register_buffer("pe", pe)

    def forward(self, x, chunk_start_idx=None):
        """Apply positional encoding to input tensor.

        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
            chunk_start_idx: Starting position indices for each sequence in the batch
                           Shape: (batch_size,) or None for no offset

        Returns:
            x + positional_encoding: Tensor with positional encoding added
        """
        batch_size, seq_len, d_model = x.size()

        if chunk_start_idx is not None:
            # Add positional encoding with offset for each sequence
            pos_encoding = torch.zeros_like(x)
            for i in range(batch_size):
                start_pos = chunk_start_idx[i].item()
                end_pos = start_pos + seq_len
                pos_encoding[i] = self.pe[0, start_pos:end_pos]
        else:
            # Standard positional encoding starting from 0
            pos_encoding = self.pe[:, :seq_len, :].expand(batch_size, -1, -1)

        return x + pos_encoding


class SEBlock2D(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid(),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class MBConv2D(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        expand_ratio=6,
        kernel_size=3,
        stride=1,
        se_ratio=0.25,
        dropout_rate=0.2,
    ):
        super().__init__()
        self.stride = stride
        self.use_residual = stride == 1 and in_channels == out_channels

        hidden_dim = in_channels * expand_ratio

        layers = []

        # Expansion phase
        if expand_ratio != 1:
            layers.extend(
                [
                    nn.Conv2d(in_channels, hidden_dim, 1, bias=False),
                    nn.BatchNorm2d(hidden_dim),
                    nn.SiLU(inplace=True),
                ],
            )

        # Depthwise convolution
        layers.extend(
            [
                nn.Conv2d(
                    hidden_dim,
                    hidden_dim,
                    kernel_size,
                    stride,
                    kernel_size // 2,
                    groups=hidden_dim,
                    bias=False,
                ),
                nn.BatchNorm2d(hidden_dim),
                nn.SiLU(inplace=True),
            ],
        )

        # Squeeze-and-Excitation
        if se_ratio > 0:
            layers.append(SEBlock2D(hidden_dim, int(1 / se_ratio)))

        # Point-wise linear projection
        layers.extend(
            [
                nn.Conv2d(hidden_dim, out_channels, 1, bias=False),
                nn.BatchNorm2d(out_channels),
            ],
        )

        self.conv = nn.Sequential(*layers)
        self.dropout = nn.Dropout2d(dropout_rate) if dropout_rate > 0 else nn.Identity()
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        if self.use_residual:
            return x + self.dropout(self.conv(x))
        return self.conv(x)


class TOFBranch(nn.Module):
    def __init__(self, num_sensors=5, values_per_sensor=64, d_model=128):
        super().__init__()
        self.num_sensors = num_sensors
        self.values_per_sensor = values_per_sensor
        self.d_model = d_model

        # EfficientNetB0 stem
        self.stem = nn.Conv2d(1, 32, 3, stride=2, padding=1, bias=False)
        self.bn_stem = nn.BatchNorm2d(32)

        # EfficientNetB0 MBConv blocks - proper B0 configuration
        self.mb_blocks = nn.ModuleList(
            [
                # Stage 1: MBConv1, k3x3, s1, e1, i32, o16, se0.25
                MBConv2D(
                    32,
                    16,
                    expand_ratio=1,
                    kernel_size=3,
                    stride=1,
                    se_ratio=0.25,
                ),
                # Stage 2: MBConv6, k3x3, s2, e6, i16, o24, se0.25 (repeat 2)
                MBConv2D(
                    16,
                    24,
                    expand_ratio=6,
                    kernel_size=3,
                    stride=2,
                    se_ratio=0.25,
                ),
                MBConv2D(
                    24,
                    24,
                    expand_ratio=6,
                    kernel_size=3,
                    stride=1,
                    se_ratio=0.25,
                ),
                # Stage 3: MBConv6, k5x5, s2, e6, i24, o40, se0.25 (repeat 2)
                MBConv2D(
                    24,
                    40,
                    expand_ratio=6,
                    kernel_size=5,
                    stride=2,
                    se_ratio=0.25,
                ),
                MBConv2D(
                    40,
                    40,
                    expand_ratio=6,
                    kernel_size=5,
                    stride=1,
                    se_ratio=0.25,
                ),
                # Stage 4: MBConv6, k3x3, s2, e6, i40, o80, se0.25 (repeat 3)
                MBConv2D(
                    40,
                    80,
                    expand_ratio=6,
                    kernel_size=3,
                    stride=2,
                    se_ratio=0.25,
                ),
                MBConv2D(
                    80,
                    80,
                    expand_ratio=6,
                    kernel_size=3,
                    stride=1,
                    se_ratio=0.25,
                ),
                MBConv2D(
                    80,
                    80,
                    expand_ratio=6,
                    kernel_size=3,
                    stride=1,
                    se_ratio=0.25,
                ),
                # Stage 5: MBConv6, k5x5, s1, e6, i80, o112, se0.25 (repeat 3)
                MBConv2D(
                    80,
                    112,
                    expand_ratio=6,
                    kernel_size=5,
                    stride=1,
                    se_ratio=0.25,
                ),
                MBConv2D(
                    112,
                    112,
                    expand_ratio=6,
                    kernel_size=5,
                    stride=1,
                    se_ratio=0.25,
                ),
                MBConv2D(
                    112,
                    112,
                    expand_ratio=6,
                    kernel_size=5,
                    stride=1,
                    se_ratio=0.25,
                ),
                # Stage 6: MBConv6, k5x5, s2, e6, i112, o192, se0.25 (repeat 4)
                MBConv2D(
                    112,
                    192,
                    expand_ratio=6,
                    kernel_size=5,
                    stride=2,
                    se_ratio=0.25,
                ),
                MBConv2D(
                    192,
                    192,
                    expand_ratio=6,
                    kernel_size=5,
                    stride=1,
                    se_ratio=0.25,
                ),
                MBConv2D(
                    192,
                    192,
                    expand_ratio=6,
                    kernel_size=5,
                    stride=1,
                    se_ratio=0.25,
                ),
                MBConv2D(
                    192,
                    192,
                    expand_ratio=6,
                    kernel_size=5,
                    stride=1,
                    se_ratio=0.25,
                ),
                # Stage 7: MBConv6, k3x3, s1, e6, i192, o320, se0.25
                MBConv2D(
                    192,
                    320,
                    expand_ratio=6,
                    kernel_size=3,
                    stride=1,
                    se_ratio=0.25,
                ),
            ],
        )

        # EfficientNetB0 head
        self.conv_head = nn.Conv2d(320, 1280, 1, bias=False)
        self.bn_head = nn.BatchNorm2d(1280)
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.feature_proj = nn.Linear(1280, d_model)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

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

                # Apply EfficientNetB0 backbone
                out = F.silu(self.bn_stem(self.stem(sensor_data)))

                # Apply MBConv blocks
                for mb_block in self.mb_blocks:
                    out = mb_block(out)

                # Apply head convolution
                out = F.silu(self.bn_head(self.conv_head(out)))

                # Global pooling and projection
                out = self.global_pool(out)  # (batch_size, 1280, 1, 1)
                out = out.view(batch_size, -1)  # (batch_size, 1280)
                sensor_feat = self.feature_proj(out)  # (batch_size, d_model)
                timestep_features.append(sensor_feat)

            # Average sensor features for this timestep
            timestep_feat = torch.stack(timestep_features, dim=1).mean(dim=1)
            outputs.append(timestep_feat)

        # Stack to get (batch_size, seq_len, d_model), then transpose for 2D attention
        x = torch.stack(outputs, dim=1)  # (batch_size, seq_len, d_model)
        return x.transpose(1, 2)  # (batch_size, d_model, seq_len) for 2D attention


class OtherSensorsBranch(nn.Module):
    def __init__(self, input_dim, d_model=128, dropout=0.1):
        super().__init__()
        # 2-layer linear model with ReLU activation
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x, chunk_start_idx=None):
        # x shape: (batch_size, seq_len, input_dim)
        batch_size, seq_len, input_dim = x.size()

        # Apply feature extraction to each timestep
        x = x.view(-1, input_dim)  # (batch_size * seq_len, input_dim)
        x = self.feature_extractor(x)  # (batch_size * seq_len, d_model)
        x = x.view(batch_size, seq_len, -1)  # (batch_size, seq_len, d_model)

        # Transpose to (batch_size, d_model, seq_len) for consistency with TOF branch
        return x.transpose(1, 2)


class TransformerEncoderLayer(nn.Module):
    """Custom Transformer encoder layer with positional encoding."""

    def __init__(self, d_model, num_heads, d_ff=None, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff or 4 * d_model

        # Multi-head attention
        self.self_attention = nn.MultiheadAttention(
            d_model,
            num_heads,
            dropout=dropout,
            batch_first=True,
        )

        # Feed forward network
        self.ffn = nn.Sequential(
            nn.Linear(d_model, self.d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(self.d_ff, d_model),
            nn.Dropout(dropout),
        )

        # Layer normalization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # x shape: (batch_size, seq_len, d_model)

        # Multi-head attention with residual connection
        attn_out, _ = self.self_attention(x, x, x)
        x = self.norm1(x + self.dropout(attn_out))

        # Feed forward with residual connection
        ffn_out = self.ffn(x)
        return self.norm2(x + ffn_out)


class FeatureSelectionAttention(nn.Module):
    """Feature selection using multi-head attention to reduce dimensionality."""

    def __init__(self, d_model, d_reduced, num_heads=4, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.d_reduced = d_reduced

        # Learnable query tokens for feature selection
        self.query_tokens = nn.Parameter(torch.randn(1, d_reduced, d_model))

        # Multi-head attention for feature selection
        self.feature_attention = nn.MultiheadAttention(
            d_model,
            num_heads,
            dropout=dropout,
            batch_first=True,
        )

        # Layer normalization and dropout
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        # Projection to reduced dimension
        self.projection = nn.Linear(d_model, d_reduced)

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.query_tokens)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # x shape: (batch_size, seq_len, d_model)
        batch_size, seq_len, d_model = x.size()

        # Expand query tokens for batch
        queries = self.query_tokens.expand(
            batch_size,
            -1,
            -1,
        )  # (batch_size, d_reduced, d_model)

        # Use input as keys and values, queries as learned feature selectors
        selected_features, _ = self.feature_attention(
            queries,  # Query: learnable feature selectors
            x,  # Key: input features
            x,  # Value: input features
        )

        # Apply normalization and projection
        selected_features = self.norm(selected_features)
        selected_features = self.dropout(selected_features)
        selected_features = self.projection(
            selected_features,
        )  # (batch_size, d_reduced, d_reduced)

        # Transpose to (batch_size, d_reduced, seq_len) for consistency
        return selected_features.transpose(1, 2)


class CustomTransformer(nn.Module):
    """Custom Transformer with positional encoding and configurable number of layers."""

    def __init__(
        self,
        d_model,
        num_heads,
        num_layers=1,
        d_ff=None,
        dropout=0.1,
        max_seq_length=5000,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_layers = num_layers

        # Positional encoding
        self.pos_encoding = PositionalEncoding(d_model, max_seq_length)

        # Transformer encoder layers
        self.layers = nn.ModuleList(
            [
                TransformerEncoderLayer(d_model, num_heads, d_ff, dropout)
                for _ in range(num_layers)
            ],
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, chunk_start_idx=None):
        # x shape: (batch_size, seq_len, d_model)

        # Add positional encoding
        x = self.pos_encoding(x, chunk_start_idx)
        x = self.dropout(x)

        # Apply transformer layers
        for layer in self.layers:
            x = layer(x)

        return x


class FeatureSelectionTransformer(nn.Module):
    """Transformer with feature selection step before sequential processing."""

    def __init__(
        self,
        d_model,
        d_reduced,
        num_heads,
        num_layers=1,
        d_ff=None,
        dropout=0.1,
        max_seq_length=5000,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_reduced = d_reduced

        # Feature selection step
        self.feature_selector = FeatureSelectionAttention(
            d_model,
            d_reduced,
            num_heads=num_heads // 2,
            dropout=dropout,
        )

        # Positional encoding for reduced features
        self.pos_encoding = PositionalEncoding(d_reduced, max_seq_length)

        # Sequential transformer layers on reduced features
        self.layers = nn.ModuleList(
            [
                TransformerEncoderLayer(d_reduced, num_heads, d_ff, dropout)
                for _ in range(num_layers)
            ],
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, chunk_start_idx=None):
        # x shape: (batch_size, seq_len, d_model)

        # Step 1: Feature selection (d_model -> d_reduced)
        x = self.feature_selector(x)  # (batch_size, d_reduced, seq_len)
        x = x.transpose(1, 2)  # (batch_size, seq_len, d_reduced)

        # Step 2: Add positional encoding to reduced features
        x = self.pos_encoding(x, chunk_start_idx)
        x = self.dropout(x)

        # Step 3: Apply sequential transformer layers on reduced features
        for layer in self.layers:
            x = layer(x)

        return x  # (batch_size, seq_len, d_reduced)


class GestureBranchedModel(nn.Module):
    def __init__(
        self,
        num_classes=18,
        d_model=128,
        num_heads=8,
        num_layers=1,
        acc_dim=4,
        rot_dim=8,
        thm_dim=5,
        dropout=0.1,
        max_seq_length=5000,
    ):
        super().__init__()
        self.d_model = d_model
        # Feature branches with configurable feature dimensions
        self.tof_branch = TOFBranch(d_model=d_model)
        self.acc_branch = OtherSensorsBranch(
            input_dim=acc_dim,
            d_model=d_model,
            dropout=dropout,
        )
        self.rot_branch = OtherSensorsBranch(
            input_dim=rot_dim,
            d_model=d_model,
            dropout=dropout,
        )
        self.thm_branch = OtherSensorsBranch(
            input_dim=thm_dim,
            d_model=d_model,
            dropout=dropout,
        )

        # Feature fusion by concatenation (4 branches)
        self.fusion_norm = nn.BatchNorm1d(d_model * 4)

        # Feature Selection Transformer: d_model*4 -> d_model -> transformer
        self.feature_transformer = FeatureSelectionTransformer(
            d_model * 4,  # Input: concatenated features from all 4 branches
            d_model,  # Output: reduced back to original d_model dimension
            num_heads,
            num_layers,
            dropout=dropout,
            max_seq_length=max_seq_length,
        )

        # Global pooling and classification
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, num_classes),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(
        self,
        tof_features,
        acc_features,
        rot_features,
        thm_features,
        chunk_start_idx=None,
    ):
        # Step 1: Process each sensor branch
        tof_out = self.tof_branch(tof_features)  # (batch_size, d_model, seq_len)
        acc_out = self.acc_branch(acc_features)  # (batch_size, d_model, seq_len)
        rot_out = self.rot_branch(rot_features)  # (batch_size, d_model, seq_len)
        thm_out = self.thm_branch(thm_features)  # (batch_size, d_model, seq_len)

        # Step 2: Concatenate all sensor features
        fused = torch.cat(
            (tof_out, acc_out, rot_out, thm_out),
            dim=1,
        )  # (batch_size, 4*d_model, seq_len)

        # Apply normalization
        fused = self.fusion_norm(fused)

        # Step 3: Transpose for transformer input (batch_size, seq_len, 4*d_model)
        fused = fused.transpose(1, 2)

        # Step 4: Apply feature selection transformer (reduces dim and applies sequential transformers)
        transformed = self.feature_transformer(
            fused,
            chunk_start_idx,
        )  # (batch_size, seq_len, d_model)

        # Step 5: Global pooling over sequence dimension
        pooled = self.global_pool(transformed.transpose(1, 2)).squeeze(
            -1,
        )  # (batch_size, d_model)

        # Step 6: Final classification
        return self.classifier(pooled)


def create_model(
    num_classes=18,
    d_model=128,
    num_heads=8,
    num_layers=1,
    acc_dim=4,
    rot_dim=8,
    thm_dim=5,
    dropout=0.1,
    max_seq_length=5000,
):
    """Create a GestureBranchedModel with specified parameters.

    Args:
        num_classes (int): Number of gesture classes (default: 18)
        d_model (int): Model dimension (default: 128)
        num_heads (int): Number of attention heads (default: 8)
        num_layers (int): Number of transformer layers (default: 1)
        acc_dim (int): Accelerometer feature dimension (default: 4)
        rot_dim (int): Rotation feature dimension (default: 8)
        thm_dim (int): Thermal feature dimension (default: 5)
        dropout (float): Dropout rate (default: 0.1)
        max_seq_length (int): Maximum sequence length for positional encoding (default: 5000)

    Returns:
        GestureBranchedModel: Initialized model
    """
    return GestureBranchedModel(
        num_classes=num_classes,
        d_model=d_model,
        num_heads=num_heads,
        num_layers=num_layers,
        acc_dim=acc_dim,
        rot_dim=rot_dim,
        thm_dim=thm_dim,
        dropout=dropout,
        max_seq_length=max_seq_length,
    )
