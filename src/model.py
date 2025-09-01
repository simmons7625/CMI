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

        # Note: Cross-sensor correlation removed since each branch handles single sensor

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

    def _apply_cross_sensor_correlation(self, tof_data):
        """Apply cross-sensor correlation based on physical sensor layout."""
        # TOF sensor layout (based on sensor.png):
        # Sensor layout: TOF1 (center), TOF2 (top), TOF3 (right), TOF4 (bottom), TOF5 (left)
        # Adjacency relationships for noise filtering and feature enhancement

        enhanced_data = tof_data.clone()

        # Define sensor neighbors based on physical layout
        sensor_neighbors = {
            0: [1, 2, 3, 4],  # TOF1 (center) - neighbors with all others
            1: [0, 2, 4],  # TOF2 (top) - neighbors with center, right, left
            2: [0, 1, 3],  # TOF3 (right) - neighbors with center, top, bottom
            3: [0, 2, 4],  # TOF4 (bottom) - neighbors with center, right, left
            4: [0, 1, 3],  # TOF5 (left) - neighbors with center, top, bottom
        }

        # For each sensor, enhance its readings using neighbor information
        for sensor_idx in range(5):
            sensor_start = sensor_idx * 64
            sensor_end = (sensor_idx + 1) * 64
            sensor_data = tof_data[:, :, sensor_start:sensor_end]

            # Get neighbor sensors' data
            neighbor_data_list = []
            for neighbor_idx in sensor_neighbors[sensor_idx]:
                neighbor_start = neighbor_idx * 64
                neighbor_end = (neighbor_idx + 1) * 64
                neighbor_data_list.append(tof_data[:, :, neighbor_start:neighbor_end])

            if neighbor_data_list:
                # Apply cross-sensor correlation using pair-wise weights:
                # Get weights for this sensor from all its neighbors
                weighted_contribution = torch.zeros_like(sensor_data)
                for neighbor_idx in sensor_neighbors[sensor_idx]:
                    weight = torch.sigmoid(
                        self.correlation_weights[sensor_idx, neighbor_idx],
                    )
                    neighbor_start = neighbor_idx * 64
                    neighbor_end = (neighbor_idx + 1) * 64
                    neighbor_data = tof_data[:, :, neighbor_start:neighbor_end]
                    weighted_contribution += weight * neighbor_data

                # Combine original sensor with weighted neighbor contributions
                enhanced_sensor_data = sensor_data + weighted_contribution

                enhanced_data[:, :, sensor_start:sensor_end] = enhanced_sensor_data

        return enhanced_data

    def forward(self, x):
        # x shape: (batch_size, seq_len, 64) # Single sensor * 64 values
        batch_size, seq_len, _ = x.size()

        # Skip cross-sensor correlation for single sensor branches
        # Each branch now handles only one sensor

        # Reshape for processing (single sensor)
        x = x.view(batch_size, seq_len, self.num_sensors, self.values_per_sensor)

        # Process each timestep for the single sensor
        outputs = []
        for t in range(seq_len):
            # Extract sensor data for this timestep and reshape to 8x8 grid
            sensor_data = x[:, t, 0, :].view(
                batch_size,
                1,
                8,
                8,
            )  # Only process first (and only) sensor

            # Apply EfficientNet backbone
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
            outputs.append(sensor_feat)

        # Stack to get (batch_size, seq_len, d_model), then transpose for 2D attention
        x = torch.stack(outputs, dim=1)  # (batch_size, seq_len, d_model)
        return x.transpose(1, 2)  # (batch_size, d_model, seq_len) for 2D attention


class TOFBranchB3(nn.Module):
    def __init__(self, num_sensors=5, values_per_sensor=64, d_model=128):
        super().__init__()
        self.num_sensors = num_sensors
        self.values_per_sensor = values_per_sensor
        self.d_model = d_model

        # Note: Cross-sensor correlation removed since each branch handles single sensor

        # EfficientNetB3 stem - deeper stem
        self.stem = nn.Conv2d(1, 40, 3, stride=2, padding=1, bias=False)
        self.bn_stem = nn.BatchNorm2d(40)

        # EfficientNetB3 MBConv blocks - scaled up from B0
        self.mb_blocks = nn.ModuleList(
            [
                # Stage 1: MBConv1, k3x3, s1, e1, i40, o24, se0.25 (repeat 2)
                MBConv2D(
                    40,
                    24,
                    expand_ratio=1,
                    kernel_size=3,
                    stride=1,
                    se_ratio=0.25,
                ),
                MBConv2D(
                    24,
                    24,
                    expand_ratio=1,
                    kernel_size=3,
                    stride=1,
                    se_ratio=0.25,
                ),
                # Stage 2: MBConv6, k3x3, s2, e6, i24, o32, se0.25 (repeat 3)
                MBConv2D(
                    24,
                    32,
                    expand_ratio=6,
                    kernel_size=3,
                    stride=2,
                    se_ratio=0.25,
                ),
                MBConv2D(
                    32,
                    32,
                    expand_ratio=6,
                    kernel_size=3,
                    stride=1,
                    se_ratio=0.25,
                ),
                MBConv2D(
                    32,
                    32,
                    expand_ratio=6,
                    kernel_size=3,
                    stride=1,
                    se_ratio=0.25,
                ),
                # Stage 3: MBConv6, k5x5, s2, e6, i32, o48, se0.25 (repeat 3)
                MBConv2D(
                    32,
                    48,
                    expand_ratio=6,
                    kernel_size=5,
                    stride=2,
                    se_ratio=0.25,
                ),
                MBConv2D(
                    48,
                    48,
                    expand_ratio=6,
                    kernel_size=5,
                    stride=1,
                    se_ratio=0.25,
                ),
                MBConv2D(
                    48,
                    48,
                    expand_ratio=6,
                    kernel_size=5,
                    stride=1,
                    se_ratio=0.25,
                ),
                # Stage 4: MBConv6, k3x3, s2, e6, i48, o96, se0.25 (repeat 5)
                MBConv2D(
                    48,
                    96,
                    expand_ratio=6,
                    kernel_size=3,
                    stride=2,
                    se_ratio=0.25,
                ),
                MBConv2D(
                    96,
                    96,
                    expand_ratio=6,
                    kernel_size=3,
                    stride=1,
                    se_ratio=0.25,
                ),
                MBConv2D(
                    96,
                    96,
                    expand_ratio=6,
                    kernel_size=3,
                    stride=1,
                    se_ratio=0.25,
                ),
                MBConv2D(
                    96,
                    96,
                    expand_ratio=6,
                    kernel_size=3,
                    stride=1,
                    se_ratio=0.25,
                ),
                MBConv2D(
                    96,
                    96,
                    expand_ratio=6,
                    kernel_size=3,
                    stride=1,
                    se_ratio=0.25,
                ),
                # Stage 5: MBConv6, k5x5, s1, e6, i96, o136, se0.25 (repeat 5)
                MBConv2D(
                    96,
                    136,
                    expand_ratio=6,
                    kernel_size=5,
                    stride=1,
                    se_ratio=0.25,
                ),
                MBConv2D(
                    136,
                    136,
                    expand_ratio=6,
                    kernel_size=5,
                    stride=1,
                    se_ratio=0.25,
                ),
                MBConv2D(
                    136,
                    136,
                    expand_ratio=6,
                    kernel_size=5,
                    stride=1,
                    se_ratio=0.25,
                ),
                MBConv2D(
                    136,
                    136,
                    expand_ratio=6,
                    kernel_size=5,
                    stride=1,
                    se_ratio=0.25,
                ),
                MBConv2D(
                    136,
                    136,
                    expand_ratio=6,
                    kernel_size=5,
                    stride=1,
                    se_ratio=0.25,
                ),
                # Stage 6: MBConv6, k5x5, s2, e6, i136, o232, se0.25 (repeat 6)
                MBConv2D(
                    136,
                    232,
                    expand_ratio=6,
                    kernel_size=5,
                    stride=2,
                    se_ratio=0.25,
                ),
                MBConv2D(
                    232,
                    232,
                    expand_ratio=6,
                    kernel_size=5,
                    stride=1,
                    se_ratio=0.25,
                ),
                MBConv2D(
                    232,
                    232,
                    expand_ratio=6,
                    kernel_size=5,
                    stride=1,
                    se_ratio=0.25,
                ),
                MBConv2D(
                    232,
                    232,
                    expand_ratio=6,
                    kernel_size=5,
                    stride=1,
                    se_ratio=0.25,
                ),
                MBConv2D(
                    232,
                    232,
                    expand_ratio=6,
                    kernel_size=5,
                    stride=1,
                    se_ratio=0.25,
                ),
                MBConv2D(
                    232,
                    232,
                    expand_ratio=6,
                    kernel_size=5,
                    stride=1,
                    se_ratio=0.25,
                ),
                # Stage 7: MBConv6, k3x3, s1, e6, i232, o384, se0.25 (repeat 2)
                MBConv2D(
                    232,
                    384,
                    expand_ratio=6,
                    kernel_size=3,
                    stride=1,
                    se_ratio=0.25,
                ),
                MBConv2D(
                    384,
                    384,
                    expand_ratio=6,
                    kernel_size=3,
                    stride=1,
                    se_ratio=0.25,
                ),
            ],
        )

        # EfficientNetB3 head - scaled up features
        self.conv_head = nn.Conv2d(384, 1536, 1, bias=False)
        self.bn_head = nn.BatchNorm2d(1536)
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.feature_proj = nn.Linear(1536, d_model)
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

    def _apply_cross_sensor_correlation(self, tof_data):
        """Apply cross-sensor correlation based on physical sensor layout."""
        # TOF sensor layout (based on sensor.png):
        # Sensor layout: TOF1 (center), TOF2 (top), TOF3 (right), TOF4 (bottom), TOF5 (left)
        # Adjacency relationships for noise filtering and feature enhancement

        enhanced_data = tof_data.clone()

        # Define sensor neighbors based on physical layout
        sensor_neighbors = {
            0: [1, 2, 3, 4],  # TOF1 (center) - neighbors with all others
            1: [0, 2, 4],  # TOF2 (top) - neighbors with center, right, left
            2: [0, 1, 3],  # TOF3 (right) - neighbors with center, top, bottom
            3: [0, 2, 4],  # TOF4 (bottom) - neighbors with center, right, left
            4: [0, 1, 3],  # TOF5 (left) - neighbors with center, top, bottom
        }

        # For each sensor, enhance its readings using neighbor information
        for sensor_idx in range(5):
            sensor_start = sensor_idx * 64
            sensor_end = (sensor_idx + 1) * 64
            sensor_data = tof_data[:, :, sensor_start:sensor_end]

            # Get neighbor sensors' data
            neighbor_data_list = []
            for neighbor_idx in sensor_neighbors[sensor_idx]:
                neighbor_start = neighbor_idx * 64
                neighbor_end = (neighbor_idx + 1) * 64
                neighbor_data_list.append(tof_data[:, :, neighbor_start:neighbor_end])

            if neighbor_data_list:
                # Apply cross-sensor correlation using pair-wise weights:
                # Get weights for this sensor from all its neighbors
                weighted_contribution = torch.zeros_like(sensor_data)
                for neighbor_idx in sensor_neighbors[sensor_idx]:
                    weight = torch.sigmoid(
                        self.correlation_weights[sensor_idx, neighbor_idx],
                    )
                    neighbor_start = neighbor_idx * 64
                    neighbor_end = (neighbor_idx + 1) * 64
                    neighbor_data = tof_data[:, :, neighbor_start:neighbor_end]
                    weighted_contribution += weight * neighbor_data

                # Combine original sensor with weighted neighbor contributions
                enhanced_sensor_data = sensor_data + weighted_contribution

                enhanced_data[:, :, sensor_start:sensor_end] = enhanced_sensor_data

        return enhanced_data

    def forward(self, x):
        # x shape: (batch_size, seq_len, 64) # Single sensor * 64 values
        batch_size, seq_len, _ = x.size()

        # Skip cross-sensor correlation for single sensor branches
        # Each branch now handles only one sensor

        # Reshape for processing (single sensor)
        x = x.view(batch_size, seq_len, self.num_sensors, self.values_per_sensor)

        # Process each timestep for the single sensor
        outputs = []
        for t in range(seq_len):
            # Extract sensor data for this timestep and reshape to 8x8 grid
            sensor_data = x[:, t, 0, :].view(
                batch_size,
                1,
                8,
                8,
            )  # Only process first (and only) sensor

            # Apply EfficientNetB3 backbone
            out = F.silu(self.bn_stem(self.stem(sensor_data)))

            # Apply MBConv blocks
            for mb_block in self.mb_blocks:
                out = mb_block(out)

            # Apply head convolution
            out = F.silu(self.bn_head(self.conv_head(out)))

            # Global pooling and projection
            out = self.global_pool(out)  # (batch_size, 1536, 1, 1)
            out = out.view(batch_size, -1)  # (batch_size, 1536)
            sensor_feat = self.feature_proj(out)  # (batch_size, d_model)
            outputs.append(sensor_feat)

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

    def forward(self, x):
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

        # Multi-head attention
        attn_out, _ = self.self_attention(x, x, x)
        x = self.norm1(x + self.dropout(attn_out))

        # Feed forward
        ffn_out = self.ffn(x)
        return self.norm2(x + ffn_out)


class FeatureSelectionAttention(nn.Module):
    """Feature selection using multi-head attention to reduce dimensionality."""

    def __init__(self, d_model, hidden_dim, num_heads=4, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.hidden_dim = hidden_dim

        # Learnable query tokens for feature selection
        self.query_tokens = nn.Parameter(torch.randn(1, hidden_dim, d_model))

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
        self.projection = nn.Linear(d_model, hidden_dim)

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
        )  # (batch_size, hidden_dim, d_model)

        # Attention-based feature selection
        selected_features, _ = self.feature_attention(
            queries,  # Query: learnable feature selectors
            x,  # Key: input features
            x,  # Value: input features
        )

        # Normalize and project
        selected_features = self.norm(selected_features)
        selected_features = self.dropout(selected_features)
        selected_features = self.projection(
            selected_features,
        )  # (batch_size, hidden_dim, hidden_dim)

        # Transpose output
        return selected_features.transpose(1, 2)


class FeatureSelectionTransformer(nn.Module):
    """Transformer with feature selection step after sequential processing."""

    def __init__(
        self,
        d_model,
        hidden_dim,
        num_heads,
        num_layers=1,
        d_ff=None,
        dropout=0.1,
        max_seq_length=5000,
    ):
        super().__init__()
        self.d_model = d_model
        self.hidden_dim = hidden_dim

        # Feature selection step
        self.feature_selector = FeatureSelectionAttention(
            d_model,
            hidden_dim,
            num_heads=num_heads // 2,
            dropout=dropout,
        )

        # Positional encoding for full features
        self.pos_encoding = PositionalEncoding(d_model, max_seq_length)

        # Sequential transformer layers on full features
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

        # Feature selection after sequence processing
        x = self.feature_selector(x)  # (batch_size, hidden_dim, seq_len)
        return x.transpose(1, 2)  # (batch_size, seq_len, hidden_dim)


class FeatureSelectionGRU(nn.Module):
    """GRU with feature selection step after sequential processing."""

    def __init__(
        self,
        d_model,
        hidden_dim,
        num_layers=1,
        dropout=0.1,
        bidirectional=True,
    ):
        super().__init__()
        self.d_model = d_model
        self.hidden_dim = hidden_dim
        self.bidirectional = bidirectional

        # Feature selection layer
        self.feature_selector = FeatureSelectionAttention(
            d_model,
            hidden_dim,
            num_heads=4,
            dropout=dropout,
        )

        # GRU layers on full features
        self.gru = nn.GRU(
            input_size=d_model,
            hidden_size=d_model // (2 if bidirectional else 1),
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional,
            batch_first=True,
        )

        # Layer normalization and dropout
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.GRU):
                for name, param in m.named_parameters():
                    if "weight_ih" in name:
                        nn.init.xavier_uniform_(param.data)
                    elif "weight_hh" in name:
                        nn.init.orthogonal_(param.data)
                    elif "bias" in name:
                        nn.init.constant_(param.data, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x, chunk_start_idx=None):
        # x shape: (batch_size, seq_len, d_model)

        # Apply dropout
        x = self.dropout(x)

        # Apply GRU
        gru_out, _ = self.gru(x)  # (batch_size, seq_len, d_model)

        # Normalize
        gru_out = self.norm(gru_out)

        # Feature selection after sequence processing
        x = self.feature_selector(gru_out)  # (batch_size, hidden_dim, seq_len)
        return x.transpose(1, 2)  # (batch_size, seq_len, hidden_dim)


class GestureBranchedModel(nn.Module):
    def __init__(
        self,
        num_classes=18,
        d_model=128,
        hidden_dim=None,
        num_heads=8,
        num_layers=1,
        acc_dim=4,
        rot_dim=8,
        thm_dim=5,
        dropout=0.1,
        max_seq_length=5000,
        sequence_processor="transformer",  # "transformer" or "gru"
        tof_backbone="b0",  # "b0" or "b3" - choose EfficientNet backbone
    ):
        super().__init__()
        self.d_model = d_model
        self.hidden_dim = hidden_dim if hidden_dim is not None else d_model
        # Feature branches with configurable feature dimensions
        # Create separate TOF branch instances for each sensor
        self.num_sensors = 5  # TOF sensors
        self.tof_branches = nn.ModuleList()
        for _i in range(self.num_sensors):
            if tof_backbone.lower() == "b3":
                # Create TOF branch for single sensor (1 sensor, 64 values)
                tof_branch = TOFBranchB3(
                    num_sensors=1,
                    values_per_sensor=64,
                    d_model=d_model,
                )
            elif tof_backbone.lower() == "b0":
                # Create TOF branch for single sensor (1 sensor, 64 values)
                tof_branch = TOFBranch(
                    num_sensors=1,
                    values_per_sensor=64,
                    d_model=d_model,
                )
            else:
                msg = f"Unknown tof_backbone: {tof_backbone}. Choose 'b0' or 'b3'."
                raise ValueError(msg)
            self.tof_branches.append(tof_branch)
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

        # Feature fusion by concatenation (5 TOF branches + 3 other sensor branches = 8 branches)
        self.fusion_norm = nn.BatchNorm1d(d_model * (self.num_sensors + 3))

        # Feature Selection Processor: d_model*(5+3) -> hidden_dim -> transformer/gru
        total_feature_dim = d_model * (
            self.num_sensors + 3
        )  # 5 TOF branches + 3 other sensor branches
        if sequence_processor.lower() == "gru":
            self.feature_processor = FeatureSelectionGRU(
                total_feature_dim,  # Input: concatenated features from all branches
                self.hidden_dim,  # Output: reduced to hidden_dim dimension
                num_layers=num_layers,
                dropout=dropout,
                bidirectional=True,
            )
        elif sequence_processor.lower() == "transformer":
            self.feature_processor = FeatureSelectionTransformer(
                total_feature_dim,  # Input: concatenated features from all branches
                self.hidden_dim,  # Output: reduced to hidden_dim dimension
                num_heads,
                num_layers,
                dropout=dropout,
                max_seq_length=max_seq_length,
            )
        else:
            msg = f"Unknown sequence_processor: {sequence_processor}. Choose 'transformer' or 'gru'."
            raise ValueError(msg)

        # Global pooling and classification
        self.global_pool = nn.AdaptiveAvgPool1d(1)

        # Embedding extraction (before final classification)
        self.embedding_extractor = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # Final classifier
        self.classifier = nn.Linear(self.hidden_dim, num_classes)
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
    ):
        # Process each TOF sensor separately
        # tof_features shape: (batch_size, seq_len, 320) # 5 sensors * 64 values
        tof_outputs = []
        for sensor_idx in range(self.num_sensors):
            # Extract features for this sensor (64 values per sensor)
            start_idx = sensor_idx * 64
            end_idx = (sensor_idx + 1) * 64
            sensor_tof_data = tof_features[
                :,
                :,
                start_idx:end_idx,
            ]  # (batch_size, seq_len, 64)

            # Process through corresponding TOF branch
            sensor_tof_out = self.tof_branches[sensor_idx](
                sensor_tof_data,
            )  # (batch_size, d_model, seq_len)
            tof_outputs.append(sensor_tof_out)

        # Process other sensor branches
        acc_out = self.acc_branch(acc_features)  # (batch_size, d_model, seq_len)
        rot_out = self.rot_branch(rot_features)  # (batch_size, d_model, seq_len)
        thm_out = self.thm_branch(thm_features)  # (batch_size, d_model, seq_len)

        # Concatenate all features (5 TOF branches + 3 other sensor branches)
        fused = torch.cat(
            [*tof_outputs, acc_out, rot_out, thm_out],
            dim=1,
        )  # (batch_size, (5+3)*d_model, seq_len)

        # Apply normalization
        fused = self.fusion_norm(fused)

        # Transpose for processor
        fused = fused.transpose(1, 2)  # (batch_size, seq_len, 4*d_model)

        # Apply processor with feature selection
        transformed = self.feature_processor(
            fused,
            chunk_start_idx=None,
        )  # (batch_size, seq_len, hidden_dim)

        # Global pooling
        pooled = self.global_pool(transformed.transpose(1, 2)).squeeze(
            -1,
        )  # (batch_size, hidden_dim)

        # Extract embeddings
        embeddings = self.embedding_extractor(pooled)  # (batch_size, hidden_dim)

        # Classification
        return self.classifier(embeddings)  # (batch_size, num_classes)


def create_model(
    num_classes=18,
    d_model=128,
    hidden_dim=128,
    num_heads=8,
    num_layers=1,
    acc_dim=3,
    rot_dim=4,
    thm_dim=5,
    dropout=0.1,
    max_seq_length=5000,
    sequence_processor="transformer",
    tof_backbone="b0",
):
    """Create a GestureBranchedModel with specified parameters.

    Args:
        num_classes (int): Number of gesture classes (default: 18)
        d_model (int): Model dimension (default: 128)
        hidden_dim (int): Reduced feature dimension after feature selection (default: d_model)
        num_heads (int): Number of attention heads (default: 8)
        num_layers (int): Number of transformer/GRU layers (default: 1)
        acc_dim (int): Accelerometer feature dimension (default: 4)
        rot_dim (int): Rotation feature dimension (default: 8)
        thm_dim (int): Thermal feature dimension (default: 5)
        dropout (float): Dropout rate (default: 0.1)
        max_seq_length (int): Maximum sequence length for positional encoding (default: 5000)
        sequence_processor (str): Sequence processor type, "transformer" or "gru" (default: "transformer")
        tof_backbone (str): TOF backbone architecture, "b0" or "b3" (default: "b0")
        chunk_size (int): Default chunk size for training (default: 100)

    Returns:
        GestureBranchedModel: Initialized model
    """
    return GestureBranchedModel(
        num_classes=num_classes,
        d_model=d_model,
        hidden_dim=hidden_dim,
        num_heads=num_heads,
        num_layers=num_layers,
        acc_dim=acc_dim,
        rot_dim=rot_dim,
        thm_dim=thm_dim,
        dropout=dropout,
        max_seq_length=max_seq_length,
        sequence_processor=sequence_processor,
        tof_backbone=tof_backbone,
    )
