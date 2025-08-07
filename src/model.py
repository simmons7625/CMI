import torch
import torch.nn.functional as F
from torch import nn


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
                    nn.ReLU6(inplace=True),
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
                nn.ReLU6(inplace=True),
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
    def __init__(self, input_dim, d_model=128, num_heads=8, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.attention = nn.MultiheadAttention(d_model, num_heads, batch_first=True)
        self.norm = nn.LayerNorm(d_model)
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
        # x shape: (batch_size, seq_len, input_dim)
        x = self.input_proj(x)
        attn_out, _ = self.attention(x, x, x)
        x = self.norm(x + self.dropout(attn_out))
        # Transpose to (batch_size, d_model, seq_len) for consistency with TOF branch
        return x.transpose(1, 2)


class GestureBranchedModel(nn.Module):
    def __init__(
        self,
        num_classes=18,
        d_model=128,
        num_heads=8,
        seq_len=None,
        acc_dim=4,
        rot_dim=8,
        thm_dim=5,
    ):
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        # Feature branches with configurable feature dimensions
        self.tof_branch = TOFBranch(d_model=d_model)
        self.acc_branch = OtherSensorsBranch(
            input_dim=acc_dim,
            d_model=d_model,
            num_heads=num_heads,
        )
        self.rot_branch = OtherSensorsBranch(
            input_dim=rot_dim,
            d_model=d_model,
            num_heads=num_heads,
        )
        self.thm_branch = OtherSensorsBranch(
            input_dim=thm_dim,
            d_model=d_model,
            num_heads=num_heads,
        )
        # Attention layers
        self.tof_attention = nn.MultiheadAttention(d_model, num_heads, batch_first=True)
        self.acc_attention = nn.MultiheadAttention(d_model, num_heads, batch_first=True)
        self.rot_attention = nn.MultiheadAttention(d_model, num_heads, batch_first=True)
        self.thm_attention = nn.MultiheadAttention(d_model, num_heads, batch_first=True)

        # Feature fusion by concatenation (4 branches)
        self.fusion_norm = nn.BatchNorm1d(d_model * 4)

        # Global pooling and classification
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(
            nn.Linear(d_model * 4, d_model * 2),
            nn.ReLU(),
            nn.Linear(d_model * 2, d_model),
            nn.ReLU(),
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

    def forward(self, tof_features, acc_features, rot_features, thm_features):
        # Process each branch
        tof_out = self.tof_branch(tof_features)  # (batch_size, d_model, seq_len)
        acc_out = self.acc_branch(acc_features)  # (batch_size, d_model, seq_len)
        rot_out = self.rot_branch(rot_features)  # (batch_size, d_model, seq_len)
        thm_out = self.thm_branch(thm_features)  # (batch_size, d_model, seq_len)

        # Apply attention - need to transpose for MultiheadAttention
        tof_out = tof_out.transpose(1, 2)  # (batch_size, seq_len, d_model)
        acc_out = acc_out.transpose(1, 2)  # (batch_size, seq_len, d_model)
        rot_out = rot_out.transpose(1, 2)  # (batch_size, seq_len, d_model)
        thm_out = thm_out.transpose(1, 2)  # (batch_size, seq_len, d_model)

        tof_out = self.tof_attention(tof_out, tof_out, tof_out)[0]
        acc_out = self.acc_attention(acc_out, acc_out, acc_out)[0]
        rot_out = self.rot_attention(rot_out, rot_out, rot_out)[0]
        thm_out = self.thm_attention(thm_out, thm_out, thm_out)[0]

        # Transpose back for concatenation
        tof_out = tof_out.transpose(1, 2)  # (batch_size, d_model, seq_len)
        acc_out = acc_out.transpose(1, 2)  # (batch_size, d_model, seq_len)
        rot_out = rot_out.transpose(1, 2)  # (batch_size, d_model, seq_len)
        thm_out = thm_out.transpose(1, 2)  # (batch_size, d_model, seq_len)

        # Concatenate features along the channel dimension
        fused = torch.cat(
            (tof_out, acc_out, rot_out, thm_out),
            dim=1,
        )  # (batch_size, 4*d_model, seq_len)

        # Apply normalization
        fused = self.fusion_norm(fused)
        # Global pooling over sequence dimension
        pooled = self.global_pool(fused).squeeze(-1)  # (batch_size, d_model*4)

        # Classification to 18 classes
        return self.classifier(pooled)


def create_model(
    num_classes=18,
    d_model=128,
    num_heads=8,
    seq_len=None,
    acc_dim=4,
    rot_dim=8,
    thm_dim=5,
):
    """Create a GestureBranchedModel with specified parameters.

    Args:
        num_classes (int): Number of gesture classes (default: 18)
        d_model (int): Model dimension (default: 128)
        num_heads (int): Number of attention heads (default: 8)
        seq_len (int, optional): Fixed sequence length for 2D attention
        acc_dim (int): Accelerometer feature dimension (default: 4)
        rot_dim (int): Rotation feature dimension (default: 8)
        thm_dim (int): Thermal feature dimension (default: 5)

    Returns:
        GestureBranchedModel: Initialized model
    """
    return GestureBranchedModel(
        num_classes=num_classes,
        d_model=d_model,
        num_heads=num_heads,
        seq_len=seq_len,
        acc_dim=acc_dim,
        rot_dim=rot_dim,
        thm_dim=thm_dim,
    )
