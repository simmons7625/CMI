# Model Architecture Documentation

## Overview
The gesture recognition model uses a four-branch architecture that processes Time-of-Flight (ToF), accelerometer, rotation (gyroscope), and thermal sensor data separately, then fuses them for classification into 18 gesture classes.

## Architecture Diagram

```mermaid
graph TD
    %% Input Data
    A[ToF Sensor Data<br/>batch_size × seq_len × 320<br/>5 sensors × 64 values] --> B[Reshape to 8×8 Grid<br/>per sensor per timestep]
    C1[Accelerometer Data<br/>batch_size × seq_len × 3] --> D1[Linear Projection<br/>3 → d_model]
    C2[Rotation Data<br/>batch_size × seq_len × 4] --> D2[Linear Projection<br/>4 → d_model]
    C3[Thermal Data<br/>batch_size × seq_len × 5] --> D3[Linear Projection<br/>5 → d_model]
    
    %% ToF Branch - EfficientNet Processing
    B --> E[Conv2D Stem<br/>1→32, 3×3, BN, ReLU]
    E --> F[MBConv2D Stage 1<br/>32→16, expand=1]
    F --> G[MBConv2D Stage 2<br/>16→24, expand=6, stride=2<br/>8×8→4×4]
    G --> H[MBConv2D Stage 2<br/>24→24, expand=6]
    H --> I[MBConv2D Stage 3<br/>24→40, expand=6, stride=2<br/>4×4→2×2]
    I --> J[MBConv2D Stage 3<br/>40→40, expand=6]
    J --> K[MBConv2D Stage 4<br/>40→80, expand=6]
    K --> L[MBConv2D Stage 5<br/>80→112, expand=6]
    L --> M[MBConv2D Stage 6<br/>112→192, expand=6]
    M --> N[Global Average Pool<br/>AdaptiveAvgPool2d]
    N --> O[Feature Projection<br/>192 → d_model]
    O --> P[Average Sensor Features<br/>per timestep]
    P --> Q[ToF Features<br/>batch_size × d_model × seq_len]
    
    %% Other Sensors Branches
    D1 --> R1[MultiHead Attention<br/>Temporal modeling]
    R1 --> S1[LayerNorm + Dropout]
    S1 --> T1[ACC Features<br/>batch_size × d_model × seq_len]
    
    D2 --> R2[MultiHead Attention<br/>Temporal modeling]
    R2 --> S2[LayerNorm + Dropout]
    S2 --> T2[ROT Features<br/>batch_size × d_model × seq_len]
    
    D3 --> R3[MultiHead Attention<br/>Temporal modeling]
    R3 --> S3[LayerNorm + Dropout]
    S3 --> T3[THM Features<br/>batch_size × d_model × seq_len]
    
    %% MultiHead Attention
    Q --> U[MultiHead Attention<br/>TOF Branch]
    T1 --> V1[MultiHead Attention<br/>ACC Branch]
    T2 --> V2[MultiHead Attention<br/>ROT Branch]
    T3 --> V3[MultiHead Attention<br/>THM Branch]
    U --> W[Attended ToF<br/>batch_size × d_model × seq_len]
    V1 --> X1[Attended ACC<br/>batch_size × d_model × seq_len]
    V2 --> X2[Attended ROT<br/>batch_size × d_model × seq_len]
    V3 --> X3[Attended THM<br/>batch_size × d_model × seq_len]
    
    %% Feature Fusion
    W --> Y[Concatenate Features<br/>dim=1]
    X1 --> Y
    X2 --> Y
    X3 --> Y
    Y --> Z[Fused Features<br/>batch_size × 4×d_model × seq_len]
    Z --> AA[BatchNorm1d]
    AA --> BB[Global Average Pool<br/>AdaptiveAvgPool1d]
    BB --> CC[Pooled Features<br/>batch_size × 4×d_model]
    
    %% Classification Head
    CC --> DD[Linear 512→256<br/>ReLU + Dropout 0.3]
    DD --> EE[Linear 256→128<br/>ReLU + Dropout 0.2]
    EE --> FF[Linear 128→18<br/>Final Classification]
    FF --> GG[Gesture Logits<br/>batch_size × 18]
    
    %% MBConv2D Detail Subgraph
    subgraph MBConv2D_Detail [MBConv2D Block Structure]
        MB1[Input] --> MB2{Expand Ratio > 1?}
        MB2 -->|Yes| MB3[1×1 Conv2d Expansion<br/>BN + ReLU6]
        MB2 -->|No| MB4[Depthwise Conv2d<br/>Groups = channels<br/>BN + ReLU6]
        MB3 --> MB4
        MB4 --> MB5[Squeeze & Excitation<br/>Global Pool + FC + Sigmoid]
        MB5 --> MB6[1×1 Conv2d Projection<br/>BN]
        MB6 --> MB7{Residual Connection?}
        MB7 -->|Same dims| MB8[Add Skip Connection]
        MB7 -->|Different| MB9[Output]
        MB8 --> MB9
    end
    
    %% SE Block Detail
    subgraph SE_Detail [Squeeze-and-Excitation Block]
        SE1[Input: H×W×C] --> SE2[Global Average Pool<br/>1×1×C]
        SE2 --> SE3[FC: C → C/r<br/>ReLU]
        SE3 --> SE4[FC: C/r → C<br/>Sigmoid]
        SE4 --> SE5[Channel-wise Multiply<br/>with Input]
        SE5 --> SE6[Output: H×W×C]
    end
    
    %% Styling
    classDef inputNode fill:#e1f5fe
    classDef processNode fill:#f3e5f5
    classDef attentionNode fill:#fff3e0
    classDef fusionNode fill:#e8f5e8
    classDef outputNode fill:#ffebee
    
    class A,C inputNode
    class E,F,G,H,I,J,K,L,M,N,O,P,D,R,S processNode
    class U,V,W,X attentionNode
    class Y,Z,AA,BB,CC fusionNode
    class DD,EE,FF,GG outputNode
```

## Architecture Components

### 1. TOF Branch - EfficientNet-based Spatial Processing

**Input**: `(batch_size, seq_len, 320)` - 5 sensors × 64 values each

**Key Components:**
- **Spatial Reshaping**: Each sensor's 64 values → 8×8 grid for Conv2D processing
- **EfficientNet Backbone**: MBConv2D blocks with squeeze-and-excitation
- **Feature Extraction**: Global average pooling → projection to d_model

**Architecture Details:**
```
Stem: Conv2d(1→32, 3×3) + BatchNorm2d + ReLU

MBConv2D Blocks:
- Stage 1: MBConv2D(32→16, expand_ratio=1, kernel=3×3, stride=1)
- Stage 2: MBConv2D(16→24, expand_ratio=6, kernel=3×3, stride=2) # 8×8→4×4
- Stage 2: MBConv2D(24→24, expand_ratio=6, kernel=3×3, stride=1)
- Stage 3: MBConv2D(24→40, expand_ratio=6, kernel=5×5, stride=2) # 4×4→2×2
- Stage 3: MBConv2D(40→40, expand_ratio=6, kernel=5×5, stride=1)
- Stage 4: MBConv2D(40→80, expand_ratio=6, kernel=3×3, stride=1)
- Stage 5: MBConv2D(80→112, expand_ratio=6, kernel=5×5, stride=1)
- Stage 6: MBConv2D(112→192, expand_ratio=6, kernel=5×5, stride=1)

Global Average Pooling: AdaptiveAvgPool2d(1)
Feature Projection: Linear(192 → d_model)
```

**Output**: `(batch_size, d_model, seq_len)`

### 2. Other Sensors Branches - Traditional Processing

**Accelerometer Branch:**
- **Input**: `(batch_size, seq_len, 3)` - acc_x, acc_y, acc_z
- **Components**: Linear(3 → d_model) + MultiHeadAttention + LayerNorm + Dropout
- **Output**: `(batch_size, d_model, seq_len)`

**Rotation Branch:**
- **Input**: `(batch_size, seq_len, 4)` - rot_w, rot_x, rot_y, rot_z (quaternion)
- **Components**: Linear(4 → d_model) + MultiHeadAttention + LayerNorm + Dropout
- **Output**: `(batch_size, d_model, seq_len)`

**Thermal Branch:**
- **Input**: `(batch_size, seq_len, 5)` - thm_1 through thm_5
- **Components**: Linear(5 → d_model) + MultiHeadAttention + LayerNorm + Dropout
- **Output**: `(batch_size, d_model, seq_len)`

### 3. MBConv2D Block Details

Each MBConv2D block implements the Mobile Inverted Bottleneck with:
- **Expansion Phase**: 1×1 Conv2d (if expand_ratio > 1)
- **Depthwise Convolution**: Grouped convolution for spatial processing
- **Squeeze-and-Excitation**: Channel attention mechanism
- **Point-wise Projection**: 1×1 Conv2d for dimension reduction
- **Residual Connection**: Skip connection when input/output dimensions match

### 4. MultiHead Attention Mechanism

**Per-Branch MultiHeadAttention** operates on each branch separately:
- **Purpose**: Temporal modeling and feature refinement for each sensor modality
- **Input**: `(batch_size, seq_len, d_model)` (after transpose)
- **Process**: Apply temporal attention → Transpose back
- **Output**: `(batch_size, d_model, seq_len)`

### 5. Feature Fusion & Classification

**Fusion Strategy:**
```python
# All branches output (batch_size, d_model, seq_len)
tof_features = tof_branch(tof_input)      # (batch_size, 128, seq_len)
acc_features = acc_branch(acc_input)      # (batch_size, 128, seq_len)
rot_features = rot_branch(rot_input)      # (batch_size, 128, seq_len)
thm_features = thm_branch(thm_input)      # (batch_size, 128, seq_len)

# Apply MultiHead attention to each branch
tof_features = tof_attention(tof_features)
acc_features = acc_attention(acc_features)
rot_features = rot_attention(rot_features)
thm_features = thm_attention(thm_features)

# Concatenate along feature dimension
fused = torch.cat([tof_features, acc_features, rot_features, thm_features], dim=1)  # (batch_size, 512, seq_len)
```

**Classification Head:**
```python
# Global pooling over sequence dimension
pooled = AdaptiveAvgPool1d(1)(fused).squeeze(-1)  # (batch_size, 512)

# Multi-layer classifier with dropout
classifier = Sequential(
    Linear(512 → 256) + ReLU + Dropout(0.3),
    Linear(256 → 128) + ReLU + Dropout(0.2), 
    Linear(128 → 18)  # 18 gesture classes
)
```

## Data Flow

1. **ToF Processing**: 
   - 320 features → 5×64 → 5×(8×8) grids
   - Each grid processed by EfficientNet backbone
   - Sensor features averaged per timestep
   - Output: Sequential ToF embeddings

2. **Other Sensors Processing**:
   - Accelerometer: 3 features → Linear projection → Temporal attention
   - Rotation: 4 features → Linear projection → Temporal attention  
   - Thermal: 5 features → Linear projection → Temporal attention
   - Output: Sequential sensor embeddings for each modality

3. **Feature Fusion**:
   - Concatenate ToF + ACC + ROT + THM embeddings (4 branches)
   - Apply MultiHead attention to each branch separately
   - Global temporal pooling

4. **Classification**:
   - Multi-layer MLP with dropout
   - Output: Logits for 18 gesture classes

## Model Parameters

- **d_model**: 128 (embedding dimension)
- **num_heads**: 8 (attention heads)
- **num_classes**: 18 (gesture classes)
- **seq_len**: Variable (sequence length, optional for 2D attention)

## Target Classes

**Target Behaviors (8 classes):**
- Above ear - pull hair
- Cheek - pinch skin  
- Eyebrow - pull hair
- Eyelash - pull hair
- Forehead - pull hairline
- Forehead - scratch
- Neck - pinch skin
- Neck - scratch

**Non-target Behaviors (10 classes):**
- Text on phone
- Wave hello
- Write name in air
- Pull air toward your face
- Feel around in tray and pull out an object
- Glasses on/off
- Drink from bottle/cup
- Scratch knee/leg skin
- Write name on leg
- Pinch knee/leg skin

## Usage

```python
from src.model import create_model

# Create model
model = create_model(
    num_classes=18,
    d_model=128, 
    num_heads=8,
    seq_len=100  # Optional for 2D attention
)

# Forward pass
tof_data = torch.randn(batch_size, seq_len, 320)  # ToF sensor data
acc_data = torch.randn(batch_size, seq_len, 3)    # Accelerometer data
rot_data = torch.randn(batch_size, seq_len, 4)    # Rotation/gyroscope data
thm_data = torch.randn(batch_size, seq_len, 5)    # Thermal sensor data

logits = model(tof_data, acc_data, rot_data, thm_data)  # (batch_size, 18)
```