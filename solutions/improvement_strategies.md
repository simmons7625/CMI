# CMI Detect Behavior with Sensor Data - 改善戦略

## コンペティション理解に基づく課題分析

### 現在のベースライン手法
- **アンサンブル**: TensorFlow/Keras + PyTorch の2つの異なるアーキテクチャ
- **特徴量**: 重力除去、角速度計算、TOF統計量
- **アーキテクチャ**: ResNet+SE + LSTM/GRU vs Three-branch + BERT Transformer

### 主要課題
1. **マルチモーダル融合**: 4種類のセンサー（加速度、ジャイロ、熱、TOF）の効果的統合
2. **可変長時系列**: 異なる長さのシーケンス処理
3. **高次元データ**: 339特徴量の効率的処理
4. **個人差**: 被験者間のジェスチャー実行方法の違い
5. **欠損値処理**: TOFセンサーの-1.0値への対応

## 有効と考えられる改善手法

### 1. 高度な特徴量エンジニアリング

#### 1.1 物理学に基づく特徴量
```python
# 既存の重力除去に加えて
def extract_physics_features(acc_data, rot_data):
    # 並進運動エネルギー
    kinetic_energy = 0.5 * np.sum(acc_data**2, axis=1)
    
    # 回転運動エネルギー
    angular_velocity = calculate_angular_velocity_from_quat(rot_data)
    rotational_energy = 0.5 * np.sum(angular_velocity**2, axis=1)
    
    # モーメンタム変化率（力の代理）
    momentum_change = np.diff(acc_data, axis=0)
    
    # 慣性モーメントの変化
    inertia_tensor_change = calculate_inertia_change(rot_data)
    
    return kinetic_energy, rotational_energy, momentum_change, inertia_tensor_change
```

#### 1.2 スペクトル領域特徴量
```python
from scipy import signal
from scipy.fft import fft, fftfreq

def extract_frequency_features(time_series_data, sample_rate=200):
    # FFT変換
    fft_vals = fft(time_series_data, axis=0)
    freqs = fftfreq(len(time_series_data), 1/sample_rate)
    
    # スペクトル密度
    power_spectrum = np.abs(fft_vals)**2
    
    # 支配的周波数
    dominant_freq = freqs[np.argmax(power_spectrum, axis=0)]
    
    # スペクトル重心
    spectral_centroid = np.sum(freqs[:, None] * power_spectrum, axis=0) / np.sum(power_spectrum, axis=0)
    
    # ウェーブレット変換
    wavelet_coeffs = signal.cwt(time_series_data, signal.ricker, np.arange(1, 31))
    
    return dominant_freq, spectral_centroid, wavelet_coeffs
```

#### 1.3 TOFセンサーの高次特徴量
```python
def extract_advanced_tof_features(tof_data):
    # 空間的勾配（エッジ検出）
    gradient_x = np.gradient(tof_data.reshape(-1, 8, 8), axis=1)
    gradient_y = np.gradient(tof_data.reshape(-1, 8, 8), axis=2)
    gradient_magnitude = np.sqrt(gradient_x**2 + gradient_y**2)
    
    # テクスチャ特徴量（GLCM）
    from skimage.feature import graycomatrix, graycoprops
    texture_features = []
    for frame in tof_data:
        glcm = graycomatrix(frame.reshape(8, 8).astype(int), [1], [0], normed=True)
        contrast = graycoprops(glcm, 'contrast')[0, 0]
        homogeneity = graycoprops(glcm, 'homogeneity')[0, 0]
        texture_features.append([contrast, homogeneity])
    
    # 光学フロー（動き推定）
    optical_flow = calculate_optical_flow(tof_data)
    
    return gradient_magnitude, texture_features, optical_flow
```

### 2. 高度なアーキテクチャ改善

#### 2.1 Time-Attention Transformer
```python
class TimeAttentionTransformer(nn.Module):
    def __init__(self, d_model=512, nhead=8, num_layers=6):
        super().__init__()
        # Temporal position encoding
        self.pos_encoding = PositionalEncoding(d_model)
        
        # Multi-scale temporal attention
        self.temporal_attention = nn.MultiheadAttention(d_model, nhead)
        
        # Cross-modal attention between sensors
        self.cross_modal_attention = CrossModalAttention(d_model)
        
        # Hierarchical transformer layers
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
    
    def forward(self, imu, thm, tof):
        # Multi-scale feature extraction
        features = self.extract_multiscale_features(imu, thm, tof)
        
        # Cross-modal attention
        attended_features = self.cross_modal_attention(features)
        
        # Temporal modeling
        output = self.transformer(attended_features)
        return output
```

#### 2.2 Graph Neural Network for Sensor Correlation
```python
class SensorGraphNN(nn.Module):
    def __init__(self, num_sensors=4, hidden_dim=256):
        super().__init__()
        # センサー間の空間的関係をモデル化
        self.sensor_graph = self.build_sensor_graph()
        
        # Graph Convolutional layers
        self.gconv1 = GraphConv(input_dim, hidden_dim)
        self.gconv2 = GraphConv(hidden_dim, hidden_dim)
        
        # Temporal Graph Attention
        self.temporal_gat = TemporalGraphAttention(hidden_dim)
    
    def build_sensor_graph(self):
        # センサーの物理的配置に基づくグラフ構造
        # IMU-THM, THM-TOF, TOF-TOF間の隣接関係
        adjacency_matrix = torch.tensor([
            [1, 1, 1, 0, 0],  # IMU
            [1, 1, 1, 1, 1],  # THM (central)
            [1, 1, 1, 1, 1],  # TOF1
            [0, 1, 1, 1, 1],  # TOF2
            [0, 1, 1, 1, 1],  # TOF3
        ])
        return adjacency_matrix
```

#### 2.3 Temporal Contrastive Learning
```python
class TemporalContrastiveLearning(nn.Module):
    def __init__(self, encoder, temperature=0.1):
        super().__init__()
        self.encoder = encoder
        self.temperature = temperature
        
    def forward(self, anchor, positive, negative):
        # 時系列データの前後関係を学習
        z_anchor = self.encoder(anchor)
        z_positive = self.encoder(positive)  # 同じジェスチャーの異なる部分
        z_negative = self.encoder(negative)  # 異なるジェスチャー
        
        # InfoNCE loss
        loss = self.infonce_loss(z_anchor, z_positive, z_negative)
        return loss
    
    def infonce_loss(self, anchor, positive, negative):
        pos_sim = F.cosine_similarity(anchor, positive) / self.temperature
        neg_sim = F.cosine_similarity(anchor, negative) / self.temperature
        
        loss = -torch.log(torch.exp(pos_sim) / (torch.exp(pos_sim) + torch.exp(neg_sim).sum()))
        return loss.mean()
```

### 3. データ拡張・正則化手法

#### 3.1 センサー特化データ拡張
```python
class SensorAugmentation:
    def __init__(self):
        self.noise_levels = {
            'imu': 0.01,
            'thm': 0.005,
            'tof': 0.02
        }
    
    def temporal_jittering(self, sequence, jitter_ratio=0.1):
        # 時間軸での微小なずれを追加
        seq_len = len(sequence)
        jitter_amount = int(seq_len * jitter_ratio)
        start_idx = np.random.randint(0, jitter_amount + 1)
        end_idx = start_idx + seq_len - jitter_amount
        return sequence[start_idx:end_idx]
    
    def sensor_dropout(self, data, dropout_prob=0.1):
        # ランダムにセンサーチャンネルをマスク
        mask = np.random.random(data.shape[-1]) > dropout_prob
        return data * mask
    
    def temporal_masking(self, sequence, mask_ratio=0.15):
        # 時系列の一部をマスク（BERT風）
        seq_len = len(sequence)
        mask_len = int(seq_len * mask_ratio)
        mask_start = np.random.randint(0, seq_len - mask_len)
        
        masked_seq = sequence.copy()
        masked_seq[mask_start:mask_start + mask_len] = 0
        return masked_seq
```

#### 3.2 被験者適応型正則化
```python
class SubjectAdaptiveRegularization:
    def __init__(self, num_subjects):
        # 被験者ごとの特徴量分布を学習
        self.subject_encoders = nn.ModuleList([
            nn.Linear(demographics_dim, latent_dim) 
            for _ in range(num_subjects)
        ])
        
    def subject_aware_loss(self, predictions, targets, subject_ids):
        # 被験者別の分布を考慮した損失関数
        subject_weights = self.calculate_subject_weights(subject_ids)
        weighted_loss = F.cross_entropy(predictions, targets, reduction='none')
        return (weighted_loss * subject_weights).mean()
```

### 4. 高度なアンサンブル手法

#### 4.1 動的重み付けアンサンブル
```python
class DynamicWeightedEnsemble:
    def __init__(self, models, meta_model):
        self.models = models
        self.meta_model = meta_model  # 重みを予測するモデル
        
    def predict(self, sequence, demographics):
        # 各モデルの予測と信頼度を取得
        predictions = []
        confidences = []
        
        for model in self.models:
            pred = model.predict(sequence, demographics)
            conf = self.calculate_confidence(pred)
            predictions.append(pred)
            confidences.append(conf)
        
        # メタモデルで動的重みを計算
        weights = self.meta_model.predict_weights(sequence, demographics, confidences)
        
        # 重み付き平均
        final_pred = np.average(predictions, weights=weights, axis=0)
        return final_pred
```

#### 4.2 スタッキング with 時系列分析
```python
class TimeSeriesStacking:
    def __init__(self, base_models, meta_model):
        self.base_models = base_models
        self.meta_model = meta_model
        
    def create_meta_features(self, sequence, demographics):
        # ベースモデルの予測
        base_predictions = [model.predict_proba(sequence) for model in self.base_models]
        
        # 時系列統計特徴量
        temporal_stats = self.extract_temporal_statistics(sequence)
        
        # 信頼度特徴量
        confidence_features = self.extract_confidence_features(base_predictions)
        
        # 被験者特徴量
        subject_features = self.encode_demographics(demographics)
        
        meta_features = np.concatenate([
            np.concatenate(base_predictions),
            temporal_stats,
            confidence_features,
            subject_features
        ])
        
        return meta_features
```

### 5. 損失関数の改善

#### 5.1 Focal Loss for Class Imbalance
```python
class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        
    def forward(self, predictions, targets):
        ce_loss = F.cross_entropy(predictions, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * ce_loss
        return focal_loss.mean()
```

#### 5.2 Temporal Consistency Loss
```python
class TemporalConsistencyLoss(nn.Module):
    def __init__(self, weight=0.1):
        super().__init__()
        self.weight = weight
        
    def forward(self, predictions, temporal_features):
        # 時系列の一貫性を保つ損失
        consistency_loss = 0
        for i in range(1, len(predictions)):
            feature_diff = torch.norm(temporal_features[i] - temporal_features[i-1], dim=1)
            pred_diff = torch.norm(predictions[i] - predictions[i-1], dim=1)
            consistency_loss += F.mse_loss(feature_diff, pred_diff)
        
        return self.weight * consistency_loss / (len(predictions) - 1)
```

### 6. 実装優先度

#### 高優先度（すぐに効果が期待できる）
1. **物理学ベース特徴量**: 重力除去の拡張、慣性特徴量
2. **TOF高次特徴量**: 空間的勾配、テクスチャ特徴量
3. **スペクトル特徴量**: FFT、ウェーブレット変換
4. **改良データ拡張**: センサー特化、時間軸ジッタリング

#### 中優先度（アーキテクチャ改善）
1. **Cross-Modal Attention**: センサー間相互作用の明示的モデル化
2. **Multi-scale Temporal Modeling**: 異なる時間スケールでの特徴抽出
3. **Temporal Contrastive Learning**: 自己教師あり事前学習

#### 低優先度（実験的手法）
1. **Graph Neural Networks**: センサー配置の空間的関係
2. **Meta-Learning**: 被験者適応
3. **Neural Architecture Search**: 自動アーキテクチャ最適化

## 実装戦略

### Phase 1: 特徴量エンジニアリング強化
- 物理学ベース特徴量の追加
- TOF画像処理技術の適用
- スペクトル解析の導入

### Phase 2: アーキテクチャ改善
- Cross-Modal Attentionの実装
- Multi-scale特徴抽出の追加
- 改良されたアンサンブル手法

### Phase 3: 高度な正則化・学習手法
- Contrastive Learningの導入
- 被験者適応型手法
- 動的アンサンブル重み付け

このような段階的なアプローチにより、既存のベースラインを体系的に改善し、コンペティションでの性能向上を図ることができます。