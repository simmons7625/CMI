# ハイブリッドモデル提案: 深層学習特徴量 + LightGBM分類器

## アプローチ概要

現在のアンサンブルモデルの分類層直前の特徴量を抽出し、被験者情報と結合してLightGBMで最終分類を行うハイブリッド手法。

```
センサーデータ → 深層学習モデル → 特徴量抽出 → LightGBM → 最終予測
                                    ↗          ↗
                        被験者情報 ─────────────┘
```

## 実装戦略

### 1. 特徴抽出器の作成

#### 1.1 TensorFlowモデルからの特徴抽出
```python
def create_feature_extractor_tf(models1, final_feature_cols, scaler, pad_len):
    """TensorFlowモデルから分類層直前の特徴量を抽出"""
    
    def extract_features_tf(sequence):
        # 既存の前処理を適用
        df_seq = sequence.to_pandas()
        
        # 重力除去・角速度計算
        linear_accel = remove_gravity_from_acc(df_seq, df_seq)
        df_seq['linear_acc_x'], df_seq['linear_acc_y'], df_seq['linear_acc_z'] = linear_accel[:, 0], linear_accel[:, 1], linear_accel[:, 2]
        df_seq['linear_acc_mag'] = np.sqrt(df_seq['linear_acc_x']**2 + df_seq['linear_acc_y']**2 + df_seq['linear_acc_z']**2)
        df_seq['linear_acc_mag_jerk'] = df_seq['linear_acc_mag'].diff().fillna(0)
        
        angular_vel = calculate_angular_velocity_from_quat(df_seq)
        df_seq['angular_vel_x'], df_seq['angular_vel_y'], df_seq['angular_vel_z'] = angular_vel[:, 0], angular_vel[:, 1], angular_vel[:, 2]
        df_seq['angular_distance'] = calculate_angular_distance(df_seq)
        
        # TOF統計量
        for i in range(1, 6):
            pixel_cols = [f"tof_{i}_v{p}" for p in range(64)]
            tof_data = df_seq[pixel_cols].replace(-1, np.nan)
            df_seq[f'tof_{i}_mean'] = tof_data.mean(axis=1)
            df_seq[f'tof_{i}_std'] = tof_data.std(axis=1)
            df_seq[f'tof_{i}_min'] = tof_data.min(axis=1)
            df_seq[f'tof_{i}_max'] = tof_data.max(axis=1)
        
        # スケーリング・パディング
        mat_unscaled = df_seq[final_feature_cols].ffill().bfill().fillna(0).values.astype('float32')
        mat_scaled = scaler.transform(mat_unscaled)
        pad_input = pad_sequences([mat_scaled], maxlen=pad_len, padding='post', truncating='post', dtype='float32')
        
        # 各モデルから分類層直前の特徴量を抽出
        all_features = []
        for model in models1:
            # 分類層直前のレイヤーを取得
            feature_extractor = Model(inputs=model.input, 
                                    outputs=model.layers[-2].output)  # 最後から2番目のレイヤー
            features = feature_extractor.predict(pad_input, verbose=0)[0]
            all_features.append(features)
        
        # 全モデルの特徴量を結合
        combined_features = np.concatenate(all_features)
        return combined_features
    
    return extract_features_tf
```

#### 1.2 PyTorchモデルからの特徴抽出
```python
def create_feature_extractor_pytorch(models2, dataset):
    """PyTorchモデルから分類層直前の特徴量を抽出"""
    
    def extract_features_pytorch(sequence):
        # 既存の前処理を適用
        imu, thm, tof = dataset.full_dataset.inference_process(sequence)
        
        all_features = []
        with torch.no_grad():
            imu, thm, tof = to_cuda(imu, thm, tof)
            
            for model in models2:
                # 分類層直前の特徴量を抽出
                # モデルの最後のLinear層を除外
                feature_layers = list(model.classifier.children())[:-1]
                feature_extractor = nn.Sequential(*feature_layers)
                
                # フォワードパスで特徴量取得
                imu_feat = model.imu_branch(imu.permute(0, 2, 1))
                thm_feat = model.thm_branch(thm.permute(0, 2, 1))
                tof_feat = model.tof_branch(tof.permute(0, 2, 1))
                
                bert_input = torch.cat([imu_feat, thm_feat, tof_feat], dim=-1).permute(0, 2, 1)
                cls_token = model.cls_token.expand(bert_input.size(0), -1, -1)
                bert_input = torch.cat([cls_token, bert_input], dim=1)
                outputs = model.bert(inputs_embeds=bert_input)
                pred_cls = outputs.last_hidden_state[:, 0, :]
                
                # 分類層直前の特徴量
                features = feature_extractor(pred_cls).cpu().numpy()[0]
                all_features.append(features)
        
        # 全モデルの特徴量を結合
        combined_features = np.concatenate(all_features)
        return combined_features
    
    return extract_features_pytorch
```

### 2. ハイブリッド特徴量の作成

```python
def create_hybrid_features(sequence, demographics, tf_extractor, pytorch_extractor):
    """深層学習特徴量と被験者情報を結合"""
    
    # 深層学習特徴量の抽出
    tf_features = tf_extractor(sequence)
    pytorch_features = pytorch_extractor(sequence)
    
    # 被験者情報の前処理
    demo_features = np.array([
        demographics['adult_child'].iloc[0],
        demographics['age'].iloc[0],
        demographics['sex'].iloc[0],
        demographics['handedness'].iloc[0],
        demographics['height_cm'].iloc[0],
        demographics['shoulder_to_wrist_cm'].iloc[0],
        demographics['elbow_to_wrist_cm'].iloc[0]
    ])
    
    # 被験者情報の正規化
    demo_scaler = StandardScaler()
    demo_features_scaled = demo_scaler.fit_transform(demo_features.reshape(1, -1))[0]
    
    # 全特徴量の結合
    hybrid_features = np.concatenate([
        tf_features,           # TensorFlow特徴量
        pytorch_features,      # PyTorch特徴量  
        demo_features_scaled   # 被験者情報
    ])
    
    return hybrid_features
```

### 3. LightGBM分類器の実装

```python
class HybridLightGBMClassifier:
    def __init__(self, tf_models, pytorch_models, dataset):
        self.tf_extractor = create_feature_extractor_tf(tf_models, final_feature_cols, scaler, pad_len)
        self.pytorch_extractor = create_feature_extractor_pytorch(pytorch_models, dataset)
        
        # LightGBMパラメータ
        self.lgb_params = {
            'objective': 'multiclass',
            'num_class': 18,
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'feature_fraction': 0.9,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbose': 0,
            'random_state': 42
        }
        
        self.model = None
        self.demo_scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
    
    def prepare_training_data(self, train_sequences, train_demographics, train_labels):
        """訓練データの準備"""
        X_hybrid = []
        y = []
        
        for seq_id in tqdm(train_sequences['sequence_id'].unique(), desc="特徴量抽出中"):
            # シーケンスデータの取得
            sequence = train_sequences.filter(pl.col('sequence_id') == seq_id)
            demographics = train_demographics.filter(pl.col('subject') == sequence['subject'][0])
            
            # ハイブリッド特徴量の作成
            hybrid_features = create_hybrid_features(
                sequence, demographics, 
                self.tf_extractor, self.pytorch_extractor
            )
            
            X_hybrid.append(hybrid_features)
            y.append(sequence['gesture'][0])
        
        X_hybrid = np.array(X_hybrid)
        y_encoded = self.label_encoder.fit_transform(y)
        
        return X_hybrid, y_encoded
    
    def train(self, X_hybrid, y_encoded, validation_split=0.2):
        """LightGBM分類器の訓練"""
        # 訓練・検証分割
        X_train, X_val, y_train, y_val = train_test_split(
            X_hybrid, y_encoded, test_size=validation_split, 
            stratify=y_encoded, random_state=42
        )
        
        # LightGBMデータセットの作成
        train_data = lgb.Dataset(X_train, label=y_train)
        val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
        
        # モデル訓練
        self.model = lgb.train(
            self.lgb_params,
            train_data,
            valid_sets=[val_data],
            num_boost_round=1000,
            callbacks=[lgb.early_stopping(50), lgb.log_evaluation(100)]
        )
        
        return self.model
    
    def predict(self, sequence, demographics):
        """単一シーケンスの予測"""
        hybrid_features = create_hybrid_features(
            sequence, demographics,
            self.tf_extractor, self.pytorch_extractor
        )
        
        # LightGBMで予測
        probabilities = self.model.predict(hybrid_features.reshape(1, -1))[0]
        predicted_class_idx = np.argmax(probabilities)
        predicted_class = self.label_encoder.inverse_transform([predicted_class_idx])[0]
        
        return predicted_class
```

### 4. 完整なパイプライン

```python
class HybridPipeline:
    def __init__(self, tf_models, pytorch_models, dataset):
        self.hybrid_classifier = HybridLightGBMClassifier(tf_models, pytorch_models, dataset)
        
    def train_pipeline(self, train_data_path, train_demographics_path):
        """完整な訓練パイプライン"""
        # データロード
        train_sequences = pl.read_csv(train_data_path)
        train_demographics = pl.read_csv(train_demographics_path)
        
        # ハイブリッド特徴量の準備
        print("ハイブリッド特徴量を準備中...")
        X_hybrid, y_encoded = self.hybrid_classifier.prepare_training_data(
            train_sequences, train_demographics, train_sequences['gesture']
        )
        
        # LightGBM訓練
        print("LightGBM分類器を訓練中...")
        self.hybrid_classifier.train(X_hybrid, y_encoded)
        
        print("ハイブリッドモデルの訓練完了！")
    
    def predict(self, sequence, demographics):
        """予測実行"""
        return self.hybrid_classifier.predict(sequence, demographics)
```

## 期待される利点

### 1. **最適な特徴量利用**
- 深層学習モデルの強力な表現学習能力
- LightGBMの表形式データに対する優秀な性能
- 被験者情報の直接的な活用

### 2. **実装の簡単さ**
- 既存モデルを特徴抽出器として再利用
- 新しいアーキテクチャ設計が不要
- 段階的な改善が可能

### 3. **解釈可能性**
- LightGBMの特徴量重要度分析
- どの深層学習特徴量が重要かの理解
- 被験者情報の貢献度の可視化

### 4. **堅牢性**
- 複数のモデルからの多様な特徴量
- 被験者情報による個人差の明示的考慮
- アンサンブル効果の維持

## 実装上の注意点

### 1. **特徴量次元の管理**
```python
# 特徴量次元の確認
tf_feature_dim = 128 * 20  # 20モデル × 128次元
pytorch_feature_dim = 303 * 5  # 5モデル × 303次元  
demo_feature_dim = 7  # 被験者情報
total_dim = tf_feature_dim + pytorch_feature_dim + demo_feature_dim
print(f"総特徴量次元: {total_dim}")
```

### 2. **メモリ効率化**
```python
# バッチ処理による効率化
def batch_feature_extraction(sequences, batch_size=32):
    for i in range(0, len(sequences), batch_size):
        batch = sequences[i:i+batch_size]
        yield process_batch(batch)
```

### 3. **交差検証での評価**
```python
# StratifiedKFoldでの性能評価
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
for fold, (train_idx, val_idx) in enumerate(skf.split(X_hybrid, y_encoded)):
    # fold別の訓練・評価
    pass
```

このハイブリッドアプローチは、現在のモデルの強みを活かしながら被験者情報を効果的に統合する優れた戦略です。実装も比較的簡単で、段階的な改善が可能な点が特に魅力的です。