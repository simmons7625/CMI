# baseline.ipynbの処理内容

## 概要
このノートブックは2つの深層学習モデル（TensorFlow/KerasとPyTorch）をアンサンブルしてCMI Detect Behavior with Sensor Dataコンペティションの予測を行うコードです。

## モデル1: TensorFlow/Keras実装

### データ前処理
- **重力除去**: 加速度データから重力成分を除去し`linear_acc_x/y/z`を計算
- **角速度計算**: クォータニオンから角速度`angular_vel_x/y/z`を導出
- **TOF統計**: 各TOFセンサーのピクセル値から統計量（mean/std/min/max）を算出
- **特徴量エンジニアリング**: 
  - 加速度マグニチュード: `sqrt(acc_x² + acc_y² + acc_z²)`
  - ジャーク: 加速度マグニチュードの1階差分

### モデルアーキテクチャ
- **二分岐構造**: IMU（慣性測定装置）とTOF（Time-of-Flight）センサーを別々に処理
- **ResNet+SE Block**: Residual接続にSqueeze-and-Excitationを組み合わせた畳み込みブロック
- **Attention機構**: 双方向LSTM/GRUの出力にアテンション層を適用
- **アンサンブル**: 20個のモデル（D-111シリーズ10個 + v0629シリーズ10個）

### データ拡張
- **MixUp**: 異なるサンプル間で線形補間により仮想的な訓練データを生成

## モデル2: PyTorch実装

### データ前処理
- モデル1と同様の重力除去・角速度計算
- TOF地域別統計: 64ピクセルを複数の地域に分割して統計量を計算
- より詳細な特徴量生成（地域別統計、複数の分割パターン）

### モデルアーキテクチャ
- **三分岐構造**: IMU、熱センサー（THM）、TOFを独立処理
- **ResNet+SE**: 各分岐でResidual+Squeeze-and-Excitationブロック
- **BERT Transformer**: 三分岐の出力をTransformerで統合
- **分類ヘッド**: 多層パーセプトロンで最終予測

### クロスバリデーション
- 5-fold StratifiedKFold でモデル評価・訓練

## アンサンブル予測

### 最終予測関数
```python
def predict(sequence, demographics):
    pred1 = predict1(sequence, demographics)  # TensorFlowモデル
    pred2 = predict2(sequence, demographics)  # PyTorchモデル
    avg_pred = pred1 * 0.6 + pred2 * 0.4     # 重み付き平均
    return dataset.le.classes_[avg_pred.argmax()]
```

### 特徴
- **重み**: TensorFlowモデル60%、PyTorchモデル40%
- **出力**: 18クラスの確率分布から最大値のクラスを選択
- **評価フレームワーク**: Kaggleの`cmi_inference_server`で推論サーバーとして動作

このアプローチは異なるアーキテクチャ（CNN+LSTM vs Transformer）と特徴量エンジニアリング手法を組み合わせることで、単一モデルより高い性能を実現しています。

## 詳細な処理フロー

### 1. 環境設定とライブラリインポート
- TensorFlow 2.18.0とPyTorchの両方を使用
- 乱数シード固定（42）でReproducibilityを確保
- GPU環境の設定（Tesla T4 × 2）

### 2. 特徴量エンジニアリング関数

#### 重力除去関数 (`remove_gravity_from_acc`)
```python
def remove_gravity_from_acc(acc_data, rot_data):
    # クォータニオンから回転行列を計算
    # 世界座標系の重力ベクトル [0, 0, 9.81] をセンサー座標系に変換
    # 測定された加速度から重力成分を減算
```

#### 角速度計算関数 (`calculate_angular_velocity_from_quat`)
```python
def calculate_angular_velocity_from_quat(rot_data, time_delta=1/200):
    # 連続するクォータニオン間の相対回転を計算
    # rotation vectorを時間で微分して角速度を算出
    # サンプリング周波数200Hzを仮定
```

#### 角距離計算関数 (`calculate_angular_distance`)
```python
def calculate_angular_distance(rot_data):
    # 連続する2つのクォータニオン間の角度差を計算
    # 相対回転のrotation vectorのノルムが角距離
```

### 3. モデル1: TensorFlow/Keras実装の詳細

#### ネットワークアーキテクチャ
- **入力**: パディング長127、特徴量次元332（IMU 7次元 + TOF 325次元）
- **IMU分岐**: 
  - Residual SE CNNブロック（64→128チャンネル）
  - カーネルサイズ3→5で段階的に拡張
- **TOF分岐**:
  - 軽量な畳み込み層（64→128チャンネル）
  - より単純な構造でTOFデータを処理
- **統合部**:
  - 双方向LSTM（128ユニット）
  - 双方向GRU（128ユニット）
  - ガウシアンノイズ層（0.09）
  - アテンション機構で重要な時刻に注目

#### SE（Squeeze-and-Excitation）ブロック
```python
def se_block(x, reduction=8):
    # チャンネル間の重要度を学習
    # Global Average Pooling → FC → ReLU → FC → Sigmoid
    # 元の特徴量にアテンション重みを乗算
```

#### MixUp データ拡張
```python
class MixupGenerator(Sequence):
    # λ ~ Beta(α, α) でサンプリング
    # X_mix = λ * X1 + (1-λ) * X2
    # y_mix = λ * y1 + (1-λ) * y2
    # 正則化効果で汎化性能向上
```

### 4. モデル2: PyTorch実装の詳細

#### データセットクラス (`CMIFeDataset`)
- **特徴量生成**: 動的に工学的特徴量を計算
- **スケーリング**: StandardScalerで正規化
- **パディング**: 95パーセンタイル長でsequenceを統一
- **欠損値処理**: センサー別に異なる欠損値で埋める

#### ネットワークアーキテクチャ (`CMIModel`)
- **三分岐設計**:
  - IMU分岐: 深いResNet+SEブロック
  - THM分岐: 軽量な畳み込み
  - TOF分岐: 軽量な畳み込み
- **BERT Transformer**:
  - CLS tokenを先頭に追加
  - Multi-head Attention（10ヘッド）
  - 8層のTransformerエンコーダー
- **分類器**: 
  - 937→303→18次元の多層パーセプトロン
  - BatchNorm + ReLU + Dropout

#### 5-fold交差検証
```python
class CMIFoldDataset:
    # StratifiedKFoldでクラス比を保持
    # 各foldでtrain/validationセットを分割
    # クラス別統計を表示
```

### 5. 推論パイプライン

#### 予測関数の統合
```python
def predict(sequence, demographics):
    pred1 = predict1(sequence, demographics)  # 20モデルアンサンブル
    pred2 = predict2(sequence, demographics)  # 5モデルアンサンブル  
    avg_pred = pred1 * 0.6 + pred2 * 0.4     # 重み付き平均
    return dataset.le.classes_[avg_pred.argmax()]
```

#### Kaggle評価システム連携
```python
inference_server = kaggle_evaluation.cmi_inference_server.CMIInferenceServer(predict)
# 競技環境では inference_server.serve()
# ローカルでは run_local_gateway() でテスト
```

## 技術的特徴

### 1. マルチモーダル学習
- **IMU**: 慣性測定（加速度・角速度）
- **THM**: 熱センサー（5チャンネル）
- **TOF**: 距離センサー（5センサー×64ピクセル）

### 2. 時系列処理
- **可変長対応**: パディング/トランケーションで統一
- **時間的依存関係**: LSTM/GRU/Transformerで捕捉
- **アテンション**: 重要な時刻に自動的に注目

### 3. 正則化技術
- **Dropout**: 過学習防止
- **MixUp**: データ拡張
- **Weight Decay**: L2正則化
- **Early Stopping**: 汎化性能の最適化

### 4. アンサンブル戦略
- **異種モデル**: CNN+RNN vs Transformer
- **複数fold**: 交差検証による安定性向上
- **重み付き平均**: 性能に基づく重み調整

このベースラインは、センサーデータの物理的性質を考慮した特徴量エンジニアリングと、深層学習の表現学習能力を組み合わせることで、高い分類性能を実現しています。