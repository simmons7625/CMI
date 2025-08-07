# CMI Detect Behavior with Sensor Data - Competition Abstract

## English

### Competition Overview
The "CMI Detect Behavior with Sensor Data" is a machine learning competition hosted on Kaggle that focuses on developing algorithms to classify specific human gestures and behaviors using multimodal sensor data from wearable devices.

### Objective
Participants are tasked with building classification models that can accurately identify 18 different gesture types from time-series sensor data. The challenge involves distinguishing between 8 target behaviors (primarily self-touching gestures like hair pulling and skin pinching) and 10 non-target control behaviors (everyday activities like texting or waving).

### Dataset Description
- **Training Data**: 574,946 rows of sensor readings across multiple sequences
- **Sensor Modalities**: 4 types with 339 total features
  - Accelerometer (3 features): 3-axis linear acceleration
  - Gyroscope (4 features): Quaternion rotation measurements
  - Thermal sensors (5 features): Temperature readings
  - Time-of-Flight sensors (320 features): Distance measurements from 5 sensors × 64 values each
- **Demographics**: Subject metadata including age, gender, handedness, and physical measurements
- **Sequences**: Variable-length time series grouped by unique sequence IDs
- **Subjects**: Multiple participants with different characteristics

### Target Classes
**Target Behaviors (8 classes):**
- Above ear - pull hair
- Cheek - pinch skin
- Eyebrow - pull hair
- Eyelash - pull hair
- Forehead - pull hairline
- Forehead - scratch
- Neck - pinch skin
- Neck - scratch

**Non-Target Behaviors (10 classes):**
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

### Technical Challenges
1. **Multimodal Data Fusion**: Combining different sensor types effectively
2. **Variable Sequence Lengths**: Handling time series of different durations
3. **Missing Values**: Managing sensor readings marked as -1.0
4. **High Dimensionality**: Processing 339 sensor features efficiently
5. **Temporal Dependencies**: Capturing time-based patterns in gesture execution
6. **Individual Variability**: Accounting for differences in how people perform gestures

### Evaluation
- **Metric**: Classification accuracy (likely)
- **Submission Format**: One prediction per sequence ID
- **Validation**: Local testing using provided Kaggle evaluation framework
- **Classes**: Must match exact gesture string format from the 18 predefined categories

### Applications
This research has significant implications for:
- **Healthcare Monitoring**: Early detection of repetitive behaviors associated with various conditions
- **Behavioral Analysis**: Understanding patterns in human movement and self-interaction
- **Wearable Technology**: Improving gesture recognition in smart devices
- **Accessibility**: Developing assistive technologies for behavior monitoring

---

## 日本語

### コンペティション概要
「CMI Detect Behavior with Sensor Data」は、ウェアラブルデバイスからのマルチモーダルセンサーデータを使用して、特定の人間のジェスチャーや行動を分類するアルゴリズムの開発に焦点を当てたKaggle主催の機械学習コンペティションです。

### 目的
参加者は、時系列センサーデータから18種類の異なるジェスチャータイプを正確に識別する分類モデルの構築が求められます。このチャレンジでは、8つのターゲット行動（主に髪を引っ張る、皮膚をつまむなどの自己接触ジェスチャー）と10の非ターゲット制御行動（テキスト入力や手を振るなどの日常活動）を区別することが含まれます。

### データセット説明
- **訓練データ**: 複数のシーケンスにわたる574,946行のセンサー読み取り値
- **センサーモダリティ**: 4タイプで合計339の特徴量
  - 加速度計（3特徴量）: 3軸線形加速度
  - ジャイロスコープ（4特徴量）: クォータニオン回転測定
  - 熱センサー（5特徴量）: 温度読み取り値
  - Time-of-Flightセンサー（320特徴量）: 5センサー × 64値の距離測定
- **人口統計**: 年齢、性別、利き手、身体測定を含む被験者メタデータ
- **シーケンス**: 一意のシーケンスIDでグループ化された可変長時系列
- **被験者**: 異なる特性を持つ複数の参加者

### ターゲットクラス
**ターゲット行動（8クラス）:**
- 耳上 - 髪を引っ張る
- 頬 - 皮膚をつまむ
- 眉毛 - 髪を引っ張る
- まつ毛 - 髪を引っ張る
- 額 - ヘアラインを引っ張る
- 額 - かく
- 首 - 皮膚をつまむ
- 首 - かく

**非ターゲット行動（10クラス）:**
- 携帯電話でテキスト入力
- 手を振る
- 空中で名前を書く
- 空気を顔に向かって引く
- トレイの中を探って物を取り出す
- 眼鏡をかけ外しする
- ボトル/カップから飲む
- 膝/脚の皮膚をかく
- 脚に名前を書く
- 膝/脚の皮膚をつまむ

### 技術的課題
1. **マルチモーダルデータ融合**: 異なるセンサータイプの効果的な組み合わせ
2. **可変シーケンス長**: 異なる持続時間の時系列の処理
3. **欠損値**: -1.0としてマークされたセンサー読み取り値の管理
4. **高次元性**: 339のセンサー特徴量の効率的な処理
5. **時間依存性**: ジェスチャー実行における時間ベースのパターンの捕捉
6. **個人差**: 人がジェスチャーを実行する方法の違いへの対応

### 評価
- **メトリック**: 分類精度（推定）
- **提出形式**: シーケンスIDごとに1つの予測
- **検証**: 提供されたKaggle評価フレームワークを使用したローカルテスト
- **クラス**: 18の事前定義されたカテゴリからの正確なジェスチャー文字列形式と一致する必要がある

### 応用
この研究は以下の分野に重要な意味を持ちます：
- **ヘルスケアモニタリング**: さまざまな状態に関連する反復行動の早期発見
- **行動分析**: 人間の動きと自己相互作用のパターンの理解
- **ウェアラブル技術**: スマートデバイスにおけるジェスチャー認識の改善
- **アクセシビリティ**: 行動監視のための支援技術の開発

---

## Key Technical Specifications | 主要技術仕様

| Aspect | English | 日本語 |
|--------|---------|--------|
| **Data Size** | 574,946 training samples | 574,946の訓練サンプル |
| **Features** | 339 sensor features | 339のセンサー特徴量 |
| **Classes** | 18 gesture categories | 18のジェスチャーカテゴリ |
| **Sensors** | 4 modalities (ACC, GYRO, THERMAL, TOF) | 4つのモダリティ（加速度、ジャイロ、熱、TOF） |
| **Challenge Type** | Multiclass time-series classification | 多クラス時系列分類 |
| **Evaluation** | Sequence-level prediction accuracy | シーケンスレベルの予測精度 |

---

*This competition represents a significant advancement in wearable sensor technology and behavioral pattern recognition, with potential applications spanning healthcare, human-computer interaction, and assistive technologies.*

*このコンペティションは、ウェアラブルセンサー技術と行動パターン認識における重要な進歩を表しており、ヘルスケア、ヒューマンコンピュータインタラクション、支援技術にわたる潜在的な応用があります。*