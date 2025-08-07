# CMI Detect Behavior with Sensor Data - Data Explanation

## Competition Overview

**Competition:** CMI Detect Behavior with Sensor Data
**Task Type:** Multiclass Classification (18 classes)
**Data Type:** Time-series sensor data from wearable devices
**Objective:** Detect specific gestures/behaviors from multimodal sensor readings

## Dataset Files

### Core Data Files
- `train.csv` - Training dataset with sensor readings and target labels
- `test.csv` - Test dataset with sensor readings (no labels)
- `train_demographics.csv` - Subject demographic information for training
- `test_demographics.csv` - Subject demographic information for testing

### File Sizes and Structure
- **Training data:** 574,946 rows of time-series sensor readings
- **Sensor features:** 339 total features across 4 sensor modalities
- **Subjects:** Multiple participants with unique IDs (format: SUBJ_XXXXXX)
- **Sequences:** Data grouped by `sequence_id` representing individual gesture instances

## Data Schema

### Main Sensor Data (`train.csv` / `test.csv`)

**Identification Columns:**
- `row_id` - Unique identifier for each row
- `sequence_type` - Type of sequence (Target/Non-target)
- `sequence_id` - Unique identifier for each gesture sequence (e.g., SEQ_000007)
- `sequence_counter` - Row counter within each sequence
- `subject` - Subject identifier (e.g., SUBJ_059520)

**Context Columns:**
- `orientation` - Device/subject orientation during recording
- `behavior` - High-level behavior category
- `phase` - Phase of the gesture (e.g., "Transition")
- `gesture` - **TARGET VARIABLE** - The specific gesture being performed

**Sensor Features (339 total):**

#### 1. Accelerometer (3 features)
- `acc_x`, `acc_y`, `acc_z` - 3-axis acceleration measurements

#### 2. Gyroscope/Rotation (4 features)  
- `rot_w`, `rot_x`, `rot_y`, `rot_z` - Quaternion rotation measurements

#### 3. Thermal Sensors (5 features)
- `thm_1` through `thm_5` - Temperature readings from 5 thermal sensors

#### 4. Time-of-Flight Sensors (320 features)
- `tof_1_v0` through `tof_1_v63` - ToF sensor 1 (64 distance measurements)
- `tof_2_v0` through `tof_2_v63` - ToF sensor 2 (64 distance measurements)  
- `tof_3_v0` through `tof_3_v63` - ToF sensor 3 (64 distance measurements)
- `tof_4_v0` through `tof_4_v63` - ToF sensor 4 (64 distance measurements)
- `tof_5_v0` through `tof_5_v63` - ToF sensor 5 (64 distance measurements)

### Demographics Data (`*_demographics.csv`)

**Subject Information:**
- `subject` - Subject identifier (links to main data)
- `adult_child` - Age category (1=adult, 0=child)
- `age` - Age in years
- `sex` - Gender (1=male, 0=female)
- `handedness` - Dominant hand (1=right, 0=left)
- `height_cm` - Height in centimeters
- `shoulder_to_wrist_cm` - Arm length measurement
- `elbow_to_wrist_cm` - Forearm length measurement

## Target Classes (18 gestures)

### Target Behaviors (8 classes)
Behaviors of primary interest for detection:
1. `Above ear - pull hair`
2. `Cheek - pinch skin`
3. `Eyebrow - pull hair`
4. `Eyelash - pull hair`
5. `Forehead - pull hairline`
6. `Forehead - scratch`
7. `Neck - pinch skin`
8. `Neck - scratch`

### Non-Target Behaviors (10 classes)
Control behaviors for comparison:
1. `Text on phone`
2. `Wave hello`
3. `Write name in air`
4. `Pull air toward your face`
5. `Feel around in tray and pull out an object`
6. `Glasses on/off`
7. `Drink from bottle/cup`
8. `Scratch knee/leg skin`
9. `Write name on leg`
10. `Pinch knee/leg skin`

## Data Characteristics

### Missing Values
- Missing sensor readings are represented as `-1.0`
- Common in Time-of-Flight sensor arrays when sensors are out of range
- Requires careful handling in preprocessing

### Sequence Structure
- Each `sequence_id` represents one complete gesture instance
- Sequences have **variable lengths** (different number of time steps)
- Data is temporally ordered within sequences via `sequence_counter`

### Data Distribution
- Multiple sequences per subject across different gestures
- Balanced representation of target vs non-target behaviors
- Various orientations and phases captured per gesture

## Technical Specifications

### Sensor Technology
- **Accelerometer:** Measures linear acceleration in 3D space
- **Gyroscope:** Measures rotational movement (quaternion format)
- **Thermal Sensors:** Capture temperature variations (5 sensors)
- **Time-of-Flight (ToF):** Distance measurements using light (5 sensors × 64 measurements each)

### Sampling and Timing
- Time-series data with regular sampling intervals
- Sequential measurements within each gesture sequence
- Multiple phases captured per gesture (transition, execution, etc.)

### Coordinate Systems
- Accelerometer: Standard 3D Cartesian coordinates (x, y, z)
- Gyroscope: Quaternion representation (w, x, y, z) for rotation
- ToF sensors: Distance measurements in sensor-specific coordinate frames

## Data Quality Considerations

### Challenges for Modeling
1. **Variable sequence lengths** - Requires padding or dynamic length handling
2. **Missing values (-1.0)** - Need imputation or masking strategies  
3. **High dimensionality** - 339 features require dimensionality reduction or feature selection
4. **Multimodal fusion** - Different sensor types need appropriate combination
5. **Subject variability** - Individual differences in gesture execution

### Preprocessing Recommendations
1. **Handle missing values:** Imputation, masking, or exclusion strategies
2. **Sequence alignment:** Padding/truncation for fixed-length models
3. **Feature scaling:** Normalize different sensor modalities appropriately
4. **Temporal smoothing:** Apply filtering to reduce sensor noise
5. **Feature engineering:** Extract statistical and spectral features from raw signals

## Evaluation Approach

**Submission Format:**
- One prediction per `sequence_id` in test data
- Must match exact gesture strings from the 18 target classes
- Evaluated using classification metrics (likely accuracy or F1-score)

**Local Testing:**
- Use provided `CMIGateway` evaluation framework
- Simulates competition evaluation environment
- Processes sequences individually with timeout constraints