# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a **Kaggle competition project** for "CMI Detect Behavior with Sensor Data" - a multiclass classification challenge focused on detecting specific gestures/behaviors from wearable sensor data.

**Objective:** Classify 18 different gestures from time-series sensor data collected from wearable devices, distinguishing between target behaviors (8 classes like "Cheek - pinch skin", "Above ear - pull hair") and non-target behaviors (10 classes like "Text on phone", "Wave hello").

## Data Architecture

**Training Data Structure:**
- **574,946 rows** of time-series sensor readings in `data/raw/train.csv`
- Data is grouped by `sequence_id` representing individual gesture sequences
- Each sequence belongs to a `subject` (format: SUBJ_XXXXXX)

**Sensor Features (339 total):**
- **Accelerometer:** `acc_x`, `acc_y`, `acc_z`
- **Gyroscope/Rotation:** `rot_w`, `rot_x`, `rot_y`, `rot_z` (quaternion)
- **Thermal sensors:** `thm_1` through `thm_5`
- **Time-of-flight sensors:** `tof_1_v0` through `tof_5_v63` (320 features, 5 sensors × 64 values each)

**Demographics Data:**
- Subject metadata in `*_demographics.csv`: age, sex, handedness, height, arm measurements
- Missing sensor values represented as `-1.0`

**Target Classes (18 gestures):**
```python
# Target behaviors (8 classes)
target_gestures = [
    'Above ear - pull hair', 'Cheek - pinch skin', 'Eyebrow - pull hair',
    'Eyelash - pull hair', 'Forehead - pull hairline', 'Forehead - scratch',
    'Neck - pinch skin', 'Neck - scratch'
]

# Non-target behaviors (10 classes)  
non_target_gestures = [
    'Text on phone', 'Wave hello', 'Write name in air', 'Pull air toward your face',
    'Feel around in tray and pull out an object', 'Glasses on/off', 
    'Drink from bottle/cup', 'Scratch knee/leg skin', 'Write name on leg', 
    'Pinch knee/leg skin'
]
```

## Development Environment

**Primary Data Processing:** Use **Polars** (preferred) or Pandas for data manipulation
- The competition evaluation framework expects Polars DataFrames
- Gateway code in `data/raw/kaggle_evaluation/` uses Polars extensively

**Key Dependencies:**
```python
import polars as pl  # Primary data processing
import pandas as pd  # Alternative data processing  
import numpy as np   # Numerical operations
import grpc          # For evaluation framework communication
```

**Development Setup:**
```bash
# Install core dependencies (no requirements.txt currently exists)
pip install polars pandas numpy grpcio grpcio-tools

# Verify installation by testing evaluation framework
python data/raw/kaggle_evaluation/cmi_gateway.py
```

## Common Development Commands

**Data Exploration:**
```bash
# Check data dimensions
head -1 data/raw/train.csv | tr ',' '\n' | wc -l  # Count columns
wc -l data/raw/train.csv  # Count rows

# Sample data inspection
head -5 data/raw/train.csv
head -5 data/raw/train_demographics.csv
```

**Project Structure Commands:**
```bash
# Create standard ML directories if missing
mkdir -p src models notebooks results data/processed

# Check data files
ls -la data/raw/
```

## Code Architecture

**Directory Structure:**
```
/home/rl/CMI/
├── data/
│   ├── raw/           # Original competition data
│   └── processed/     # Cleaned/engineered features
├── src/               # Source code modules
├── models/            # Trained model artifacts  
├── notebooks/         # EDA and prototyping
├── results/           # Experiment outputs
└── config/            # Configuration files
```

**Competition Evaluation Framework:**
- Located in `data/raw/kaggle_evaluation/`
- `CMIGateway` class handles data batching and validation
- Processes sequences individually with 30-minute timeout
- Expects single string prediction per sequence
- Framework uses gRPC for communication

**Key Implementation Notes:**
- Each sequence must be classified as exactly one of the 18 gesture classes
- Model must handle variable-length time series (sequences have different lengths)
- Missing sensor values (`-1.0`) require appropriate handling
- Demographics data should be joined on `subject` field

## Development Workflow

1. **EDA Phase:** Use `notebooks/` for exploratory data analysis
2. **Feature Engineering:** Create processed datasets in `data/processed/`
3. **Model Development:** Implement core logic in `src/`
4. **Experimentation:** Track results in `results/`
5. **Model Persistence:** Save trained models in `models/`

**Data Loading Pattern:**
```python
import polars as pl

# Load training data
train = pl.read_csv('data/raw/train.csv')
demographics = pl.read_csv('data/raw/train_demographics.csv')

# Group by sequence for time-series modeling
sequences = train.group_by('sequence_id')
```

**Evaluation Integration:**
- Test locally using `CMIGateway` framework in `data/raw/kaggle_evaluation/`
- Framework expects models to predict one gesture per sequence
- Validation ensures predictions match exact gesture string format

## Testing & Validation

**Local Evaluation Testing:**
```bash
# Test the evaluation framework
cd data/raw/kaggle_evaluation
python cmi_gateway.py

# Check if evaluation server runs
python cmi_inference_server.py
```

**Model Integration Pattern:**
```python
# Your model should integrate with CMIGateway like this:
from data.raw.kaggle_evaluation.cmi_gateway import CMIGateway

class YourModel:
    def predict(self, sequence_data, demographics_data):
        # Return single gesture string from the 18 valid classes
        return "Cheek - pinch skin"  # Example
```

## Project Configuration Notes

**Missing Standard Files:** This project currently lacks standard Python configuration files:
- No `requirements.txt` or `pyproject.toml` - dependencies must be installed manually
- No test configuration (`pytest.ini`) - testing framework needs to be set up
- No CI/CD pipeline - local testing only via Kaggle evaluation framework

**Git Configuration:** 
- `.gitignore` excludes `data/` directory and `.claude/` folder
- Data files are tracked but ignored for commits