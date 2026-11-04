# PickleSense Pose AI

PickleSense Pose AI is the machine-vision stroke-classification component of the PickleSense system.

It uses MediaPipe Pose data recorded during pickleball strokes and trains a Temporal Convolutional Network (TCN) to classify six stroke types.

## Stroke Classes

The model classifies:

1. Forehand
2. Backhand
3. Serve
4. Volley
5. Dink
6. Overhead

## Model Input

Each pose frame contains:

- 12 selected MediaPipe world landmarks
- XYZ coordinates for each selected landmark
- 8 joint-angle features

Selected landmarks:

- Left shoulder
- Right shoulder
- Left elbow
- Right elbow
- Left wrist
- Right wrist
- Left hip
- Right hip
- Left knee
- Right knee
- Left ankle
- Right ankle

Feature count:

```text
12 landmarks × 3 coordinates = 36
8 joint angles                = 8
---------------------------------
Total                         = 44 features
```

Each stroke trial is resampled to:

```text
64 frames
```

The final model input shape is:

```text
[batch, 44, 64]
```

## Model Output

The Pose-TCN produces six class logits:

```text
0 = Forehand
1 = Backhand
2 = Serve
3 = Volley
4 = Dink
5 = Overhead
```

Softmax is applied during inference to obtain class probabilities.

## Project Structure

```text
picklesense_pose_ai/
│
├── data/
│   ├── raw/
│   ├── processed/
│   └── manifests/
│
├── config/
│   ├── __init__.py
│   └── settings.py
│
├── dataset/
│   ├── __init__.py
│   ├── scanner.py
│   ├── validator.py
│   ├── splitter.py
│   └── pose_dataset.py
│
├── preprocessing/
│   ├── __init__.py
│   ├── pose_features.py
│   ├── pose_normalizer.py
│   ├── resampler.py
│   └── augmentation.py
│
├── models/
│   ├── __init__.py
│   └── pose_tcn.py
│
├── training/
│   ├── __init__.py
│   ├── train.py
│   └── early_stopping.py
│
├── evaluation/
│   ├── __init__.py
│   ├── metrics.py
│   └── evaluate.py
│
├── inference/
│   ├── __init__.py
│   ├── predictor.py
│   └── live_pose_classifier.py
│
├── outputs/
│   ├── checkpoints/
│   ├── models/
│   ├── plots/
│   ├── reports/
│   └── logs/
│
├── scripts/
│   ├── build_manifest.py
│   ├── prepare_dataset.py
│   ├── train_model.py
│   └── evaluate_model.py
│
├── main.py
├── requirements.txt
└── README.md
```

# Dataset

Place the collected PickleSense recordings inside:

```text
data/raw/
```

Example:

```text
data/raw/
└── Participant01_Session01_Backhand/
    └── Trial_01/
        ├── imu_data.csv
        ├── pose_data.csv
        └── session_info.json
```

Only `pose_data.csv` and `session_info.json` are required by this pose-classification project.

## Required Metadata

Each `session_info.json` should contain at least:

```text
participant_id
session_number
stroke_type
trial_number
handedness
pose_frames
duration_seconds
```

# Installation

From the project directory:

```powershell
python -m pip install -r requirements.txt
```

Dependencies:

```text
numpy
pandas
torch
scikit-learn
matplotlib
```

# Dataset Preparation

## Step 1 — Build Manifest

Run:

```powershell
python -m scripts.build_manifest
```

This scans:

```text
data/raw/
```

and creates:

```text
data/manifests/dataset_manifest.csv
```

## Step 2 — Validate and Split Dataset

Run:

```powershell
python -m scripts.prepare_dataset
```

This:

1. Validates every pose recording.
2. Removes unusable trials.
3. Creates participant-separated datasets.
4. Checks for participant leakage.

Generated files:

```text
data/manifests/
├── dataset_manifest.csv
├── validated_manifest.csv
├── train_manifest.csv
├── validation_manifest.csv
└── test_manifest.csv
```

Participants are kept completely separate between training, validation, and testing.

A participant must never appear in more than one dataset split.

# Pose Preprocessing

The processing pipeline is:

```text
pose_data.csv
      ↓
Valid detected pose frames
      ↓
12 selected world landmarks
      ↓
Pelvis-centered coordinates
      ↓
Shoulder-width scale normalization
      ↓
8 normalized joint angles
      ↓
44 features per frame
      ↓
Training-set standardization
      ↓
Temporal resampling
      ↓
64 frames
      ↓
[44, 64] model input
```

Dataset-level normalization statistics are computed using training participants only.

The resulting normalizer is saved as:

```text
outputs/pose_normalizer.json
```

Validation, test, and live inference all reuse this same normalizer.

# Training Augmentation

Training samples may receive small:

- Temporal speed variations
- Start/end temporal cropping
- Gaussian feature noise

Augmentation is applied only to the training dataset.

Validation and testing are never augmented.

# Pose-TCN Architecture

The classifier uses a Temporal Convolutional Network.

```text
Input
[44 × 64]

↓

Conv1D
44 → 64
Kernel 5

↓

Residual TCN Block
64 → 128
Dilation 1

↓

Residual TCN Block
128 → 128
Dilation 2

↓

Residual TCN Block
128 → 256
Dilation 4

↓

Global Average Pooling

↓

Dense
256 → 128

↓

Dense
128 → 6
```

The final layer outputs six stroke-class logits.

# Training

Run:

```powershell
python -m scripts.train_model
```

Training uses:

```text
Optimizer: AdamW
Loss: CrossEntropyLoss
Learning rate: 0.001
Batch size: 32
Maximum epochs: 100
Early stopping: enabled
Learning-rate reduction: enabled
```

The best model is selected according to validation loss.

Outputs:

```text
outputs/
├── pose_normalizer.json
├── training_history.json
├── checkpoints/
│   └── picklesense_pose_tcn_best.pt
└── models/
    └── picklesense_pose_tcn_final.pt
```

# Evaluation

Run:

```powershell
python -m scripts.evaluate_model
```

Evaluation is performed only on held-out test participants.

Metrics include:

- Accuracy
- Macro Precision
- Macro Recall
- Macro F1
- Per-class Precision
- Per-class Recall
- Per-class F1
- Confusion Matrix
- Per-trial predictions
- Prediction confidence
- Six-class probability distribution

Outputs:

```text
outputs/reports/test_evaluation.json
outputs/plots/confusion_matrix.png
```

# Single-Trial Inference

After training, classify an existing `pose_data.csv` with:

```powershell
python -m inference.predictor "C:\path\to\Trial_01\pose_data.csv"
```

Example output:

```text
Prediction: Forehand
Confidence: 94.37%

Forehand  : 94.37%
Backhand  :  1.21%
Serve     :  0.80%
Volley    :  1.56%
Dink      :  1.42%
Overhead  :  0.64%
```

# Live Integration

`inference/live_pose_classifier.py` is intended for integration with the PickleSense dashboard.

The intended live workflow is:

```text
START RECORDING
      ↓
Collect MediaPipe pose frames
      ↓
Participant performs one stroke
      ↓
STOP RECORDING
      ↓
Pose-TCN inference
      ↓
Dashboard prediction
```

The first version assumes one isolated stroke per recording.

Automatic continuous stroke segmentation is considered a future upgrade.

# Recommended Training Workflow

Run these commands in order:

```powershell
python -m scripts.build_manifest
python -m scripts.prepare_dataset
python -m scripts.train_model
python -m scripts.evaluate_model
```

Do not train the model until the dataset validator and participant split complete successfully.

# Important Evaluation Rule

Never randomly split individual pose frames between training and testing.

Testing must use participants that were not used during training.

This ensures the reported performance measures whether PickleSense generalizes to unseen players rather than memorizing individual participants.