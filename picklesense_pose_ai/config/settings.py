from pathlib import Path


# ---------------------------------------------------------
# Project paths
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"

RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MANIFEST_DIR = DATA_DIR / "manifests"

OUTPUT_DIR = PROJECT_ROOT / "outputs"

CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"
MODEL_DIR = OUTPUT_DIR / "models"
PLOT_DIR = OUTPUT_DIR / "plots"
REPORT_DIR = OUTPUT_DIR / "reports"
LOG_DIR = OUTPUT_DIR / "logs"


# ---------------------------------------------------------
# Stroke classes
# ---------------------------------------------------------

STROKE_CLASSES = [
    "Forehand",
    "Backhand",
    "Serve",
    "Volley",
    "Dink",
    "Overhead",
]

CLASS_TO_INDEX = {
    class_name: index
    for index, class_name in enumerate(STROKE_CLASSES)
}

INDEX_TO_CLASS = {
    index: class_name
    for class_name, index in CLASS_TO_INDEX.items()
}

NUM_CLASSES = len(STROKE_CLASSES)


# ---------------------------------------------------------
# Selected MediaPipe Pose landmarks
# ---------------------------------------------------------
#
# MediaPipe indices:
#
# 11 = left shoulder
# 12 = right shoulder
# 13 = left elbow
# 14 = right elbow
# 15 = left wrist
# 16 = right wrist
# 23 = left hip
# 24 = right hip
# 25 = left knee
# 26 = right knee
# 27 = left ankle
# 28 = right ankle
#

SELECTED_LANDMARKS = [
    11,
    12,
    13,
    14,
    15,
    16,
    23,
    24,
    25,
    26,
    27,
    28,
]

NUM_SELECTED_LANDMARKS = len(
    SELECTED_LANDMARKS
)


# ---------------------------------------------------------
# Joint-angle features
# ---------------------------------------------------------

JOINT_ANGLE_FIELDS = [
    "left_elbow_angle",
    "right_elbow_angle",
    "left_shoulder_angle",
    "right_shoulder_angle",
    "left_hip_angle",
    "right_hip_angle",
    "left_knee_angle",
    "right_knee_angle",
]


# ---------------------------------------------------------
# Pose feature dimensions
# ---------------------------------------------------------

WORLD_COORDS_PER_LANDMARK = 3

LANDMARK_FEATURE_COUNT = (
    NUM_SELECTED_LANDMARKS
    * WORLD_COORDS_PER_LANDMARK
)

ANGLE_FEATURE_COUNT = len(
    JOINT_ANGLE_FIELDS
)

FEATURE_COUNT = (
    LANDMARK_FEATURE_COUNT
    + ANGLE_FEATURE_COUNT
)

# Expected:
# 12 landmarks × 3 XYZ = 36
# 8 angles             = 8
# Total                = 44


# ---------------------------------------------------------
# Temporal preprocessing
# ---------------------------------------------------------

SEQUENCE_LENGTH = 64

MIN_VALID_POSE_FRAMES = 10


# ---------------------------------------------------------
# Normalization
# ---------------------------------------------------------

LEFT_HIP_INDEX = 23
RIGHT_HIP_INDEX = 24

LEFT_SHOULDER_INDEX = 11
RIGHT_SHOULDER_INDEX = 12

NORMALIZE_POSITION = True
NORMALIZE_SCALE = True


# ---------------------------------------------------------
# Training
# ---------------------------------------------------------

BATCH_SIZE = 32

LEARNING_RATE = 0.001

WEIGHT_DECAY = 1e-4

MAX_EPOCHS = 100

EARLY_STOPPING_PATIENCE = 12

DROPOUT = 0.30

RANDOM_SEED = 42


# ---------------------------------------------------------
# DataLoader
# ---------------------------------------------------------

NUM_WORKERS = 0

PIN_MEMORY = True


# ---------------------------------------------------------
# Dataset split
# ---------------------------------------------------------
#
# These are fallback ratios.
#
# We will prefer participant-based splitting instead
# of randomly splitting individual trials.
#

TRAIN_RATIO = 0.70
VALIDATION_RATIO = 0.15
TEST_RATIO = 0.15


# ---------------------------------------------------------
# Filenames
# ---------------------------------------------------------

POSE_FILENAME = "pose_data.csv"
SESSION_INFO_FILENAME = "session_info.json"

DATASET_MANIFEST_FILENAME = (
    "dataset_manifest.csv"
)

TRAIN_MANIFEST_FILENAME = (
    "train_manifest.csv"
)

VALIDATION_MANIFEST_FILENAME = (
    "validation_manifest.csv"
)

TEST_MANIFEST_FILENAME = (
    "test_manifest.csv"
)


# ---------------------------------------------------------
# Model output
# ---------------------------------------------------------

BEST_MODEL_FILENAME = (
    "picklesense_pose_tcn_best.pt"
)

FINAL_MODEL_FILENAME = (
    "picklesense_pose_tcn_final.pt"
)