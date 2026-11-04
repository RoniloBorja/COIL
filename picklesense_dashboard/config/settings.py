from pathlib import Path


# ---------------------------------------------------------
# Project paths
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

ASSETS_DIR = PROJECT_ROOT / "assets"
MODELS_DIR = ASSETS_DIR / "models"

TRAINED_MODELS_DIR = (
    PROJECT_ROOT
    / "trained_models"
)

POSE_MODEL_DIR = (
    TRAINED_MODELS_DIR
    / "pose"
)


# ---------------------------------------------------------
# MediaPipe model
# ---------------------------------------------------------

POSE_LANDMARKER_MODEL = (
    MODELS_DIR
    / "pose_landmarker_lite.task"
)


# ---------------------------------------------------------
# Pose-TCN trained model
# ---------------------------------------------------------

POSE_TCN_MODEL_PATH = (
    POSE_MODEL_DIR
    / "picklesense_pose_tcn_final.pt"
)

POSE_NORMALIZER_PATH = (
    POSE_MODEL_DIR
    / "pose_normalizer.json"
)


# ---------------------------------------------------------
# Camera
# ---------------------------------------------------------

CAMERA_INDEX = 0

CAMERA_WIDTH = 1280
CAMERA_HEIGHT = 720

EXPECTED_CAMERA_FPS = 30


# ---------------------------------------------------------
# MediaPipe Pose
# ---------------------------------------------------------

POSE_MIN_DETECTION_CONFIDENCE = 0.5
POSE_MIN_PRESENCE_CONFIDENCE = 0.5
POSE_MIN_TRACKING_CONFIDENCE = 0.5

POSE_NUM_POSES = 1


# ---------------------------------------------------------
# MAX78000 communication
# ---------------------------------------------------------
#
# Initial design:
# MAX78000 sends data to laptop.
#
# We will keep the transport configurable so
# UART / USB serial can be changed later.
# ---------------------------------------------------------

MAX78000_SERIAL_PORT = "COM12"

MAX78000_BAUD_RATE = 115200

MAX78000_SERIAL_TIMEOUT_SECONDS = 0.1


# ---------------------------------------------------------
# MPU6500
# ---------------------------------------------------------

EXPECTED_IMU_RATE_HZ = 100

IMU_WINDOW_SIZE = 128


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

NUM_STROKE_CLASSES = len(
    STROKE_CLASSES
)


# ---------------------------------------------------------
# IMU scale configuration
# ---------------------------------------------------------
#
# Initial planned sensor configuration:
#
# Accelerometer:
# ±2 g
#
# Gyroscope:
# ±2000 degrees / second
#
# MPU6500 scale factors:
#
# ±2 g      = 16384 LSB/g
# ±2000 dps = 16.4 LSB/(deg/s)
# ---------------------------------------------------------

ACCEL_LSB_PER_G = 16384.0

GYRO_LSB_PER_DPS = 16.4


# ---------------------------------------------------------
# Dashboard update rates
# ---------------------------------------------------------

UI_UPDATE_INTERVAL_MS = 50

STATUS_UPDATE_INTERVAL_MS = 250


# ---------------------------------------------------------
# Prediction display
# ---------------------------------------------------------

MIN_PREDICTION_CONFIDENCE = 0.50


# ---------------------------------------------------------
# Window
# ---------------------------------------------------------

WINDOW_TITLE = "PickleSense"

WINDOW_WIDTH = 1400
WINDOW_HEIGHT = 850


# ---------------------------------------------------------
# Dashboard state defaults
# ---------------------------------------------------------

DEFAULT_PREDICTION_TEXT = "WAITING"

DEFAULT_CONFIDENCE_TEXT = "--"

DEFAULT_CONNECTION_TEXT = "Disconnected"