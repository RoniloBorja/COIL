from pathlib import Path


# --------------------------------------------------
# Network / ESP32
# --------------------------------------------------

UDP_BIND_IP = "192.168.1.10"
UDP_PORT = 5005

EXPECTED_IMU_RATE_HZ = 100


# --------------------------------------------------
# Camera
# --------------------------------------------------

CAMERA_INDEX = 0

CAMERA_WIDTH = 1280
CAMERA_HEIGHT = 720

EXPECTED_CAMERA_FPS = 30


# --------------------------------------------------
# Recording
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

RECORDINGS_ROOT = (
    PROJECT_ROOT
    / "PickleSense Recordings"
)


# --------------------------------------------------
# UDP Receiver
# --------------------------------------------------

UDP_BUFFER_SIZE = 1024

UDP_SOCKET_TIMEOUT_SECONDS = 1.0


# --------------------------------------------------
# Status / Monitoring
# --------------------------------------------------

STATUS_UPDATE_INTERVAL_SECONDS = 1.0

IMU_CONNECTION_TIMEOUT_SECONDS = 2.0


# --------------------------------------------------
# CSV File Names
# --------------------------------------------------

IMU_CSV_FILENAME = "imu_data.csv"
POSE_CSV_FILENAME = "pose_data.csv"
SESSION_INFO_FILENAME = "session_info.json"