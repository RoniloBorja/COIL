import time
import cv2

from config.settings import (
    CAMERA_INDEX,
    CAMERA_WIDTH,
    CAMERA_HEIGHT,
    EXPECTED_CAMERA_FPS,
)

def scan_available_cameras(
    max_index=10,
):
    available = []

    for index in range(max_index):
        capture = cv2.VideoCapture(
            index,
            cv2.CAP_DSHOW,
        )

        if capture.isOpened():
            success, _ = capture.read()

            if success:
                available.append(index)

        capture.release()

    return available

class CameraCapture:
    def __init__(
        self,
        camera_index=CAMERA_INDEX,
    ):
        self.camera_index = camera_index

        self.capture = None

        self.running = False

        self.frame_number = 0
        self.frames_received = 0

        self.last_frame_timestamp_ns = None

        self.current_fps = 0.0

        self._fps_start_ns = None
        self._fps_frame_count = 0

    def start(self):
        if self.running:
            return

        self.capture = cv2.VideoCapture(
            self.camera_index,
            cv2.CAP_DSHOW,
        )

        if not self.capture.isOpened():
            self.capture.release()
            self.capture = None

            raise RuntimeError(
                f"Could not open camera index {self.camera_index}"
                f"{CAMERA_INDEX}"
            )
        self.capture.set(
            cv2.CAP_PROP_FOURCC,
            cv2.VideoWriter_fourcc(*"MJPG"),
        )
        self.capture.set(
            cv2.CAP_PROP_FRAME_WIDTH,
            CAMERA_WIDTH,
        )

        self.capture.set(
            cv2.CAP_PROP_FRAME_HEIGHT,
            CAMERA_HEIGHT,
        )

        self.capture.set(
            cv2.CAP_PROP_FPS,
            EXPECTED_CAMERA_FPS,
        )

        self.frame_number = 0
        self.frames_received = 0

        self.last_frame_timestamp_ns = None

        self.current_fps = 0.0

        self._fps_start_ns = None
        self._fps_frame_count = 0

        self.running = True

    def read(self):
        if (
            not self.running
            or self.capture is None
        ):
            return None

        success, frame = self.capture.read()

        if not success:
            return None

        timestamp_ns = time.monotonic_ns()

        self.frame_number += 1
        self.frames_received += 1

        self.last_frame_timestamp_ns = (
            timestamp_ns
        )

        self._update_fps(timestamp_ns)

        return {
            "frame_number": self.frame_number,
            "laptop_timestamp_ns": timestamp_ns,
            "frame": frame,
        }

    def _update_fps(
        self,
        timestamp_ns: int,
    ):
        if self._fps_start_ns is None:
            self._fps_start_ns = timestamp_ns
            self._fps_frame_count = 1
            return

        self._fps_frame_count += 1

        elapsed_ns = (
            timestamp_ns
            - self._fps_start_ns
        )

        if elapsed_ns >= 1_000_000_000:
            self.current_fps = (
                (self._fps_frame_count - 1)
                * 1_000_000_000
                / elapsed_ns
            )

            self._fps_start_ns = timestamp_ns
            self._fps_frame_count = 1

    def get_status(self):
        if self.last_frame_timestamp_ns is None:
            last_frame_age_seconds = None

        else:
            last_frame_age_seconds = (
                time.monotonic_ns()
                - self.last_frame_timestamp_ns
            ) / 1_000_000_000

        return {
            "running": self.running,
            "frames_received": self.frames_received,
            "current_fps": self.current_fps,
            "last_frame_age_seconds": (
                last_frame_age_seconds
            ),
        }

    def stop(self):
        self.running = False

        if self.capture is not None:
            self.capture.release()
            self.capture = None
    def set_camera_index(
        self,
        camera_index: int,
    ):
        if self.running:
            raise RuntimeError(
                "Camera must be stopped before "
                "changing camera source"
            )

        self.camera_index = int(
            camera_index
        )