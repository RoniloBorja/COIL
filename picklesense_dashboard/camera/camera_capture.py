import cv2
import threading
import time

from config.settings import (
    CAMERA_INDEX,
    CAMERA_WIDTH,
    CAMERA_HEIGHT,
    EXPECTED_CAMERA_FPS,
)


class CameraCapture:
    def __init__(
        self,
        camera_index=CAMERA_INDEX,
    ):
        self.camera_index = int(
            camera_index
        )

        self.capture = None

        self.running = False
        self.thread = None

        self.latest_frame = None
        self.latest_timestamp_ns = None

        self.frames_received = 0

        self.current_fps = 0.0

        self._fps_window_start_ns = None
        self._fps_window_frames = 0

        self._lock = (
            threading.Lock()
        )

    def start(
        self,
    ):
        if self.running:
            return

        self.capture = (
            cv2.VideoCapture(
                self.camera_index,
                cv2.CAP_DSHOW,
            )
        )

        if not self.capture.isOpened():
            self.capture.release()

            self.capture = None

            raise RuntimeError(
                f"Could not open camera "
                f"{self.camera_index}"
            )

        self.capture.set(
            cv2.CAP_PROP_FOURCC,
            cv2.VideoWriter_fourcc(
                *"MJPG"
            ),
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

        self.running = True

        self._fps_window_start_ns = (
            time.monotonic_ns()
        )

        self._fps_window_frames = 0

        self.thread = (
            threading.Thread(
                target=self._capture_loop,
                daemon=True,
            )
        )

        self.thread.start()

    def stop(
        self,
    ):
        self.running = False

        if (
            self.thread is not None
            and self.thread.is_alive()
        ):
            self.thread.join(
                timeout=1.0
            )

        self.thread = None

        if self.capture is not None:
            try:
                self.capture.release()

            except Exception:
                pass

        self.capture = None

    def set_camera_index(
        self,
        camera_index,
    ):
        if self.running:
            raise RuntimeError(
                "Camera must be stopped "
                "before changing source."
            )

        self.camera_index = int(
            camera_index
        )

    def _capture_loop(
        self,
    ):
        while self.running:
            if self.capture is None:
                break

            success, frame = (
                self.capture.read()
            )

            if not success:
                continue

            timestamp_ns = (
                time.monotonic_ns()
            )

            with self._lock:
                self.latest_frame = (
                    frame
                )

                self.latest_timestamp_ns = (
                    timestamp_ns
                )

                self.frames_received += 1

            self._update_fps(
                timestamp_ns
            )

    def _update_fps(
        self,
        now_ns,
    ):
        self._fps_window_frames += 1

        if (
            self._fps_window_start_ns
            is None
        ):
            self._fps_window_start_ns = (
                now_ns
            )

            return

        elapsed_ns = (
            now_ns
            - self._fps_window_start_ns
        )

        if elapsed_ns >= 1_000_000_000:
            elapsed_seconds = (
                elapsed_ns
                / 1_000_000_000
            )

            self.current_fps = (
                self._fps_window_frames
                / elapsed_seconds
            )

            self._fps_window_frames = 0

            self._fps_window_start_ns = (
                now_ns
            )

    def get_latest_frame(
        self,
    ):
        with self._lock:
            if self.latest_frame is None:
                return None

            return {
                "frame": (
                    self.latest_frame.copy()
                ),

                "timestamp_ns": (
                    self.latest_timestamp_ns
                ),
            }

    def get_status(
        self,
    ):
        with self._lock:
            last_frame_age_seconds = None

            if (
                self.latest_timestamp_ns
                is not None
            ):
                last_frame_age_seconds = (
                    time.monotonic_ns()
                    - self.latest_timestamp_ns
                ) / 1_000_000_000

            active = (
                self.running
                and self.capture is not None
                and self.capture.isOpened()
            )

            receiving = (
                active
                and last_frame_age_seconds
                is not None
                and last_frame_age_seconds
                < 2.0
            )

            return {
                "active": (
                    active
                ),

                "receiving": (
                    receiving
                ),

                "camera_index": (
                    self.camera_index
                ),

                "frames_received": (
                    self.frames_received
                ),

                "current_fps": (
                    self.current_fps
                ),

                "last_frame_age_seconds": (
                    last_frame_age_seconds
                ),
            }


def scan_available_cameras(
    max_index=10,
):
    available = []

    for index in range(
        max_index
    ):
        capture = (
            cv2.VideoCapture(
                index,
                cv2.CAP_DSHOW,
            )
        )

        if capture.isOpened():
            success, _ = (
                capture.read()
            )

            if success:
                available.append(
                    index
                )

        capture.release()

    return available