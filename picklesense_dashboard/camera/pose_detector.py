from pathlib import Path

import cv2
import mediapipe as mp

from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from config.settings import (
    POSE_LANDMARKER_MODEL,
    POSE_MIN_DETECTION_CONFIDENCE,
    POSE_MIN_PRESENCE_CONFIDENCE,
    POSE_MIN_TRACKING_CONFIDENCE,
    POSE_NUM_POSES,
)


class PoseDetector:
    def __init__(
        self,
        model_path=POSE_LANDMARKER_MODEL,
    ):
        self.model_path = Path(
            model_path
        )

        self.detector = None

        self.running = False

        self.frames_processed = 0
        self.frames_with_pose = 0
        self.frames_without_pose = 0

    def start(
        self,
    ):
        if self.running:
            return

        if not self.model_path.exists():
            raise FileNotFoundError(
                "MediaPipe pose model not found:\n"
                f"{self.model_path}"
            )

        base_options = (
            python.BaseOptions(
                model_asset_path=str(
                    self.model_path
                )
            )
        )

        options = (
            vision.PoseLandmarkerOptions(
                base_options=(
                    base_options
                ),

                running_mode=(
                    vision.RunningMode.VIDEO
                ),

                num_poses=(
                    POSE_NUM_POSES
                ),

                min_pose_detection_confidence=(
                    POSE_MIN_DETECTION_CONFIDENCE
                ),

                min_pose_presence_confidence=(
                    POSE_MIN_PRESENCE_CONFIDENCE
                ),

                min_tracking_confidence=(
                    POSE_MIN_TRACKING_CONFIDENCE
                ),

                output_segmentation_masks=False,
            )
        )

        self.detector = (
            vision.PoseLandmarker.create_from_options(
                options
            )
        )

        self.running = True

    def stop(
        self,
    ):
        self.running = False

        if self.detector is not None:
            try:
                self.detector.close()

            except Exception:
                pass

        self.detector = None

    def process(
        self,
        frame,
        timestamp_ns,
    ):
        if not self.running:
            raise RuntimeError(
                "Pose detector is not running."
            )

        if self.detector is None:
            raise RuntimeError(
                "Pose detector has not been initialized."
            )

        # --------------------------------------------------
        # OpenCV BGR → RGB
        # --------------------------------------------------

        rgb_frame = (
            cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2RGB,
            )
        )

        mp_image = mp.Image(
            image_format=(
                mp.ImageFormat.SRGB
            ),
            data=rgb_frame,
        )

        # MediaPipe VIDEO mode expects milliseconds.
        timestamp_ms = (
            timestamp_ns
            // 1_000_000
        )

        result = (
            self.detector.detect_for_video(
                mp_image,
                int(
                    timestamp_ms
                ),
            )
        )

        self.frames_processed += 1

        # --------------------------------------------------
        # No pose detected
        # --------------------------------------------------

        if not result.pose_landmarks:
            self.frames_without_pose += 1

            return {
                "pose_detected": False,

                "timestamp_ns": (
                    timestamp_ns
                ),

                "landmarks": None,

                "world_landmarks": None,
            }

        self.frames_with_pose += 1

        # --------------------------------------------------
        # Normalized image landmarks
        # --------------------------------------------------

        landmarks = []

        pose_landmarks = (
            result.pose_landmarks[0]
        )

        for landmark in pose_landmarks:
            landmarks.append(
                (
                    float(
                        landmark.x
                    ),

                    float(
                        landmark.y
                    ),

                    float(
                        landmark.z
                    ),

                    float(
                        landmark.visibility
                    ),
                )
            )

        # --------------------------------------------------
        # World landmarks
        # --------------------------------------------------

        world_landmarks = None

        if result.pose_world_landmarks:
            world_landmarks = []

            pose_world_landmarks = (
                result.pose_world_landmarks[0]
            )

            for landmark in pose_world_landmarks:
                world_landmarks.append(
                    (
                        float(
                            landmark.x
                        ),

                        float(
                            landmark.y
                        ),

                        float(
                            landmark.z
                        ),

                        float(
                            landmark.visibility
                        ),
                    )
                )

        return {
            "pose_detected": True,

            "timestamp_ns": (
                timestamp_ns
            ),

            "landmarks": (
                landmarks
            ),

            "world_landmarks": (
                world_landmarks
            ),
        }

    def get_status(
        self,
    ):
        detection_rate = 0.0

        if self.frames_processed > 0:
            detection_rate = (
                self.frames_with_pose
                / self.frames_processed
            )

        return {
            "running": (
                self.running
            ),

            "frames_processed": (
                self.frames_processed
            ),

            "frames_with_pose": (
                self.frames_with_pose
            ),

            "frames_without_pose": (
                self.frames_without_pose
            ),

            "detection_rate": (
                detection_rate
            ),
        }