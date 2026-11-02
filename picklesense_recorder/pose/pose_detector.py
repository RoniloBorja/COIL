from pathlib import Path

import cv2
import mediapipe as mp
from processing.joint_angles import calculate_joint_angles
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from models.pose_sample import PoseSample


PROJECT_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_MODEL_PATH = (
    PROJECT_ROOT
    / "assets"
    / "models"
    / "pose_landmarker_lite.task"
)


class PoseDetector:
    def __init__(
        self,
        model_path: Path = DEFAULT_MODEL_PATH,
    ):
        self.model_path = Path(model_path)

        self.landmarker = None

        self.frames_processed = 0
        self.frames_with_pose = 0
        self.frames_without_pose = 0

        self.running = False

    def start(self):
        if self.running:
            return

        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Pose model not found: "
                f"{self.model_path}"
            )

        base_options = python.BaseOptions(
            model_asset_path=str(
                self.model_path
            )
        )

        options = (
            vision.PoseLandmarkerOptions(
                base_options=base_options,
                running_mode=(
                    vision.RunningMode.VIDEO
                ),
                num_poses=1,
                min_pose_detection_confidence=0.5,
                min_pose_presence_confidence=0.5,
                min_tracking_confidence=0.5,
                output_segmentation_masks=False,
            )
        )

        self.landmarker = (
            vision.PoseLandmarker.create_from_options(
                options
            )
        )

        self.frames_processed = 0
        self.frames_with_pose = 0
        self.frames_without_pose = 0

        self.running = True

    def process(
        self,
        frame,
        frame_number: int,
        laptop_timestamp_ns: int,
    ) -> PoseSample:
        if (
            not self.running
            or self.landmarker is None
        ):
            raise RuntimeError(
                "PoseDetector has not been started"
            )

        rgb_frame = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB,
        )

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb_frame,
        )

        timestamp_ms = (
            laptop_timestamp_ns
            // 1_000_000
        )

        result = (
            self.landmarker.detect_for_video(
                mp_image,
                timestamp_ms,
            )
        )

        self.frames_processed += 1

        if not result.pose_landmarks:
            self.frames_without_pose += 1

            return PoseSample(
                frame_number=frame_number,
                laptop_timestamp_ns=(
                    laptop_timestamp_ns
                ),
                pose_detected=False,
                landmarks=None,
                world_landmarks=None,
                joint_angles=None,
            )

        self.frames_with_pose += 1

        # --------------------------------------------------
        # Normalized landmarks
        # --------------------------------------------------

        detected_pose = (
            result.pose_landmarks[0]
        )

        landmarks = []

        for landmark in detected_pose:
            landmarks.append(
                (
                    float(landmark.x),
                    float(landmark.y),
                    float(landmark.z),
                    float(
                        landmark.visibility
                        if landmark.visibility is not None
                        else 0.0
                    ),
                )
            )

        # --------------------------------------------------
        # World landmarks
        # --------------------------------------------------

        world_landmarks = None

        if result.pose_world_landmarks:
            detected_world_pose = (
                result.pose_world_landmarks[0]
            )

            world_landmarks = []

            for landmark in detected_world_pose:
                world_landmarks.append(
                    (
                        float(landmark.x),
                        float(landmark.y),
                        float(landmark.z),
                        float(
                            landmark.visibility
                            if landmark.visibility is not None
                            else 0.0
                        ),
                    )
                )

        # --------------------------------------------------
        # Derived joint angles
        # --------------------------------------------------

        joint_angles = calculate_joint_angles(
            world_landmarks
        )

        return PoseSample(
            frame_number=frame_number,
            laptop_timestamp_ns=(
                laptop_timestamp_ns
            ),
            pose_detected=True,
            landmarks=landmarks,
            world_landmarks=world_landmarks,
            joint_angles=joint_angles,
        )

    def get_status(self):
        return {
            "running": self.running,
            "frames_processed": (
                self.frames_processed
            ),
            "frames_with_pose": (
                self.frames_with_pose
            ),
            "frames_without_pose": (
                self.frames_without_pose
            ),
        }

    def stop(self):
        self.running = False

        if self.landmarker is not None:
            self.landmarker.close()
            self.landmarker = None