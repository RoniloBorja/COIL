from collections import deque
from pathlib import Path

import numpy as np
import torch

from config.settings import (
    POSE_TCN_MODEL_PATH,
    POSE_NORMALIZER_PATH,
    STROKE_CLASSES,
)

from inference.prediction_state import (
    PredictionState,
)


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

SEQUENCE_LENGTH = 64


class PosePredictor:
    def __init__(
        self,
        prediction_state: PredictionState,
        model=None,
        normalizer=None,
        device=None,
    ):
        self.prediction_state = (
            prediction_state
        )

        if device is None:
            device = (
                "cuda"
                if torch.cuda.is_available()
                else "cpu"
            )

        self.device = torch.device(
            device
        )

        self.model = model
        self.normalizer = normalizer

        self.frame_buffer = deque(
            maxlen=SEQUENCE_LENGTH
        )

        self.ready = False

    def load(
        self,
    ):
        """
        Load the trained Pose-TCN and
        normalizer.

        This method is kept separate so the
        dashboard can still launch before the
        trained model files exist.
        """

        if not Path(
            POSE_TCN_MODEL_PATH
        ).exists():
            raise FileNotFoundError(
                "Pose-TCN model not found:\n"
                f"{POSE_TCN_MODEL_PATH}"
            )

        if not Path(
            POSE_NORMALIZER_PATH
        ).exists():
            raise FileNotFoundError(
                "Pose normalizer not found:\n"
                f"{POSE_NORMALIZER_PATH}"
            )

        # -------------------------------------------------
        # Import from the pose training project later.
        #
        # For now, this file defines the runtime interface.
        # -------------------------------------------------

        raise NotImplementedError(
            "Pose model loading will be connected "
            "after the trained Pose-TCN model is available."
        )

    def reset(
        self,
    ):
        self.frame_buffer.clear()

        self.prediction_state.clear_vision()

    def add_pose_frame(
        self,
        pose_result,
    ):
        """
        Accept one result from camera/pose_detector.py.

        pose_result must contain:

            pose_detected
            world_landmarks
            timestamp_ns
        """

        if not pose_result:
            return False

        if not pose_result.get(
            "pose_detected",
            False,
        ):
            return False

        world_landmarks = (
            pose_result.get(
                "world_landmarks"
            )
        )

        if world_landmarks is None:
            return False

        if len(world_landmarks) < 29:
            return False

        features = (
            self._extract_features(
                world_landmarks
            )
        )

        if features is None:
            return False

        self.frame_buffer.append(
            features
        )

        return True

    def _extract_features(
        self,
        world_landmarks,
    ):
        """
        Build the 36 world-coordinate features
        used by the Pose-TCN skeleton input.

        Joint angles will be added when the
        deployed runtime is connected to the
        same preprocessing code used during
        training.
        """

        left_hip = np.array(
            world_landmarks[23][:3],
            dtype=np.float32,
        )

        right_hip = np.array(
            world_landmarks[24][:3],
            dtype=np.float32,
        )

        pelvis_center = (
            left_hip
            + right_hip
        ) / 2.0

        left_shoulder = np.array(
            world_landmarks[11][:3],
            dtype=np.float32,
        )

        right_shoulder = np.array(
            world_landmarks[12][:3],
            dtype=np.float32,
        )

        shoulder_width = np.linalg.norm(
            left_shoulder
            - right_shoulder
        )

        if shoulder_width < 1e-6:
            return None

        feature_values = []

        for landmark_index in (
            SELECTED_LANDMARKS
        ):
            xyz = np.array(
                world_landmarks[
                    landmark_index
                ][:3],
                dtype=np.float32,
            )

            xyz = (
                xyz
                - pelvis_center
            )

            xyz = (
                xyz
                / shoulder_width
            )

            feature_values.extend(
                xyz.tolist()
            )

        return np.asarray(
            feature_values,
            dtype=np.float32,
        )

    def can_predict(
        self,
    ):
        return (
            len(
                self.frame_buffer
            )
            >= SEQUENCE_LENGTH
        )

    @torch.no_grad()
    def predict(
        self,
    ):
        """
        Placeholder runtime hook.

        Once the model is trained, this will:

        1. append the 8 joint-angle features
        2. apply pose_normalizer.json
        3. reshape to [1, 44, 64]
        4. run Pose-TCN
        5. update PredictionState
        """

        if not self.can_predict():
            return None

        if (
            self.model is None
            or self.normalizer is None
        ):
            return None

        sequence = np.asarray(
            self.frame_buffer,
            dtype=np.float32,
        )

        # Expected final shape after the
        # complete preprocessing pipeline:
        #
        # [64, 44]
        #
        # then:
        #
        # [1, 44, 64]

        sequence = (
            self.normalizer.transform(
                sequence
            )
        )

        tensor = (
            torch.from_numpy(
                sequence.T
            )
            .float()
            .unsqueeze(0)
            .to(
                self.device
            )
        )

        logits = self.model(
            tensor
        )

        probabilities = (
            torch.softmax(
                logits,
                dim=1,
            )
            .cpu()
            .numpy()[0]
        )

        predicted_index = int(
            np.argmax(
                probabilities
            )
        )

        predicted_class = (
            STROKE_CLASSES[
                predicted_index
            ]
        )

        confidence = float(
            probabilities[
                predicted_index
            ]
        )

        probability_map = {
            STROKE_CLASSES[index]: float(
                probabilities[index]
            )
            for index
            in range(
                len(
                    STROKE_CLASSES
                )
            )
        }

        self.prediction_state.update_vision(
            class_name=predicted_class,
            confidence=confidence,
            probabilities=probability_map,
        )

        return {
            "class_name": (
                predicted_class
            ),

            "confidence": (
                confidence
            ),

            "probabilities": (
                probability_map
            ),
        }

    def get_buffer_status(
        self,
    ):
        return {
            "frames": len(
                self.frame_buffer
            ),

            "required_frames": (
                SEQUENCE_LENGTH
            ),

            "ready": (
                self.can_predict()
            ),
        }