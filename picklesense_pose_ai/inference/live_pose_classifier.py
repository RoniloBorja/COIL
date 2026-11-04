from collections import deque

import numpy as np
import pandas as pd
import torch

from config.settings import (
    SEQUENCE_LENGTH,
    STROKE_CLASSES,
    INDEX_TO_CLASS,
)

from preprocessing.pose_features import (
    extract_pose_features,
)

from preprocessing.pose_normalizer import (
    PoseFeatureNormalizer,
)

from preprocessing.resampler import (
    resample_sequence,
    to_model_input,
)

from models.pose_tcn import (
    create_model,
)


class LivePoseStrokeClassifier:
    def __init__(
        self,
        model_path,
        normalizer_path=None,
        device=None,
        minimum_frames=20,
    ):
        if device is None:
            device = (
                "cuda"
                if torch.cuda.is_available()
                else "cpu"
            )

        self.device = torch.device(
            device
        )

        self.minimum_frames = (
            minimum_frames
        )

        self.frame_buffer = deque(
            maxlen=SEQUENCE_LENGTH * 2
        )

        self.normalizer = (
            PoseFeatureNormalizer()
        )

        self.normalizer.load(
            normalizer_path
        )

        self.model = (
            create_model()
            .to(
                self.device
            )
        )

        state_dict = torch.load(
            model_path,
            map_location=self.device,
        )

        self.model.load_state_dict(
            state_dict
        )

        self.model.eval()

        self.last_result = None

    def reset(self):
        """
        Clear all accumulated frames.

        Call this before collecting a new
        isolated stroke.
        """

        self.frame_buffer.clear()

        self.last_result = None

    def add_frame(
        self,
        frame_data,
    ):
        """
        Add one pose-data row.

        frame_data can be:

        - dictionary
        - pandas Series

        It should contain the same columns
        used in pose_data.csv.
        """

        if isinstance(
            frame_data,
            pd.Series,
        ):
            frame_data = (
                frame_data.to_dict()
            )

        if not isinstance(
            frame_data,
            dict,
        ):
            raise TypeError(
                "frame_data must be "
                "a dictionary or pandas Series."
            )

        pose_detected = (
            frame_data.get(
                "pose_detected",
                1,
            )
        )

        try:
            pose_detected = bool(
                int(
                    pose_detected
                )
            )

        except Exception:
            pose_detected = bool(
                pose_detected
            )

        if not pose_detected:
            return False

        self.frame_buffer.append(
            dict(
                frame_data
            )
        )

        return True

    def frame_count(self):
        return len(
            self.frame_buffer
        )

    def ready(self):
        return (
            len(
                self.frame_buffer
            )
            >= self.minimum_frames
        )

    def _build_input_tensor(self):
        if not self.ready():
            raise RuntimeError(
                "Not enough pose frames "
                "for classification.\n"
                f"Current: {self.frame_count()}\n"
                f"Required: {self.minimum_frames}"
            )

        dataframe = pd.DataFrame(
            list(
                self.frame_buffer
            )
        )

        # -----------------------------------------
        # Same feature extraction used in training
        # -----------------------------------------

        features = (
            extract_pose_features(
                dataframe
            )
        )

        # -----------------------------------------
        # Same saved training normalization
        # -----------------------------------------

        features = (
            self.normalizer.transform(
                features
            )
        )

        # -----------------------------------------
        # Convert variable live sequence
        # to the model's 64-frame representation
        # -----------------------------------------

        features = (
            resample_sequence(
                features,
                target_length=(
                    SEQUENCE_LENGTH
                ),
            )
        )

        features = (
            to_model_input(
                features
            )
        )

        tensor = (
            torch.from_numpy(
                features
            )
            .float()
            .unsqueeze(0)
            .to(
                self.device
            )
        )

        return tensor

    @torch.no_grad()
    def predict(self):
        """
        Run classification using all currently
        buffered frames.

        Returns None when there are not yet
        enough frames.
        """

        if not self.ready():
            return None

        input_tensor = (
            self._build_input_tensor()
        )

        logits = self.model(
            input_tensor
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
            INDEX_TO_CLASS[
                predicted_index
            ]
        )

        confidence = float(
            probabilities[
                predicted_index
            ]
        )

        result = {
            "predicted_index": (
                predicted_index
            ),

            "predicted_class": (
                predicted_class
            ),

            "confidence": (
                confidence
            ),

            "probabilities": {
                STROKE_CLASSES[
                    index
                ]: float(
                    probabilities[
                        index
                    ]
                )

                for index
                in range(
                    len(
                        STROKE_CLASSES
                    )
                )
            },

            "frames_used": (
                self.frame_count()
            ),
        }

        self.last_result = (
            result
        )

        return result

    def classify_completed_stroke(
        self,
    ):
        """
        Intended for your current PickleSense
        workflow:

        START
        ↓
        collect one stroke
        ↓
        STOP
        ↓
        classify accumulated pose sequence

        After prediction, the buffer is cleared.
        """

        result = self.predict()

        if result is None:
            return None

        self.reset()

        return result


def print_live_result(
    result,
):
    if result is None:
        print(
            "Not enough frames "
            "for classification."
        )
        return

    print()
    print(
        "Live Pose Prediction"
    )

    print(
        "--------------------"
    )

    print(
        f"Stroke: "
        f"{result['predicted_class']}"
    )

    print(
        f"Confidence: "
        f"{result['confidence'] * 100:.2f}%"
    )

    print(
        f"Frames used: "
        f"{result['frames_used']}"
    )

    print()

    for (
        class_name,
        probability,
    ) in result[
        "probabilities"
    ].items():

        print(
            f"{class_name:10s}: "
            f"{probability * 100:6.2f}%"
        )