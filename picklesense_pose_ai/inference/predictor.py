from pathlib import Path

import numpy as np
import pandas as pd
import torch

from config.settings import (
    MODEL_DIR,
    FINAL_MODEL_FILENAME,
    INDEX_TO_CLASS,
    STROKE_CLASSES,
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


class PoseStrokePredictor:
    def __init__(
        self,
        model_path=None,
        normalizer_path=None,
        device=None,
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

        if model_path is None:
            model_path = (
                MODEL_DIR
                / FINAL_MODEL_FILENAME
            )

        self.model_path = Path(
            model_path
        )

        if not self.model_path.exists():
            raise FileNotFoundError(
                "Trained model not found:\n"
                f"{self.model_path}"
            )

        # -----------------------------------------
        # Load normalizer
        # -----------------------------------------

        self.normalizer = (
            PoseFeatureNormalizer()
        )

        self.normalizer.load(
            normalizer_path
        )

        # -----------------------------------------
        # Load model
        # -----------------------------------------

        self.model = (
            create_model()
            .to(
                self.device
            )
        )

        state_dict = torch.load(
            self.model_path,
            map_location=self.device,
        )

        self.model.load_state_dict(
            state_dict
        )

        self.model.eval()

    def preprocess_dataframe(
        self,
        dataframe,
    ):
        # -----------------------------------------
        # Extract 44 features
        #
        # [variable frames, 44]
        # -----------------------------------------

        features = (
            extract_pose_features(
                dataframe
            )
        )

        # -----------------------------------------
        # Apply training normalizer
        # -----------------------------------------

        features = (
            self.normalizer.transform(
                features
            )
        )

        # -----------------------------------------
        # Resample
        #
        # [variable, 44]
        #      ↓
        # [64, 44]
        # -----------------------------------------

        features = (
            resample_sequence(
                features
            )
        )

        # -----------------------------------------
        # Conv1D format
        #
        # [64, 44]
        #      ↓
        # [44, 64]
        # -----------------------------------------

        features = (
            to_model_input(
                features
            )
        )

        # Add batch dimension:
        #
        # [44, 64]
        #      ↓
        # [1, 44, 64]

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
    def predict_dataframe(
        self,
        dataframe,
    ):
        input_tensor = (
            self.preprocess_dataframe(
                dataframe
            )
        )

        logits = self.model(
            input_tensor
        )

        probabilities = (
            torch.softmax(
                logits,
                dim=1,
            )
        )

        probabilities = (
            probabilities
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

        class_probabilities = {
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

        return {
            "predicted_index": (
                predicted_index
            ),

            "predicted_class": (
                predicted_class
            ),

            "confidence": (
                confidence
            ),

            "probabilities": (
                class_probabilities
            ),
        }

    def predict_csv(
        self,
        pose_csv_path,
    ):
        pose_csv_path = Path(
            pose_csv_path
        )

        if not pose_csv_path.exists():
            raise FileNotFoundError(
                f"Pose CSV not found: "
                f"{pose_csv_path}"
            )

        dataframe = pd.read_csv(
            pose_csv_path
        )

        return self.predict_dataframe(
            dataframe
        )


def print_prediction(
    result,
):
    print()
    print(
        "PickleSense Pose Prediction"
    )

    print(
        "---------------------------"
    )

    print(
        f"Prediction: "
        f"{result['predicted_class']}"
    )

    print(
        f"Confidence: "
        f"{result['confidence'] * 100:.2f}%"
    )

    print()
    print(
        "Class probabilities:"
    )

    for class_name, probability in (
        result[
            "probabilities"
        ].items()
    ):
        print(
            f"  {class_name:10s}: "
            f"{probability * 100:6.2f}%"
        )


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description=(
            "Run PickleSense Pose-TCN "
            "inference on a pose_data.csv file."
        )
    )

    parser.add_argument(
        "pose_csv",
        type=str,
        help=(
            "Path to pose_data.csv"
        ),
    )

    args = parser.parse_args()

    predictor = (
        PoseStrokePredictor()
    )

    result = predictor.predict_csv(
        args.pose_csv
    )

    print_prediction(
        result
    )


if __name__ == "__main__":
    main()