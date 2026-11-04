import csv
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from config.settings import (
    CLASS_TO_INDEX,
    FEATURE_COUNT,
    SEQUENCE_LENGTH,
)

from preprocessing.pose_features import (
    load_pose_features,
)

from preprocessing.pose_normalizer import (
    PoseFeatureNormalizer,
)

from preprocessing.resampler import (
    resample_sequence,
    to_model_input,
)

from preprocessing.augmentation import (
    augment_pose_sequence,
)


class PickleSensePoseDataset(Dataset):
    def __init__(
        self,
        manifest_path,
        normalizer: PoseFeatureNormalizer,
        training=False,
        augment=False,
        random_seed=42,
    ):
        self.manifest_path = Path(
            manifest_path
        )

        self.normalizer = normalizer

        self.training = training
        self.augment = augment

        self.rng = np.random.default_rng(
            random_seed
        )

        if not self.manifest_path.exists():
            raise FileNotFoundError(
                f"Manifest not found: "
                f"{self.manifest_path}"
            )

        self.rows = (
            self._load_manifest()
        )

        if not self.rows:
            raise RuntimeError(
                f"No dataset rows found in "
                f"{self.manifest_path}"
            )

    def _load_manifest(self):
        rows = []

        with open(
            self.manifest_path,
            "r",
            encoding="utf-8",
            newline="",
        ) as file:
            reader = csv.DictReader(
                file
            )

            for row in reader:
                stroke_type = (
                    row.get(
                        "stroke_type",
                        "",
                    )
                    .strip()
                )

                if (
                    stroke_type
                    not in CLASS_TO_INDEX
                ):
                    print(
                        "Skipping unknown "
                        f"stroke class: "
                        f"{stroke_type}"
                    )
                    continue

                pose_path = row.get(
                    "pose_path",
                    "",
                )

                if not pose_path:
                    continue

                rows.append(
                    row
                )

        return rows

    def __len__(self):
        return len(
            self.rows
        )

    def __getitem__(
        self,
        index,
    ):
        row = self.rows[
            index
        ]

        pose_path = Path(
            row["pose_path"]
        )

        stroke_type = (
            row["stroke_type"]
            .strip()
        )

        label = CLASS_TO_INDEX[
            stroke_type
        ]

        # -----------------------------------------
        # 1. Extract 44 pose features
        #
        # Output:
        # [variable_frames, 44]
        # -----------------------------------------

        features = (
            load_pose_features(
                pose_path
            )
        )

        # -----------------------------------------
        # 2. Training-only augmentation
        # -----------------------------------------

        if (
            self.training
            and self.augment
        ):
            features = (
                augment_pose_sequence(
                    features,
                    rng=self.rng,
                )
            )

        # -----------------------------------------
        # 3. Dataset-level normalization
        #
        # IMPORTANT:
        # normalizer must already be fitted
        # using TRAINING DATA ONLY.
        # -----------------------------------------

        features = (
            self.normalizer.transform(
                features
            )
        )

        # -----------------------------------------
        # 4. Resample to 64 frames
        #
        # [variable, 44]
        #       ↓
        # [64, 44]
        # -----------------------------------------

        features = (
            resample_sequence(
                features,
                target_length=(
                    SEQUENCE_LENGTH
                ),
            )
        )

        # -----------------------------------------
        # 5. Convert for Conv1D
        #
        # [64, 44]
        #    ↓
        # [44, 64]
        # -----------------------------------------

        features = (
            to_model_input(
                features
            )
        )

        if features.shape != (
            FEATURE_COUNT,
            SEQUENCE_LENGTH,
        ):
            raise RuntimeError(
                "Unexpected model input "
                f"shape: {features.shape}"
            )

        # -----------------------------------------
        # 6. NumPy → PyTorch tensors
        # -----------------------------------------

        feature_tensor = (
            torch.from_numpy(
                features
            )
            .float()
        )

        label_tensor = torch.tensor(
            label,
            dtype=torch.long,
        )

        return {
            "features": (
                feature_tensor
            ),

            "label": (
                label_tensor
            ),

            "participant_id": (
                row.get(
                    "participant_id",
                    "",
                )
            ),

            "session_number": (
                row.get(
                    "session_number",
                    "",
                )
            ),

            "stroke_type": (
                stroke_type
            ),

            "trial_number": (
                row.get(
                    "trial_number",
                    "",
                )
            ),

            "pose_path": (
                str(
                    pose_path
                )
            ),
        }


def load_feature_sequences_for_normalizer(
    manifest_path,
):
    """
    Load raw extracted feature sequences from
    a manifest.

    Intended mainly for fitting the
    PoseFeatureNormalizer on the TRAINING
    split only.

    Returns:
        list of arrays shaped
        [frames, 44]
    """

    manifest_path = Path(
        manifest_path
    )

    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Manifest not found: "
            f"{manifest_path}"
        )

    feature_sequences = []

    with open(
        manifest_path,
        "r",
        encoding="utf-8",
        newline="",
    ) as file:
        reader = csv.DictReader(
            file
        )

        for row in reader:
            pose_path = row.get(
                "pose_path",
                "",
            )

            stroke_type = (
                row.get(
                    "stroke_type",
                    "",
                )
                .strip()
            )

            if (
                stroke_type
                not in CLASS_TO_INDEX
            ):
                continue

            if not pose_path:
                continue

            features = (
                load_pose_features(
                    pose_path
                )
            )

            feature_sequences.append(
                features
            )

    if not feature_sequences:
        raise RuntimeError(
            "No usable feature sequences "
            "were found."
        )

    return feature_sequences