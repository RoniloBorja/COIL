import json
from pathlib import Path

import numpy as np

from config.settings import (
    FEATURE_COUNT,
    OUTPUT_DIR,
)


NORMALIZER_FILENAME = (
    "pose_normalizer.json"
)


class PoseFeatureNormalizer:
    def __init__(self):
        self.mean = None
        self.std = None

    def fit(
        self,
        feature_sequences,
    ):
        """
        Fit normalization statistics using
        TRAINING DATA ONLY.

        feature_sequences:
            iterable of arrays shaped
            [frames, features]
        """

        all_frames = []

        for features in feature_sequences:
            features = np.asarray(
                features,
                dtype=np.float32,
            )

            if features.ndim != 2:
                raise ValueError(
                    "Expected feature array "
                    "with shape [frames, features]."
                )

            if (
                features.shape[1]
                != FEATURE_COUNT
            ):
                raise ValueError(
                    "Unexpected feature count: "
                    f"{features.shape[1]} "
                    f"!= {FEATURE_COUNT}"
                )

            all_frames.append(
                features
            )

        if not all_frames:
            raise ValueError(
                "No feature sequences "
                "provided to normalizer."
            )

        combined = np.concatenate(
            all_frames,
            axis=0,
        )

        self.mean = np.mean(
            combined,
            axis=0,
        ).astype(
            np.float32
        )

        self.std = np.std(
            combined,
            axis=0,
        ).astype(
            np.float32
        )

        # Prevent division by zero for
        # features with almost no variation.
        self.std = np.where(
            self.std < 1e-6,
            1.0,
            self.std,
        ).astype(
            np.float32
        )

        return self

    def transform(
        self,
        features,
    ):
        self._check_fitted()

        features = np.asarray(
            features,
            dtype=np.float32,
        )

        if features.ndim != 2:
            raise ValueError(
                "Expected feature array "
                "with shape [frames, features]."
            )

        if (
            features.shape[1]
            != FEATURE_COUNT
        ):
            raise ValueError(
                "Unexpected feature count: "
                f"{features.shape[1]} "
                f"!= {FEATURE_COUNT}"
            )

        normalized = (
            features
            - self.mean
        ) / self.std

        return normalized.astype(
            np.float32
        )

    def fit_transform(
        self,
        feature_sequences,
    ):
        feature_sequences = list(
            feature_sequences
        )

        self.fit(
            feature_sequences
        )

        return [
            self.transform(
                features
            )
            for features
            in feature_sequences
        ]

    def inverse_transform(
        self,
        features,
    ):
        self._check_fitted()

        features = np.asarray(
            features,
            dtype=np.float32,
        )

        restored = (
            features
            * self.std
        ) + self.mean

        return restored.astype(
            np.float32
        )

    def save(
        self,
        path=None,
    ):
        self._check_fitted()

        if path is None:
            path = (
                OUTPUT_DIR
                / NORMALIZER_FILENAME
            )

        path = Path(path)

        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        data = {
            "feature_count": (
                FEATURE_COUNT
            ),
            "mean": (
                self.mean.tolist()
            ),
            "std": (
                self.std.tolist()
            ),
        }

        with open(
            path,
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(
                data,
                file,
                indent=4,
            )

        return path

    def load(
        self,
        path=None,
    ):
        if path is None:
            path = (
                OUTPUT_DIR
                / NORMALIZER_FILENAME
            )

        path = Path(path)

        if not path.exists():
            raise FileNotFoundError(
                f"Normalizer file not found: "
                f"{path}"
            )

        with open(
            path,
            "r",
            encoding="utf-8",
        ) as file:
            data = json.load(
                file
            )

        feature_count = data.get(
            "feature_count"
        )

        if (
            feature_count
            != FEATURE_COUNT
        ):
            raise ValueError(
                "Normalizer feature count "
                "does not match current settings."
            )

        self.mean = np.asarray(
            data["mean"],
            dtype=np.float32,
        )

        self.std = np.asarray(
            data["std"],
            dtype=np.float32,
        )

        self._check_fitted()

        return self

    def _check_fitted(self):
        if (
            self.mean is None
            or self.std is None
        ):
            raise RuntimeError(
                "Normalizer has not been fitted."
            )

        if (
            len(self.mean)
            != FEATURE_COUNT
        ):
            raise RuntimeError(
                "Normalizer mean has "
                "incorrect size."
            )

        if (
            len(self.std)
            != FEATURE_COUNT
        ):
            raise RuntimeError(
                "Normalizer std has "
                "incorrect size."
            )