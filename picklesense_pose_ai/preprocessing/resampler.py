import numpy as np

from config.settings import (
    SEQUENCE_LENGTH,
    FEATURE_COUNT,
)


def resample_sequence(
    features,
    target_length=SEQUENCE_LENGTH,
):
    """
    Resample a variable-length pose feature
    sequence to a fixed number of time steps.

    Input:
        features:
            numpy array shaped
            [frames, features]

    Output:
        numpy array shaped
            [target_length, features]
    """

    features = np.asarray(
        features,
        dtype=np.float32,
    )

    if features.ndim != 2:
        raise ValueError(
            "Expected feature sequence "
            "with shape [frames, features]."
        )

    frame_count, feature_count = (
        features.shape
    )

    if (
        feature_count
        != FEATURE_COUNT
    ):
        raise ValueError(
            "Unexpected feature count: "
            f"{feature_count} "
            f"!= {FEATURE_COUNT}"
        )

    if frame_count < 2:
        raise ValueError(
            "At least 2 frames are required "
            "for temporal resampling."
        )

    if target_length < 2:
        raise ValueError(
            "Target length must be at least 2."
        )

    # --------------------------------------------------
    # If the sequence already has the target size,
    # return a copy.
    # --------------------------------------------------

    if frame_count == target_length:
        return features.copy()

    # --------------------------------------------------
    # Create normalized time coordinates.
    #
    # Original:
    # 0.0 -------------------------- 1.0
    #
    # Target:
    # same normalized time range,
    # but exactly target_length samples.
    # --------------------------------------------------

    original_time = np.linspace(
        0.0,
        1.0,
        num=frame_count,
        dtype=np.float32,
    )

    target_time = np.linspace(
        0.0,
        1.0,
        num=target_length,
        dtype=np.float32,
    )

    resampled = np.empty(
        (
            target_length,
            feature_count,
        ),
        dtype=np.float32,
    )

    # --------------------------------------------------
    # Interpolate every feature independently.
    # --------------------------------------------------

    for feature_index in range(
        feature_count
    ):
        resampled[
            :,
            feature_index,
        ] = np.interp(
            target_time,
            original_time,
            features[
                :,
                feature_index,
            ],
        )

    return resampled


def to_model_input(
    features,
):
    """
    Convert:

        [frames, features]

    into PyTorch Conv1D format:

        [features, frames]

    Example:

        [64, 44]
        becomes
        [44, 64]
    """

    features = np.asarray(
        features,
        dtype=np.float32,
    )

    expected_shape = (
        SEQUENCE_LENGTH,
        FEATURE_COUNT,
    )

    if (
        features.shape
        != expected_shape
    ):
        raise ValueError(
            "Expected resampled sequence "
            f"shape {expected_shape}, "
            f"got {features.shape}"
        )

    return np.transpose(
        features,
        (1, 0),
    ).astype(
        np.float32
    )


def resample_for_model(
    features,
):
    """
    Convenience function:

    variable length
        ↓
    [64, 44]
        ↓
    [44, 64]
    """

    resampled = (
        resample_sequence(
            features
        )
    )

    model_input = (
        to_model_input(
            resampled
        )
    )

    return model_input