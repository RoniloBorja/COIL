import numpy as np

from config.settings import (
    FEATURE_COUNT,
)


def add_gaussian_noise(
    features,
    std=0.01,
    rng=None,
):
    """
    Add small random noise to pose features.

    Intended for TRAINING DATA ONLY.

    Input shape:
        [frames, features]

    Output shape:
        [frames, features]
    """

    features = np.asarray(
        features,
        dtype=np.float32,
    )

    if features.ndim != 2:
        raise ValueError(
            "Expected shape [frames, features]."
        )

    if features.shape[1] != FEATURE_COUNT:
        raise ValueError(
            f"Unexpected feature count: "
            f"{features.shape[1]} "
            f"!= {FEATURE_COUNT}"
        )

    if rng is None:
        rng = np.random.default_rng()

    noise = rng.normal(
        loc=0.0,
        scale=std,
        size=features.shape,
    ).astype(
        np.float32
    )

    return (
        features
        + noise
    ).astype(
        np.float32
    )


def temporal_scaling(
    features,
    scale_range=(0.90, 1.10),
    rng=None,
):
    """
    Slightly speed up or slow down the motion.

    Example:
        scale < 1.0
            shorter / faster sequence

        scale > 1.0
            longer / slower sequence

    The resampler later converts the result
    back to the fixed model sequence length.
    """

    features = np.asarray(
        features,
        dtype=np.float32,
    )

    if features.ndim != 2:
        raise ValueError(
            "Expected shape [frames, features]."
        )

    frame_count = features.shape[0]

    if frame_count < 2:
        return features.copy()

    if rng is None:
        rng = np.random.default_rng()

    scale = rng.uniform(
        scale_range[0],
        scale_range[1],
    )

    new_length = max(
        2,
        int(
            round(
                frame_count
                * scale
            )
        ),
    )

    original_time = np.linspace(
        0.0,
        1.0,
        frame_count,
        dtype=np.float32,
    )

    new_time = np.linspace(
        0.0,
        1.0,
        new_length,
        dtype=np.float32,
    )

    scaled = np.empty(
        (
            new_length,
            features.shape[1],
        ),
        dtype=np.float32,
    )

    for feature_index in range(
        features.shape[1]
    ):
        scaled[
            :,
            feature_index,
        ] = np.interp(
            new_time,
            original_time,
            features[
                :,
                feature_index,
            ],
        )

    return scaled


def temporal_crop(
    features,
    max_crop_fraction=0.05,
    rng=None,
):
    """
    Randomly remove a very small number
    of frames from the beginning/end.

    This makes the model less sensitive
    to slightly inconsistent START/STOP
    timing during recording.

    TRAINING DATA ONLY.
    """

    features = np.asarray(
        features,
        dtype=np.float32,
    )

    frame_count = features.shape[0]

    if frame_count < 10:
        return features.copy()

    if rng is None:
        rng = np.random.default_rng()

    max_crop = int(
        frame_count
        * max_crop_fraction
    )

    if max_crop < 1:
        return features.copy()

    crop_start = int(
        rng.integers(
            0,
            max_crop + 1,
        )
    )

    crop_end = int(
        rng.integers(
            0,
            max_crop + 1,
        )
    )

    end_index = (
        frame_count
        - crop_end
    )

    if (
        end_index
        - crop_start
        < 2
    ):
        return features.copy()

    return features[
        crop_start:end_index
    ].copy()


def augment_pose_sequence(
    features,
    probability=0.5,
    rng=None,
):
    """
    Apply lightweight pose augmentation.

    This function should only be called
    for the training split.

    Validation/test/live inference must
    never use augmentation.
    """

    features = np.asarray(
        features,
        dtype=np.float32,
    )

    if rng is None:
        rng = np.random.default_rng()

    augmented = (
        features.copy()
    )

    # -----------------------------------------
    # Small timing variation
    # -----------------------------------------

    if rng.random() < probability:
        augmented = temporal_scaling(
            augmented,
            scale_range=(
                0.90,
                1.10,
            ),
            rng=rng,
        )

    # -----------------------------------------
    # Slight START / STOP variation
    # -----------------------------------------

    if rng.random() < probability:
        augmented = temporal_crop(
            augmented,
            max_crop_fraction=0.05,
            rng=rng,
        )

    # -----------------------------------------
    # Small feature noise
    # -----------------------------------------

    if rng.random() < probability:
        augmented = add_gaussian_noise(
            augmented,
            std=0.01,
            rng=rng,
        )

    return augmented.astype(
        np.float32
    )