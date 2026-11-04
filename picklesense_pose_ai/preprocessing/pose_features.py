import numpy as np
import pandas as pd

from config.settings import (
    SELECTED_LANDMARKS,
    JOINT_ANGLE_FIELDS,
    LEFT_HIP_INDEX,
    RIGHT_HIP_INDEX,
    LEFT_SHOULDER_INDEX,
    RIGHT_SHOULDER_INDEX,
    FEATURE_COUNT,
)


def _get_world_xyz(
    dataframe: pd.DataFrame,
    landmark_index: int,
):
    columns = [
        f"world_{landmark_index}_x",
        f"world_{landmark_index}_y",
        f"world_{landmark_index}_z",
    ]

    values = (
        dataframe[columns]
        .apply(
            pd.to_numeric,
            errors="coerce",
        )
        .to_numpy(
            dtype=np.float32
        )
    )

    return values


def _get_pose_detected_mask(
    dataframe: pd.DataFrame,
):
    if "pose_detected" not in dataframe.columns:
        return np.ones(
            len(dataframe),
            dtype=bool,
        )

    values = pd.to_numeric(
        dataframe["pose_detected"],
        errors="coerce",
    )

    values = (
        values
        .fillna(0)
        .to_numpy()
    )

    return values.astype(bool)


def _calculate_pelvis_center(
    dataframe: pd.DataFrame,
):
    left_hip = _get_world_xyz(
        dataframe,
        LEFT_HIP_INDEX,
    )

    right_hip = _get_world_xyz(
        dataframe,
        RIGHT_HIP_INDEX,
    )

    pelvis_center = (
        left_hip
        + right_hip
    ) / 2.0

    return pelvis_center


def _calculate_body_scale(
    dataframe: pd.DataFrame,
):
    left_shoulder = _get_world_xyz(
        dataframe,
        LEFT_SHOULDER_INDEX,
    )

    right_shoulder = _get_world_xyz(
        dataframe,
        RIGHT_SHOULDER_INDEX,
    )

    shoulder_vector = (
        left_shoulder
        - right_shoulder
    )

    shoulder_width = np.linalg.norm(
        shoulder_vector,
        axis=1,
    )

    # Prevent division by zero or
    # extremely small scale values.
    shoulder_width = np.where(
        shoulder_width < 1e-6,
        1.0,
        shoulder_width,
    )

    return shoulder_width


def _normalize_angle(
    angle_degrees,
):
    # Convert approximately from:
    #
    # 0..180 degrees
    #
    # to:
    #
    # 0..1
    #
    return (
        angle_degrees
        / 180.0
    )


def extract_pose_features(
    dataframe: pd.DataFrame,
):
    """
    Convert one pose_data.csv dataframe into
    the PickleSense 44-feature representation.

    Output shape:

        [valid_frames, 44]

    Feature order:

        12 selected world landmarks × XYZ
        followed by
        8 normalized joint angles.
    """

    if dataframe.empty:
        raise ValueError(
            "Pose dataframe is empty."
        )

    pose_detected_mask = (
        _get_pose_detected_mask(
            dataframe
        )
    )

    dataframe = (
        dataframe.loc[
            pose_detected_mask
        ]
        .reset_index(
            drop=True
        )
    )

    if dataframe.empty:
        raise ValueError(
            "No detected pose frames found."
        )

    pelvis_center = (
        _calculate_pelvis_center(
            dataframe
        )
    )

    body_scale = (
        _calculate_body_scale(
            dataframe
        )
    )

    landmark_features = []

    for landmark_index in (
        SELECTED_LANDMARKS
    ):
        xyz = _get_world_xyz(
            dataframe,
            landmark_index,
        )

        # -------------------------------------
        # Translation normalization
        #
        # Center the skeleton on the pelvis.
        # -------------------------------------

        xyz = (
            xyz
            - pelvis_center
        )

        # -------------------------------------
        # Scale normalization
        #
        # Normalize using shoulder width.
        # -------------------------------------

        xyz = (
            xyz
            / body_scale[:, None]
        )

        landmark_features.append(
            xyz
        )

    landmark_features = np.concatenate(
        landmark_features,
        axis=1,
    )

    angle_features = (
        dataframe[
            JOINT_ANGLE_FIELDS
        ]
        .apply(
            pd.to_numeric,
            errors="coerce",
        )
        .to_numpy(
            dtype=np.float32
        )
    )

    angle_features = (
        _normalize_angle(
            angle_features
        )
    )

    features = np.concatenate(
        [
            landmark_features,
            angle_features,
        ],
        axis=1,
    )

    if features.shape[1] != FEATURE_COUNT:
        raise RuntimeError(
            "Unexpected feature dimension: "
            f"{features.shape[1]} "
            f"!= {FEATURE_COUNT}"
        )

    # -----------------------------------------
    # Remove frames containing NaN / Inf.
    # -----------------------------------------

    valid_rows = np.isfinite(
        features
    ).all(
        axis=1
    )

    features = features[
        valid_rows
    ]

    if len(features) == 0:
        raise ValueError(
            "No complete feature frames "
            "remain after preprocessing."
        )

    return features.astype(
        np.float32
    )


def load_pose_features(
    pose_path,
):
    dataframe = pd.read_csv(
        pose_path
    )

    return extract_pose_features(
        dataframe
    )


def get_feature_names():
    names = []

    for landmark_index in (
        SELECTED_LANDMARKS
    ):
        names.extend(
            [
                f"world_{landmark_index}_x",
                f"world_{landmark_index}_y",
                f"world_{landmark_index}_z",
            ]
        )

    names.extend(
        JOINT_ANGLE_FIELDS
    )

    return names