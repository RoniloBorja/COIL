import math
from typing import Optional


def calculate_angle(
    point_a: tuple[float, float, float],
    point_b: tuple[float, float, float],
    point_c: tuple[float, float, float],
) -> Optional[float]:
    """
    Calculate the angle ABC in degrees.

    point_b is the joint vertex.
    """

    ba = (
        point_a[0] - point_b[0],
        point_a[1] - point_b[1],
        point_a[2] - point_b[2],
    )

    bc = (
        point_c[0] - point_b[0],
        point_c[1] - point_b[1],
        point_c[2] - point_b[2],
    )

    ba_length = math.sqrt(
        ba[0] ** 2
        + ba[1] ** 2
        + ba[2] ** 2
    )

    bc_length = math.sqrt(
        bc[0] ** 2
        + bc[1] ** 2
        + bc[2] ** 2
    )

    if (
        ba_length == 0
        or bc_length == 0
    ):
        return None

    dot_product = (
        ba[0] * bc[0]
        + ba[1] * bc[1]
        + ba[2] * bc[2]
    )

    cosine_angle = (
        dot_product
        / (
            ba_length
            * bc_length
        )
    )

    cosine_angle = max(
        -1.0,
        min(
            1.0,
            cosine_angle,
        ),
    )

    angle_radians = math.acos(
        cosine_angle
    )

    return math.degrees(
        angle_radians
    )


def _point(
    landmarks,
    index,
):
    landmark = landmarks[index]

    return (
        landmark[0],
        landmark[1],
        landmark[2],
    )


def calculate_joint_angles(
    world_landmarks,
):
    """
    Calculate major body joint angles from
    MediaPipe Pose world landmarks.
    """

    if (
        world_landmarks is None
        or len(world_landmarks) < 33
    ):
        return None

    angles = {}

    # --------------------------------------------------
    # Elbows
    # --------------------------------------------------

    angles["left_elbow_angle"] = calculate_angle(
        _point(world_landmarks, 11),
        _point(world_landmarks, 13),
        _point(world_landmarks, 15),
    )

    angles["right_elbow_angle"] = calculate_angle(
        _point(world_landmarks, 12),
        _point(world_landmarks, 14),
        _point(world_landmarks, 16),
    )

    # --------------------------------------------------
    # Shoulders
    # --------------------------------------------------

    angles["left_shoulder_angle"] = calculate_angle(
        _point(world_landmarks, 13),
        _point(world_landmarks, 11),
        _point(world_landmarks, 23),
    )

    angles["right_shoulder_angle"] = calculate_angle(
        _point(world_landmarks, 14),
        _point(world_landmarks, 12),
        _point(world_landmarks, 24),
    )

    # --------------------------------------------------
    # Hips
    # --------------------------------------------------

    angles["left_hip_angle"] = calculate_angle(
        _point(world_landmarks, 11),
        _point(world_landmarks, 23),
        _point(world_landmarks, 25),
    )

    angles["right_hip_angle"] = calculate_angle(
        _point(world_landmarks, 12),
        _point(world_landmarks, 24),
        _point(world_landmarks, 26),
    )

    # --------------------------------------------------
    # Knees
    # --------------------------------------------------

    angles["left_knee_angle"] = calculate_angle(
        _point(world_landmarks, 23),
        _point(world_landmarks, 25),
        _point(world_landmarks, 27),
    )

    angles["right_knee_angle"] = calculate_angle(
        _point(world_landmarks, 24),
        _point(world_landmarks, 26),
        _point(world_landmarks, 28),
    )

    return angles