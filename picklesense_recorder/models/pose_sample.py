from dataclasses import dataclass
from typing import Optional


@dataclass(slots=True)
class PoseSample:
    frame_number: int
    laptop_timestamp_ns: int
    pose_detected: bool

    # Normalized image coordinates:
    # (x, y, z, visibility)
    landmarks: Optional[
        list[
            tuple[
                float,
                float,
                float,
                float,
            ]
        ]
    ]

    # World coordinates:
    # (x, y, z, visibility)
    world_landmarks: Optional[
        list[
            tuple[
                float,
                float,
                float,
                float,
            ]
        ]
    ]

    # Derived joint angles in degrees
    joint_angles: Optional[
        dict[str, float]
    ]