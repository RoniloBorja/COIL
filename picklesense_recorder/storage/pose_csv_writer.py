import csv
from pathlib import Path

from models.pose_sample import PoseSample


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


class PoseCsvWriter:
    def __init__(
        self,
        file_path: Path,
        record_start_ns: int,
    ):
        self.file_path = Path(file_path)
        self.record_start_ns = record_start_ns

        self.file = None
        self.writer = None

        self.samples_written = 0

    def open(self):
        self.file_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.file = open(
            self.file_path,
            mode="w",
            newline="",
            encoding="utf-8",
        )

        self.writer = csv.writer(
            self.file
        )

        header = [
            "frame_number",
            "laptop_timestamp_ns",
            "relative_time_ns",
            "pose_detected",
        ]

        # --------------------------------------------------
        # Normalized image landmarks
        # --------------------------------------------------

        for landmark_index in range(33):
            header.extend(
                [
                    f"landmark_{landmark_index}_x",
                    f"landmark_{landmark_index}_y",
                    f"landmark_{landmark_index}_z",
                    f"landmark_{landmark_index}_visibility",
                ]
            )

        # --------------------------------------------------
        # World landmarks
        # --------------------------------------------------

        for landmark_index in range(33):
            header.extend(
                [
                    f"world_{landmark_index}_x",
                    f"world_{landmark_index}_y",
                    f"world_{landmark_index}_z",
                    f"world_{landmark_index}_visibility",
                ]
            )

        # --------------------------------------------------
        # Derived joint angles
        # --------------------------------------------------

        header.extend(
            JOINT_ANGLE_FIELDS
        )

        self.writer.writerow(header)

        self.samples_written = 0

    def write_sample(
        self,
        sample: PoseSample,
    ):
        if self.writer is None:
            raise RuntimeError(
                "Pose CSV writer is not open"
            )

        relative_time_ns = (
            sample.laptop_timestamp_ns
            - self.record_start_ns
        )

        row = [
            sample.frame_number,
            sample.laptop_timestamp_ns,
            relative_time_ns,
            int(sample.pose_detected),
        ]

        # --------------------------------------------------
        # Normalized image landmarks
        # --------------------------------------------------

        if (
            sample.pose_detected
            and sample.landmarks is not None
        ):
            for landmark in sample.landmarks:
                x, y, z, visibility = landmark

                row.extend(
                    [
                        x,
                        y,
                        z,
                        visibility,
                    ]
                )

        else:
            for _ in range(33):
                row.extend(
                    [
                        "",
                        "",
                        "",
                        "",
                    ]
                )

        # --------------------------------------------------
        # World landmarks
        # --------------------------------------------------

        if (
            sample.pose_detected
            and sample.world_landmarks is not None
        ):
            for landmark in sample.world_landmarks:
                x, y, z, visibility = landmark

                row.extend(
                    [
                        x,
                        y,
                        z,
                        visibility,
                    ]
                )

        else:
            for _ in range(33):
                row.extend(
                    [
                        "",
                        "",
                        "",
                        "",
                    ]
                )

        # --------------------------------------------------
        # Joint angles
        # --------------------------------------------------

        if sample.joint_angles is not None:
            for field_name in JOINT_ANGLE_FIELDS:
                value = sample.joint_angles.get(
                    field_name
                )

                if value is None:
                    row.append("")
                else:
                    row.append(value)

        else:
            for _ in JOINT_ANGLE_FIELDS:
                row.append("")

        self.writer.writerow(row)

        self.samples_written += 1

    def flush(self):
        if self.file is not None:
            self.file.flush()

    def close(self):
        if self.file is not None:
            self.file.flush()
            self.file.close()

        self.file = None
        self.writer = None