import json
from datetime import datetime

from config.settings import (
    RECORDINGS_ROOT,
    IMU_CSV_FILENAME,
    POSE_CSV_FILENAME,
    SESSION_INFO_FILENAME,
)


class SessionManager:
    def __init__(
        self,
        movement_name: str,
        trial_name: str,
    ):
        self.movement_name = (
            movement_name.strip()
        )

        self.trial_name = (
            trial_name.strip()
        )

        self.session_folder = (
            RECORDINGS_ROOT
            / self.movement_name
            / self.trial_name
        )

        self.imu_csv_path = (
            self.session_folder
            / IMU_CSV_FILENAME
        )

        self.pose_csv_path = (
            self.session_folder
            / POSE_CSV_FILENAME
        )

        self.session_info_path = (
            self.session_folder
            / SESSION_INFO_FILENAME
        )

    def create_session_folder(self):
        self.session_folder.mkdir(
            parents=True,
            exist_ok=True,
        )

    def write_session_info(
        self,
        session_id: str,
        record_start_ns: int,

        participant_id: str = "",
        session_number: int | None = None,
        stroke_type: str = "",
        trial_number: int | None = None,
        handedness: str = "",
        notes: str = "",

        record_stop_ns: int | None = None,
        imu_samples: int = 0,
        pose_frames: int = 0,
    ):
        self.create_session_folder()

        duration_seconds = None

        if (
            record_stop_ns is not None
            and record_start_ns is not None
        ):
            duration_seconds = (
                record_stop_ns
                - record_start_ns
            ) / 1_000_000_000

        session_info = {
            "session_id": session_id,

            "participant_id": (
                participant_id
            ),

            "session_number": (
                session_number
            ),

            "stroke_type": (
                stroke_type
            ),

            "trial_number": (
                trial_number
            ),

            "handedness": (
                handedness
            ),

            "movement_name": (
                self.movement_name
            ),

            "trial_name": (
                self.trial_name
            ),

            "record_start_ns": (
                record_start_ns
            ),

            "record_stop_ns": (
                record_stop_ns
            ),

            "duration_seconds": (
                duration_seconds
            ),

            "imu_samples": (
                imu_samples
            ),

            "pose_frames": (
                pose_frames
            ),

            "notes": (
                notes
            ),

            "imu_file": (
                IMU_CSV_FILENAME
            ),

            "pose_file": (
                POSE_CSV_FILENAME
            ),

            "created_at": (
                datetime.now().isoformat()
            ),
        }

        with open(
            self.session_info_path,
            mode="w",
            encoding="utf-8",
        ) as file:
            json.dump(
                session_info,
                file,
                indent=4,
            )

    def get_paths(self):
        return {
            "session_folder": (
                self.session_folder
            ),

            "imu_csv_path": (
                self.imu_csv_path
            ),

            "pose_csv_path": (
                self.pose_csv_path
            ),

            "session_info_path": (
                self.session_info_path
            ),
        }