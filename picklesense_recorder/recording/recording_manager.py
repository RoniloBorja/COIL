import threading
import time
import uuid

from models.imu_sample import ImuSample
from models.pose_sample import PoseSample

from recording.session_manager import SessionManager

from storage.imu_csv_writer import ImuCsvWriter
from storage.pose_csv_writer import PoseCsvWriter


class RecordingManager:
    def __init__(self):
        self.recording = False

        self.session_id = None

        self.record_start_ns = None
        self.record_stop_ns = None

        self.session_manager = None

        self.imu_writer = None
        self.pose_writer = None

        self.imu_samples_recorded = 0
        self.pose_frames_recorded = 0

        self.metadata = {}

        self._lock = threading.Lock()

    def start_recording(
        self,
        metadata: dict,
    ):
        with self._lock:
            if self.recording:
                raise RuntimeError(
                    "Recording is already active"
                )

            movement_name = (
                metadata.get(
                    "movement_name",
                    "",
                ).strip()
            )

            trial_name = (
                metadata.get(
                    "trial_name",
                    "",
                ).strip()
            )

            if not movement_name:
                raise ValueError(
                    "Movement name cannot be empty"
                )

            if not trial_name:
                raise ValueError(
                    "Trial name cannot be empty"
                )

            self.metadata = dict(
                metadata
            )

            self.session_id = str(
                uuid.uuid4()
            )

            self.session_manager = (
                SessionManager(
                    movement_name=movement_name,
                    trial_name=trial_name,
                )
            )

            self.session_manager.create_session_folder()

            self.record_start_ns = (
                time.monotonic_ns()
            )

            self.record_stop_ns = None

            self.imu_samples_recorded = 0
            self.pose_frames_recorded = 0

            self.imu_writer = ImuCsvWriter(
                file_path=(
                    self.session_manager.imu_csv_path
                ),
                record_start_ns=(
                    self.record_start_ns
                ),
            )

            self.pose_writer = PoseCsvWriter(
                file_path=(
                    self.session_manager.pose_csv_path
                ),
                record_start_ns=(
                    self.record_start_ns
                ),
            )

            try:
                self.imu_writer.open()
                self.pose_writer.open()

            except Exception:
                if self.imu_writer is not None:
                    self.imu_writer.close()

                if self.pose_writer is not None:
                    self.pose_writer.close()

                self.imu_writer = None
                self.pose_writer = None

                raise

            self.recording = True

            self._write_session_info(
                record_stop_ns=None
            )

            return {
                "session_id": self.session_id,
                "record_start_ns": (
                    self.record_start_ns
                ),
                "session_folder": (
                    self.session_manager.session_folder
                ),
            }

    def _write_session_info(
        self,
        record_stop_ns,
    ):
        self.session_manager.write_session_info(
            session_id=self.session_id,

            participant_id=self.metadata.get(
                "participant_id",
                "",
            ),

            session_number=self.metadata.get(
                "session_number"
            ),

            stroke_type=self.metadata.get(
                "stroke_type",
                "",
            ),

            trial_number=self.metadata.get(
                "trial_number"
            ),

            handedness=self.metadata.get(
                "handedness",
                "",
            ),

            notes=self.metadata.get(
                "notes",
                "",
            ),

            record_start_ns=(
                self.record_start_ns
            ),

            record_stop_ns=(
                record_stop_ns
            ),

            imu_samples=(
                self.imu_samples_recorded
            ),

            pose_frames=(
                self.pose_frames_recorded
            ),
        )

    def handle_imu_sample(
        self,
        sample: ImuSample,
    ):
        with self._lock:
            if not self.recording:
                return

            if self.imu_writer is None:
                return

            self.imu_writer.write_sample(
                sample
            )

            self.imu_samples_recorded += 1

    def handle_pose_sample(
        self,
        sample: PoseSample,
    ):
        with self._lock:
            if not self.recording:
                return

            if self.pose_writer is None:
                return

            self.pose_writer.write_sample(
                sample
            )

            self.pose_frames_recorded += 1

    def stop_recording(self):
        with self._lock:
            if not self.recording:
                return None

            self.record_stop_ns = (
                time.monotonic_ns()
            )

            self.recording = False

            if self.imu_writer is not None:
                self.imu_writer.close()

            if self.pose_writer is not None:
                self.pose_writer.close()

            self._write_session_info(
                record_stop_ns=(
                    self.record_stop_ns
                )
            )

            result = {
                "session_id": (
                    self.session_id
                ),

                "record_start_ns": (
                    self.record_start_ns
                ),

                "record_stop_ns": (
                    self.record_stop_ns
                ),

                "imu_samples": (
                    self.imu_samples_recorded
                ),

                "pose_frames": (
                    self.pose_frames_recorded
                ),

                "session_folder": (
                    self.session_manager.session_folder
                ),
            }

            self.imu_writer = None
            self.pose_writer = None

            return result

    def get_status(self):
        with self._lock:
            if (
                self.recording
                and self.record_start_ns
                is not None
            ):
                elapsed_seconds = (
                    time.monotonic_ns()
                    - self.record_start_ns
                ) / 1_000_000_000

            else:
                elapsed_seconds = 0.0

            session_folder = None

            if self.session_manager is not None:
                session_folder = (
                    self.session_manager.session_folder
                )

            return {
                "recording": self.recording,

                "session_id": (
                    self.session_id
                ),

                "record_start_ns": (
                    self.record_start_ns
                ),

                "record_stop_ns": (
                    self.record_stop_ns
                ),

                "elapsed_seconds": (
                    elapsed_seconds
                ),

                "imu_samples_recorded": (
                    self.imu_samples_recorded
                ),

                "pose_frames_recorded": (
                    self.pose_frames_recorded
                ),

                "session_folder": (
                    session_folder
                ),
            }