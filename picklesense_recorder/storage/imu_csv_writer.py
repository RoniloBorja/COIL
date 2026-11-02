import csv
from pathlib import Path

from models.imu_sample import ImuSample


class ImuCsvWriter:
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

        self.writer.writerow(
            [
                "sequence",
                "esp_timestamp_us",
                "laptop_timestamp_ns",
                "relative_time_ns",
                "ax_raw",
                "ay_raw",
                "az_raw",
                "gx_raw",
                "gy_raw",
                "gz_raw",
            ]
        )

        self.samples_written = 0

    def write_sample(
        self,
        sample: ImuSample,
    ):
        if self.writer is None:
            raise RuntimeError(
                "IMU CSV writer is not open"
            )

        relative_time_ns = (
            sample.laptop_timestamp_ns
            - self.record_start_ns
        )

        self.writer.writerow(
            [
                sample.sequence,
                sample.esp_timestamp_us,
                sample.laptop_timestamp_ns,
                relative_time_ns,
                sample.ax_raw,
                sample.ay_raw,
                sample.az_raw,
                sample.gx_raw,
                sample.gy_raw,
                sample.gz_raw,
            ]
        )

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