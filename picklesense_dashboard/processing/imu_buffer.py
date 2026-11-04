from collections import deque
import threading

from config.settings import (
    IMU_WINDOW_SIZE,
)

from models.imu_sample import (
    ImuSample,
)


class ImuBuffer:
    def __init__(
        self,
        max_size=IMU_WINDOW_SIZE,
    ):
        self.max_size = int(
            max_size
        )

        self._samples = deque(
            maxlen=self.max_size
        )

        self._lock = (
            threading.Lock()
        )

    def add_sample(
        self,
        sample: ImuSample,
    ):
        with self._lock:
            self._samples.append(
                sample
            )

    def clear(
        self,
    ):
        with self._lock:
            self._samples.clear()

    def __len__(
        self,
    ):
        with self._lock:
            return len(
                self._samples
            )

    def is_full(
        self,
    ):
        with self._lock:
            return (
                len(
                    self._samples
                )
                >= self.max_size
            )

    def get_samples(
        self,
    ):
        with self._lock:
            return list(
                self._samples
            )

    def get_latest(
        self,
    ):
        with self._lock:
            if not self._samples:
                return None

            return (
                self._samples[-1]
            )

    def get_window(
        self,
        size=None,
    ):
        """
        Return the most recent IMU samples.

        If size is None, return the entire
        current buffer.
        """

        with self._lock:
            if not self._samples:
                return []

            samples = list(
                self._samples
            )

        if size is None:
            return samples

        size = int(
            size
        )

        if size <= 0:
            return []

        return samples[
            -size:
        ]

    def get_status(
        self,
    ):
        with self._lock:
            sample_count = len(
                self._samples
            )

            latest_sequence = None
            latest_timestamp_us = None

            if self._samples:
                latest = (
                    self._samples[-1]
                )

                latest_sequence = (
                    latest.sequence
                )

                latest_timestamp_us = (
                    latest.timestamp_us
                )

            return {
                "sample_count": (
                    sample_count
                ),

                "capacity": (
                    self.max_size
                ),

                "full": (
                    sample_count
                    >= self.max_size
                ),

                "latest_sequence": (
                    latest_sequence
                ),

                "latest_timestamp_us": (
                    latest_timestamp_us
                ),
            }