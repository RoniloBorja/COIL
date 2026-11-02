import socket
import threading
import time
from collections import deque
from typing import Callable, Optional

from config.settings import (
    UDP_BIND_IP,
    UDP_PORT,
    UDP_BUFFER_SIZE,
    UDP_SOCKET_TIMEOUT_SECONDS,
)

from communication.imu_packet_parser import (
    ImuPacketParseError,
    parse_imu_packet,
)

from models.imu_sample import ImuSample


class UdpReceiver:
    def __init__(
        self,
        on_sample: Optional[Callable[[ImuSample], None]] = None,
    ):
        self.on_sample = on_sample

        self.socket = None
        self.thread = None

        self.running = False

        self.total_packets = 0
        self.malformed_packets = 0
        self.missing_packets = 0

        self.last_sequence = None
        self.last_packet_time_ns = None

        self.start_time_ns = None

        self.current_rate_hz = 0.0
        self.average_rate_hz = 0.0

        self._recent_packet_times = deque()

        self._lock = threading.Lock()

    def start(self):
        if self.running:
            return

        self.socket = socket.socket(
            socket.AF_INET,
            socket.SOCK_DGRAM,
        )

        self.socket.bind(
            (
                UDP_BIND_IP,
                UDP_PORT,
            )
        )

        self.socket.settimeout(
            UDP_SOCKET_TIMEOUT_SECONDS
        )

        self.running = True

        self.start_time_ns = time.monotonic_ns()

        self.thread = threading.Thread(
            target=self._receive_loop,
            daemon=True,
        )

        self.thread.start()

    def _receive_loop(self):
        while self.running:
            try:
                packet_bytes, _ = self.socket.recvfrom(
                    UDP_BUFFER_SIZE
                )

            except socket.timeout:
                continue

            except OSError:
                break

            laptop_timestamp_ns = time.monotonic_ns()

            try:
                sample = parse_imu_packet(
                    packet_bytes,
                    laptop_timestamp_ns,
                )

            except ImuPacketParseError:
                with self._lock:
                    self.malformed_packets += 1

                continue

            self._process_sample(sample)

    def _process_sample(
        self,
        sample: ImuSample,
    ):
        with self._lock:
            self.total_packets += 1

            self.last_packet_time_ns = (
                sample.laptop_timestamp_ns
            )

            if self.last_sequence is not None:
                expected_sequence = (
                    self.last_sequence + 1
                )

                if sample.sequence > expected_sequence:
                    self.missing_packets += (
                        sample.sequence
                        - expected_sequence
                    )

            self.last_sequence = sample.sequence

            self._update_rates(
                sample.laptop_timestamp_ns
            )

        if self.on_sample is not None:
            self.on_sample(sample)

    def _update_rates(
        self,
        now_ns: int,
    ):
        self._recent_packet_times.append(
            now_ns
        )

        one_second_ago_ns = (
            now_ns
            - 1_000_000_000
        )

        while (
            self._recent_packet_times
            and self._recent_packet_times[0]
            < one_second_ago_ns
        ):
            self._recent_packet_times.popleft()

        if len(self._recent_packet_times) >= 2:
            window_ns = (
                self._recent_packet_times[-1]
                - self._recent_packet_times[0]
            )

            if window_ns > 0:
                self.current_rate_hz = (
                    (len(self._recent_packet_times) - 1)
                    * 1_000_000_000
                    / window_ns
                )

        if self.start_time_ns is not None:
            elapsed_ns = (
                now_ns
                - self.start_time_ns
            )

            if elapsed_ns > 0:
                self.average_rate_hz = (
                    self.total_packets
                    * 1_000_000_000
                    / elapsed_ns
                )

    def get_status(self):
        with self._lock:
            now_ns = time.monotonic_ns()

            if self.last_packet_time_ns is None:
                last_packet_age_seconds = None
            else:
                last_packet_age_seconds = (
                    now_ns
                    - self.last_packet_time_ns
                ) / 1_000_000_000

            return {
                "running": self.running,
                "total_packets": self.total_packets,
                "missing_packets": self.missing_packets,
                "malformed_packets": self.malformed_packets,
                "current_rate_hz": self.current_rate_hz,
                "average_rate_hz": self.average_rate_hz,
                "last_sequence": self.last_sequence,
                "last_packet_age_seconds": (
                    last_packet_age_seconds
                ),
            }

    def stop(self):
        self.running = False

        if self.socket is not None:
            try:
                self.socket.close()
            except OSError:
                pass

            self.socket = None

        if (
            self.thread is not None
            and self.thread.is_alive()
        ):
            self.thread.join(timeout=1.0)

        self.thread = None