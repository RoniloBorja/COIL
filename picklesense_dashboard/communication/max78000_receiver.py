import threading
import time

import serial

from config.settings import (
    MAX78000_SERIAL_PORT,
    MAX78000_BAUD_RATE,
    MAX78000_SERIAL_TIMEOUT_SECONDS,
)

from communication.packet_parser import (
    parse_packet,
    PacketParseError,
)

from processing.imu_buffer import (
    ImuBuffer,
)


class Max78000Receiver:
    def __init__(
        self,
        imu_buffer: ImuBuffer,
        on_prediction=None,
    ):
        self.imu_buffer = (
            imu_buffer
        )

        self.on_prediction = (
            on_prediction
        )

        self.serial_connection = None

        self.running = False
        self.thread = None

        self.total_packets = 0
        self.malformed_packets = 0

        self.last_packet_time_ns = None

        self.latest_prediction = None

        self._lock = (
            threading.Lock()
        )

    def start(
        self,
    ):
        if self.running:
            return

        self.serial_connection = (
            serial.Serial(
                port=(
                    MAX78000_SERIAL_PORT
                ),
                baudrate=(
                    MAX78000_BAUD_RATE
                ),
                timeout=(
                    MAX78000_SERIAL_TIMEOUT_SECONDS
                ),
            )
        )

        self.running = True

        self.thread = (
            threading.Thread(
                target=self._receive_loop,
                daemon=True,
            )
        )

        self.thread.start()

    def stop(
        self,
    ):
        self.running = False

        if (
            self.thread is not None
            and self.thread.is_alive()
        ):
            self.thread.join(
                timeout=1.0
            )

        self.thread = None

        if (
            self.serial_connection
            is not None
        ):
            try:
                self.serial_connection.close()

            except Exception:
                pass

        self.serial_connection = None

    def _receive_loop(
        self,
    ):
        while self.running:
            try:
                raw_line = (
                    self.serial_connection
                    .readline()
                )

                if not raw_line:
                    continue

                try:
                    packet_text = (
                        raw_line
                        .decode(
                            "utf-8",
                            errors="ignore",
                        )
                        .strip()
                    )

                except Exception:
                    continue

                if not packet_text:
                    continue

                try:
                    parsed = (
                        parse_packet(
                            packet_text
                        )
                    )

                except PacketParseError:
                    with self._lock:
                        self.malformed_packets += 1

                    continue

                now_ns = (
                    time.monotonic_ns()
                )

                with self._lock:
                    self.total_packets += 1

                    self.last_packet_time_ns = (
                        now_ns
                    )

                    if (
                        parsed.prediction
                        is not None
                    ):
                        self.latest_prediction = (
                            parsed.prediction
                        )

                self.imu_buffer.add_sample(
                    parsed.imu_sample
                )

                if (
                    parsed.prediction
                    is not None
                    and self.on_prediction
                    is not None
                ):
                    try:
                        self.on_prediction(
                            parsed.prediction
                        )

                    except Exception:
                        pass

            except (
                serial.SerialException,
                OSError,
            ):
                self.running = False
                break

            except Exception:
                continue

    def get_latest_prediction(
        self,
    ):
        with self._lock:
            return (
                self.latest_prediction
            )

    def get_status(
        self,
    ):
        with self._lock:
            last_packet_age_seconds = None

            if (
                self.last_packet_time_ns
                is not None
            ):
                last_packet_age_seconds = (
                    time.monotonic_ns()
                    - self.last_packet_time_ns
                ) / 1_000_000_000

            connected = (
                self.running
                and self.serial_connection
                is not None
                and self.serial_connection.is_open
            )

            receiving = (
                connected
                and last_packet_age_seconds
                is not None
                and last_packet_age_seconds
                < 2.0
            )

            return {
                "connected": (
                    connected
                ),

                "receiving": (
                    receiving
                ),

                "total_packets": (
                    self.total_packets
                ),

                "malformed_packets": (
                    self.malformed_packets
                ),

                "last_packet_age_seconds": (
                    last_packet_age_seconds
                ),

                "latest_prediction": (
                    self.latest_prediction
                ),

                "port": (
                    MAX78000_SERIAL_PORT
                ),

                "baud_rate": (
                    MAX78000_BAUD_RATE
                ),
            }