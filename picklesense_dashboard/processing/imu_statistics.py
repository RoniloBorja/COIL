import math
from typing import Iterable

from models.imu_sample import ImuSample


def _magnitude(
    x: float,
    y: float,
    z: float,
) -> float:
    return math.sqrt(
        (x * x)
        + (y * y)
        + (z * z)
    )


def _rms(
    values: list[float],
) -> float:
    if not values:
        return 0.0

    mean_square = sum(
        value * value
        for value in values
    ) / len(values)

    return math.sqrt(
        mean_square
    )


def calculate_imu_statistics(
    samples: Iterable[ImuSample],
):
    samples = list(
        samples
    )

    if not samples:
        return {
            "sample_count": 0,
            "sampling_rate_hz": 0.0,

            "peak_acceleration_g": 0.0,
            "mean_acceleration_g": 0.0,
            "acceleration_rms_g": 0.0,

            "peak_angular_velocity_dps": 0.0,
            "mean_angular_velocity_dps": 0.0,
            "gyroscope_rms_dps": 0.0,

            "max_ax_g": 0.0,
            "max_ay_g": 0.0,
            "max_az_g": 0.0,

            "max_gx_dps": 0.0,
            "max_gy_dps": 0.0,
            "max_gz_dps": 0.0,
        }

    acceleration_magnitudes = []
    gyro_magnitudes = []

    ax_values = []
    ay_values = []
    az_values = []

    gx_values = []
    gy_values = []
    gz_values = []

    for sample in samples:
        ax = sample.ax_g
        ay = sample.ay_g
        az = sample.az_g

        gx = sample.gx_dps
        gy = sample.gy_dps
        gz = sample.gz_dps

        ax_values.append(
            ax
        )

        ay_values.append(
            ay
        )

        az_values.append(
            az
        )

        gx_values.append(
            gx
        )

        gy_values.append(
            gy
        )

        gz_values.append(
            gz
        )

        acceleration_magnitudes.append(
            _magnitude(
                ax,
                ay,
                az,
            )
        )

        gyro_magnitudes.append(
            _magnitude(
                gx,
                gy,
                gz,
            )
        )

    # --------------------------------------------------
    # Sampling rate
    # --------------------------------------------------

    sampling_rate_hz = 0.0

    if len(samples) >= 2:
        first_timestamp = (
            samples[0].timestamp_us
        )

        last_timestamp = (
            samples[-1].timestamp_us
        )

        elapsed_us = (
            last_timestamp
            - first_timestamp
        )

        if elapsed_us > 0:
            sampling_rate_hz = (
                (len(samples) - 1)
                * 1_000_000.0
                / elapsed_us
            )

    # --------------------------------------------------
    # Acceleration
    # --------------------------------------------------

    peak_acceleration = max(
        acceleration_magnitudes
    )

    mean_acceleration = sum(
        acceleration_magnitudes
    ) / len(
        acceleration_magnitudes
    )

    acceleration_rms = _rms(
        acceleration_magnitudes
    )

    # --------------------------------------------------
    # Gyroscope
    # --------------------------------------------------

    peak_angular_velocity = max(
        gyro_magnitudes
    )

    mean_angular_velocity = sum(
        gyro_magnitudes
    ) / len(
        gyro_magnitudes
    )

    gyroscope_rms = _rms(
        gyro_magnitudes
    )

    # --------------------------------------------------
    # Per-axis maximum absolute values
    # --------------------------------------------------

    max_ax = max(
        abs(value)
        for value in ax_values
    )

    max_ay = max(
        abs(value)
        for value in ay_values
    )

    max_az = max(
        abs(value)
        for value in az_values
    )

    max_gx = max(
        abs(value)
        for value in gx_values
    )

    max_gy = max(
        abs(value)
        for value in gy_values
    )

    max_gz = max(
        abs(value)
        for value in gz_values
    )

    return {
        "sample_count": (
            len(samples)
        ),

        "sampling_rate_hz": (
            sampling_rate_hz
        ),

        "peak_acceleration_g": (
            peak_acceleration
        ),

        "mean_acceleration_g": (
            mean_acceleration
        ),

        "acceleration_rms_g": (
            acceleration_rms
        ),

        "peak_angular_velocity_dps": (
            peak_angular_velocity
        ),

        "mean_angular_velocity_dps": (
            mean_angular_velocity
        ),

        "gyroscope_rms_dps": (
            gyroscope_rms
        ),

        "max_ax_g": (
            max_ax
        ),

        "max_ay_g": (
            max_ay
        ),

        "max_az_g": (
            max_az
        ),

        "max_gx_dps": (
            max_gx
        ),

        "max_gy_dps": (
            max_gy
        ),

        "max_gz_dps": (
            max_gz
        ),
    }