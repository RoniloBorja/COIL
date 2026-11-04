from dataclasses import dataclass

from config.settings import (
    ACCEL_LSB_PER_G,
    GYRO_LSB_PER_DPS,
    STROKE_CLASSES,
)

from models.imu_sample import (
    ImuSample,
)


class PacketParseError(ValueError):
    pass


@dataclass(slots=True)
class Max78000Prediction:
    class_index: int
    class_name: str
    confidence: float


@dataclass(slots=True)
class ParsedMax78000Packet:
    imu_sample: ImuSample
    prediction: Max78000Prediction | None


def _parse_int(
    value: str,
    field_name: str,
):
    try:
        return int(
            value
        )

    except ValueError as error:
        raise PacketParseError(
            f"Invalid integer for "
            f"{field_name}: {value}"
        ) from error


def _parse_float(
    value: str,
    field_name: str,
):
    try:
        return float(
            value
        )

    except ValueError as error:
        raise PacketParseError(
            f"Invalid float for "
            f"{field_name}: {value}"
        ) from error


def parse_packet(
    packet_text: str,
):
    """
    Expected packet formats:

    IMU only:
        sequence,timestamp_us,
        ax,ay,az,
        gx,gy,gz

    IMU + MAX78000 prediction:
        sequence,timestamp_us,
        ax,ay,az,
        gx,gy,gz,
        class_index,confidence

    Example:

        1524,98234120,
        1200,-340,16320,
        220,-850,410,
        1,0.893
    """

    if packet_text is None:
        raise PacketParseError(
            "Packet is None"
        )

    packet_text = (
        packet_text.strip()
    )

    if not packet_text:
        raise PacketParseError(
            "Packet is empty"
        )

    parts = [
        part.strip()
        for part
        in packet_text.split(",")
    ]

    if len(parts) not in (
        8,
        10,
    ):
        raise PacketParseError(
            "Expected 8 or 10 comma-separated "
            f"fields, received {len(parts)}"
        )

    sequence = _parse_int(
        parts[0],
        "sequence",
    )

    timestamp_us = _parse_int(
        parts[1],
        "timestamp_us",
    )

    ax_raw = _parse_int(
        parts[2],
        "ax_raw",
    )

    ay_raw = _parse_int(
        parts[3],
        "ay_raw",
    )

    az_raw = _parse_int(
        parts[4],
        "az_raw",
    )

    gx_raw = _parse_int(
        parts[5],
        "gx_raw",
    )

    gy_raw = _parse_int(
        parts[6],
        "gy_raw",
    )

    gz_raw = _parse_int(
        parts[7],
        "gz_raw",
    )

    # --------------------------------------------------
    # Convert raw MPU6500 values into physical units
    # --------------------------------------------------

    ax_g = (
        ax_raw
        / ACCEL_LSB_PER_G
    )

    ay_g = (
        ay_raw
        / ACCEL_LSB_PER_G
    )

    az_g = (
        az_raw
        / ACCEL_LSB_PER_G
    )

    gx_dps = (
        gx_raw
        / GYRO_LSB_PER_DPS
    )

    gy_dps = (
        gy_raw
        / GYRO_LSB_PER_DPS
    )

    gz_dps = (
        gz_raw
        / GYRO_LSB_PER_DPS
    )

    imu_sample = ImuSample(
        sequence=sequence,
        timestamp_us=timestamp_us,

        ax_raw=ax_raw,
        ay_raw=ay_raw,
        az_raw=az_raw,

        gx_raw=gx_raw,
        gy_raw=gy_raw,
        gz_raw=gz_raw,

        ax_g=ax_g,
        ay_g=ay_g,
        az_g=az_g,

        gx_dps=gx_dps,
        gy_dps=gy_dps,
        gz_dps=gz_dps,
    )

    prediction = None

    # --------------------------------------------------
    # Optional MAX78000 AI output
    # --------------------------------------------------

    if len(parts) == 10:
        class_index = _parse_int(
            parts[8],
            "class_index",
        )

        confidence = _parse_float(
            parts[9],
            "confidence",
        )

        if not (
            0
            <= class_index
            < len(STROKE_CLASSES)
        ):
            raise PacketParseError(
                "class_index out of range: "
                f"{class_index}"
            )

        if not (
            0.0
            <= confidence
            <= 1.0
        ):
            raise PacketParseError(
                "confidence must be between "
                "0.0 and 1.0"
            )

        prediction = (
            Max78000Prediction(
                class_index=class_index,
                class_name=(
                    STROKE_CLASSES[
                        class_index
                    ]
                ),
                confidence=confidence,
            )
        )

    return ParsedMax78000Packet(
        imu_sample=imu_sample,
        prediction=prediction,
    )