from models.imu_sample import ImuSample


EXPECTED_FIELD_COUNT = 8


class ImuPacketParseError(ValueError):
    pass


def parse_imu_packet(
    packet_bytes: bytes,
    laptop_timestamp_ns: int,
) -> ImuSample:
    """
    Parse one UDP packet from the ESP32.

    Expected packet format:

        sequence,
        esp_timestamp_us,
        ax,
        ay,
        az,
        gx,
        gy,
        gz
    """

    try:
        packet_text = packet_bytes.decode("utf-8").strip()

    except UnicodeDecodeError as exc:
        raise ImuPacketParseError(
            "Packet is not valid UTF-8"
        ) from exc


    fields = packet_text.split(",")

    if len(fields) != EXPECTED_FIELD_COUNT:
        raise ImuPacketParseError(
            f"Expected {EXPECTED_FIELD_COUNT} fields, "
            f"received {len(fields)}"
        )


    try:
        (
            sequence,
            esp_timestamp_us,

            ax_raw,
            ay_raw,
            az_raw,

            gx_raw,
            gy_raw,
            gz_raw,
        ) = map(int, fields)

    except ValueError as exc:
        raise ImuPacketParseError(
            "Packet contains a non-integer field"
        ) from exc


    return ImuSample(
        sequence=sequence,
        esp_timestamp_us=esp_timestamp_us,
        laptop_timestamp_ns=laptop_timestamp_ns,

        ax_raw=ax_raw,
        ay_raw=ay_raw,
        az_raw=az_raw,

        gx_raw=gx_raw,
        gy_raw=gy_raw,
        gz_raw=gz_raw,
    )