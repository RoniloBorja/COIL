from dataclasses import dataclass
@dataclass(slots=True)
class ImuSample:
    sequence: int
    esp_timestamp_us: int
    laptop_timestamp_ns: int

    ax_raw: int
    ay_raw: int
    az_raw: int

    gx_raw: int
    gy_raw: int
    gz_raw: int