from dataclasses import dataclass


@dataclass(slots=True)
class ImuSample:
    sequence: int
    timestamp_us: int

    ax_raw: int
    ay_raw: int
    az_raw: int

    gx_raw: int
    gy_raw: int
    gz_raw: int

    ax_g: float
    ay_g: float
    az_g: float

    gx_dps: float
    gy_dps: float
    gz_dps: float