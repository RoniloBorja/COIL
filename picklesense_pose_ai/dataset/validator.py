import csv
from pathlib import Path

import pandas as pd

from config.settings import (
    MANIFEST_DIR,
    DATASET_MANIFEST_FILENAME,
    SELECTED_LANDMARKS,
    JOINT_ANGLE_FIELDS,
    MIN_VALID_POSE_FRAMES,
)


def get_required_columns():
    columns = [
        "frame_number",
        "laptop_timestamp_ns",
        "relative_time_ns",
        "pose_detected",
    ]

    for landmark_index in SELECTED_LANDMARKS:
        columns.extend(
            [
                f"world_{landmark_index}_x",
                f"world_{landmark_index}_y",
                f"world_{landmark_index}_z",
            ]
        )

    columns.extend(
        JOINT_ANGLE_FIELDS
    )

    return columns


def validate_pose_file(pose_path: Path):
    result = {
        "valid": True,
        "reason": "",
        "total_frames": 0,
        "valid_pose_frames": 0,
    }

    if not pose_path.exists():
        result["valid"] = False
        result["reason"] = "pose_data.csv missing"
        return result

    try:
        dataframe = pd.read_csv(
            pose_path
        )

    except Exception as error:
        result["valid"] = False
        result["reason"] = (
            f"Could not read CSV: {error}"
        )
        return result

    result["total_frames"] = len(
        dataframe
    )

    if dataframe.empty:
        result["valid"] = False
        result["reason"] = "Pose CSV is empty"
        return result

    required_columns = (
        get_required_columns()
    )

    missing_columns = [
        column
        for column in required_columns
        if column not in dataframe.columns
    ]

    if missing_columns:
        result["valid"] = False

        result["reason"] = (
            "Missing required columns: "
            + ", ".join(
                missing_columns
            )
        )

        return result

    pose_detected = (
        pd.to_numeric(
            dataframe["pose_detected"],
            errors="coerce",
        )
        .fillna(0)
        .astype(bool)
    )

    valid_dataframe = (
        dataframe.loc[
            pose_detected
        ].copy()
    )

    if valid_dataframe.empty:
        result["valid"] = False
        result["reason"] = (
            "No frames with detected pose"
        )
        return result

    feature_columns = []

    for landmark_index in (
        SELECTED_LANDMARKS
    ):
        feature_columns.extend(
            [
                f"world_{landmark_index}_x",
                f"world_{landmark_index}_y",
                f"world_{landmark_index}_z",
            ]
        )

    feature_columns.extend(
        JOINT_ANGLE_FIELDS
    )

    numeric_features = (
        valid_dataframe[
            feature_columns
        ].apply(
            pd.to_numeric,
            errors="coerce",
        )
    )

    complete_rows = (
        numeric_features
        .notna()
        .all(axis=1)
    )

    valid_pose_frames = int(
        complete_rows.sum()
    )

    result["valid_pose_frames"] = (
        valid_pose_frames
    )

    if (
        valid_pose_frames
        < MIN_VALID_POSE_FRAMES
    ):
        result["valid"] = False
        result["reason"] = (
            "Not enough complete pose frames: "
            f"{valid_pose_frames} "
            f"< {MIN_VALID_POSE_FRAMES}"
        )

        return result

    result["reason"] = "OK"

    return result


def load_manifest():
    manifest_path = (
        MANIFEST_DIR
        / DATASET_MANIFEST_FILENAME
    )

    if not manifest_path.exists():
        raise FileNotFoundError(
            "Dataset manifest not found:\n"
            f"{manifest_path}\n\n"
            "Run dataset.scanner first."
        )

    rows = []

    with open(
        manifest_path,
        "r",
        encoding="utf-8",
        newline="",
    ) as file:
        reader = csv.DictReader(
            file
        )

        rows.extend(
            reader
        )

    return rows


def validate_dataset():
    rows = load_manifest()

    validated_rows = []

    valid_count = 0
    invalid_count = 0

    print()
    print("PickleSense Dataset Validation")
    print("------------------------------")

    for index, row in enumerate(
        rows,
        start=1,
    ):
        pose_path = Path(
            row["pose_path"]
        )

        validation = (
            validate_pose_file(
                pose_path
            )
        )

        row["validation_status"] = (
            "VALID"
            if validation["valid"]
            else "INVALID"
        )

        row["validation_reason"] = (
            validation["reason"]
        )

        row["actual_pose_frames"] = (
            validation["total_frames"]
        )

        row[
            "valid_pose_frames"
        ] = validation[
            "valid_pose_frames"
        ]

        validated_rows.append(
            row
        )

        if validation["valid"]:
            valid_count += 1
            status_text = "VALID"

        else:
            invalid_count += 1
            status_text = "INVALID"

        print(
            f"[{index}/{len(rows)}] "
            f"{row.get('participant_id')} | "
            f"{row.get('stroke_type')} | "
            f"Trial {row.get('trial_number')} "
            f"→ {status_text}"
        )

        if not validation["valid"]:
            print(
                f"    Reason: "
                f"{validation['reason']}"
            )

    return (
        validated_rows,
        valid_count,
        invalid_count,
    )


def save_validated_manifest(
    rows,
):
    output_path = (
        MANIFEST_DIR
        / "validated_manifest.csv"
    )

    if not rows:
        return output_path

    fieldnames = list(
        rows[0].keys()
    )

    with open(
        output_path,
        "w",
        encoding="utf-8",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(
            rows
        )

    return output_path


def main():
    (
        rows,
        valid_count,
        invalid_count,
    ) = validate_dataset()

    output_path = (
        save_validated_manifest(
            rows
        )
    )

    print()
    print("Validation Summary")
    print("------------------")
    print(
        f"Valid trials:   {valid_count}"
    )
    print(
        f"Invalid trials: {invalid_count}"
    )
    print(
        f"Total trials:   "
        f"{valid_count + invalid_count}"
    )

    print()
    print(
        "Validated manifest saved to:"
    )
    print(
        output_path
    )
    print()


if __name__ == "__main__":
    main()