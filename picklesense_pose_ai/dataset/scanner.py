import csv
import json
from pathlib import Path

from config.settings import (
    RAW_DATA_DIR,
    MANIFEST_DIR,
    DATASET_MANIFEST_FILENAME,
    POSE_FILENAME,
    SESSION_INFO_FILENAME,
)


def scan_dataset():
    rows = []

    if not RAW_DATA_DIR.exists():
        raise FileNotFoundError(
            f"Raw data directory not found: {RAW_DATA_DIR}"
        )

    for session_info_path in RAW_DATA_DIR.rglob(
        SESSION_INFO_FILENAME
    ):
        trial_folder = session_info_path.parent

        pose_path = trial_folder / POSE_FILENAME

        if not pose_path.exists():
            continue

        try:
            with open(
                session_info_path,
                "r",
                encoding="utf-8",
            ) as file:
                metadata = json.load(file)

        except Exception as error:
            print(
                f"Skipping invalid metadata file:\n"
                f"{session_info_path}\n"
                f"Reason: {error}\n"
            )
            continue

        participant_id = metadata.get(
            "participant_id",
            "",
        )

        session_number = metadata.get(
            "session_number"
        )

        stroke_type = metadata.get(
            "stroke_type",
            "",
        )

        trial_number = metadata.get(
            "trial_number"
        )

        handedness = metadata.get(
            "handedness",
            "",
        )

        movement_name = metadata.get(
            "movement_name",
            "",
        )

        trial_name = metadata.get(
            "trial_name",
            "",
        )

        pose_frames = metadata.get(
            "pose_frames",
            0,
        )

        duration_seconds = metadata.get(
            "duration_seconds"
        )

        if not participant_id:
            print(
                f"Skipping trial with missing participant_id:\n"
                f"{trial_folder}\n"
            )
            continue

        if not stroke_type:
            print(
                f"Skipping trial with missing stroke_type:\n"
                f"{trial_folder}\n"
            )
            continue

        rows.append(
            {
                "participant_id": participant_id,
                "session_number": session_number,
                "stroke_type": stroke_type,
                "trial_number": trial_number,
                "handedness": handedness,
                "movement_name": movement_name,
                "trial_name": trial_name,
                "pose_frames": pose_frames,
                "duration_seconds": duration_seconds,
                "pose_path": str(
                    pose_path.resolve()
                ),
                "metadata_path": str(
                    session_info_path.resolve()
                ),
            }
        )

    rows.sort(
        key=lambda row: (
            str(row["participant_id"]),
            str(row["session_number"]),
            str(row["stroke_type"]),
            int(row["trial_number"])
            if row["trial_number"] is not None
            else 0,
        )
    )

    return rows


def save_manifest(rows):
    MANIFEST_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    manifest_path = (
        MANIFEST_DIR
        / DATASET_MANIFEST_FILENAME
    )

    fieldnames = [
        "participant_id",
        "session_number",
        "stroke_type",
        "trial_number",
        "handedness",
        "movement_name",
        "trial_name",
        "pose_frames",
        "duration_seconds",
        "pose_path",
        "metadata_path",
    ]

    with open(
        manifest_path,
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(rows)

    return manifest_path


def print_summary(rows):
    print()
    print("PickleSense Dataset Scan")
    print("------------------------")

    print(
        f"Total trials: {len(rows)}"
    )

    participants = sorted(
        {
            row["participant_id"]
            for row in rows
        }
    )

    print(
        f"Participants: {len(participants)}"
    )

    if participants:
        print(
            "Participant IDs:",
            ", ".join(participants),
        )

    stroke_counts = {}

    for row in rows:
        stroke = row["stroke_type"]

        stroke_counts[stroke] = (
            stroke_counts.get(
                stroke,
                0,
            )
            + 1
        )

    print()
    print("Trials per stroke:")

    for stroke in sorted(
        stroke_counts
    ):
        print(
            f"  {stroke}: "
            f"{stroke_counts[stroke]}"
        )

    print()


def main():
    rows = scan_dataset()

    print_summary(rows)

    manifest_path = save_manifest(
        rows
    )

    print(
        "Manifest saved to:"
    )
    print(
        manifest_path
    )


if __name__ == "__main__":
    main()