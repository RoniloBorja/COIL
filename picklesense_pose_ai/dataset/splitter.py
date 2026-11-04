import csv
import random
from collections import Counter
from pathlib import Path

from config.settings import (
    MANIFEST_DIR,
    TRAIN_MANIFEST_FILENAME,
    VALIDATION_MANIFEST_FILENAME,
    TEST_MANIFEST_FILENAME,
    TRAIN_RATIO,
    VALIDATION_RATIO,
    TEST_RATIO,
    RANDOM_SEED,
)


VALIDATED_MANIFEST_FILENAME = (
    "validated_manifest.csv"
)


def load_valid_trials():
    manifest_path = (
        MANIFEST_DIR
        / VALIDATED_MANIFEST_FILENAME
    )

    if not manifest_path.exists():
        raise FileNotFoundError(
            "Validated manifest not found:\n"
            f"{manifest_path}\n\n"
            "Run dataset.validator first."
        )

    valid_rows = []

    with open(
        manifest_path,
        "r",
        encoding="utf-8",
        newline="",
    ) as file:
        reader = csv.DictReader(file)

        for row in reader:
            if (
                row.get(
                    "validation_status",
                    "",
                ).upper()
                == "VALID"
            ):
                valid_rows.append(row)

    if not valid_rows:
        raise RuntimeError(
            "No valid trials were found."
        )

    return valid_rows


def get_participants(rows):
    participants = sorted(
        {
            row["participant_id"]
            for row in rows
        }
    )

    return participants


def split_participants(
    participants,
):
    participant_count = len(
        participants
    )

    if participant_count < 3:
        raise RuntimeError(
            "At least 3 participants are "
            "required for participant-separated "
            "train/validation/test splitting.\n\n"
            f"Participants found: "
            f"{participant_count}"
        )

    shuffled = list(
        participants
    )

    rng = random.Random(
        RANDOM_SEED
    )

    rng.shuffle(
        shuffled
    )

    # -----------------------------------------
    # Calculate participant counts
    # -----------------------------------------

    test_count = max(
        1,
        round(
            participant_count
            * TEST_RATIO
        ),
    )

    validation_count = max(
        1,
        round(
            participant_count
            * VALIDATION_RATIO
        ),
    )

    # Ensure at least one participant
    # remains for training.

    while (
        test_count
        + validation_count
        >= participant_count
    ):
        if test_count > 1:
            test_count -= 1

        elif validation_count > 1:
            validation_count -= 1

        else:
            break

    train_count = (
        participant_count
        - validation_count
        - test_count
    )

    if train_count < 1:
        raise RuntimeError(
            "Could not create a valid "
            "participant-separated split."
        )

    # -----------------------------------------
    # Assign participants
    # -----------------------------------------

    test_participants = set(
        shuffled[
            :test_count
        ]
    )

    validation_participants = set(
        shuffled[
            test_count:
            test_count
            + validation_count
        ]
    )

    train_participants = set(
        shuffled[
            test_count
            + validation_count:
        ]
    )

    return (
        train_participants,
        validation_participants,
        test_participants,
    )


def assign_trials(
    rows,
    train_participants,
    validation_participants,
    test_participants,
):
    train_rows = []
    validation_rows = []
    test_rows = []

    for row in rows:
        participant = row[
            "participant_id"
        ]

        if participant in (
            train_participants
        ):
            train_rows.append(row)

        elif participant in (
            validation_participants
        ):
            validation_rows.append(
                row
            )

        elif participant in (
            test_participants
        ):
            test_rows.append(row)

    return (
        train_rows,
        validation_rows,
        test_rows,
    )


def save_manifest(
    rows,
    filename,
):
    output_path = (
        MANIFEST_DIR
        / filename
    )

    if not rows:
        raise RuntimeError(
            f"No rows available for "
            f"{filename}"
        )

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


def print_split_summary(
    split_name,
    rows,
):
    participants = sorted(
        {
            row["participant_id"]
            for row in rows
        }
    )

    stroke_counts = Counter(
        row["stroke_type"]
        for row in rows
    )

    print()
    print(
        f"{split_name}"
    )

    print(
        "-" * len(split_name)
    )

    print(
        f"Participants: "
        f"{len(participants)}"
    )

    print(
        "IDs:",
        ", ".join(
            participants
        ),
    )

    print(
        f"Trials: {len(rows)}"
    )

    print(
        "Stroke distribution:"
    )

    for stroke in sorted(
        stroke_counts
    ):
        print(
            f"  {stroke}: "
            f"{stroke_counts[stroke]}"
        )


def check_participant_overlap(
    train_rows,
    validation_rows,
    test_rows,
):
    train_ids = {
        row["participant_id"]
        for row in train_rows
    }

    validation_ids = {
        row["participant_id"]
        for row in validation_rows
    }

    test_ids = {
        row["participant_id"]
        for row in test_rows
    }

    if (
        train_ids
        & validation_ids
    ):
        raise RuntimeError(
            "Participant leakage detected "
            "between train and validation."
        )

    if (
        train_ids
        & test_ids
    ):
        raise RuntimeError(
            "Participant leakage detected "
            "between train and test."
        )

    if (
        validation_ids
        & test_ids
    ):
        raise RuntimeError(
            "Participant leakage detected "
            "between validation and test."
        )


def main():
    rows = load_valid_trials()

    participants = (
        get_participants(
            rows
        )
    )

    print()
    print(
        "PickleSense Participant Split"
    )

    print(
        "-----------------------------"
    )

    print(
        f"Valid trials: "
        f"{len(rows)}"
    )

    print(
        f"Participants: "
        f"{len(participants)}"
    )

    (
        train_participants,
        validation_participants,
        test_participants,
    ) = split_participants(
        participants
    )

    (
        train_rows,
        validation_rows,
        test_rows,
    ) = assign_trials(
        rows,
        train_participants,
        validation_participants,
        test_participants,
    )

    check_participant_overlap(
        train_rows,
        validation_rows,
        test_rows,
    )

    train_path = save_manifest(
        train_rows,
        TRAIN_MANIFEST_FILENAME,
    )

    validation_path = (
        save_manifest(
            validation_rows,
            VALIDATION_MANIFEST_FILENAME,
        )
    )

    test_path = save_manifest(
        test_rows,
        TEST_MANIFEST_FILENAME,
    )

    print_split_summary(
        "TRAIN",
        train_rows,
    )

    print_split_summary(
        "VALIDATION",
        validation_rows,
    )

    print_split_summary(
        "TEST",
        test_rows,
    )

    print()
    print(
        "Participant overlap check: PASS"
    )

    print()
    print(
        "Manifests saved:"
    )

    print(
        f"Train:      {train_path}"
    )

    print(
        f"Validation: {validation_path}"
    )

    print(
        f"Test:       {test_path}"
    )

    print()


if __name__ == "__main__":
    main()