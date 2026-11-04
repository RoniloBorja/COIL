from dataset.validator import (
    validate_dataset,
    save_validated_manifest,
)

from dataset.splitter import (
    get_participants,
    split_participants,
    assign_trials,
    check_participant_overlap,
    save_manifest,
    print_split_summary,
)

from config.settings import (
    TRAIN_MANIFEST_FILENAME,
    VALIDATION_MANIFEST_FILENAME,
    TEST_MANIFEST_FILENAME,
)


def main():
    print()
    print("Preparing PickleSense dataset...")
    print()

    # --------------------------------------------------
    # Step 1: Validate all trials
    # --------------------------------------------------

    (
        validated_rows,
        valid_count,
        invalid_count,
    ) = validate_dataset()

    validated_manifest_path = (
        save_validated_manifest(
            validated_rows
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
        f"Validated manifest:"
    )

    print(
        validated_manifest_path
    )

    # --------------------------------------------------
    # Step 2: Keep only valid trials
    # --------------------------------------------------

    valid_rows = [
        row
        for row in validated_rows
        if (
            row.get(
                "validation_status",
                "",
            ).upper()
            == "VALID"
        )
    ]

    if not valid_rows:
        raise RuntimeError(
            "No valid trials remain "
            "after validation."
        )

    # --------------------------------------------------
    # Step 3: Get participants
    # --------------------------------------------------

    participants = (
        get_participants(
            valid_rows
        )
    )

    print()
    print(
        f"Participants found: "
        f"{len(participants)}"
    )

    print(
        ", ".join(
            participants
        )
    )

    # --------------------------------------------------
    # Step 4: Participant-separated split
    # --------------------------------------------------

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
        valid_rows,
        train_participants,
        validation_participants,
        test_participants,
    )

    # --------------------------------------------------
    # Step 5: Leakage check
    # --------------------------------------------------

    check_participant_overlap(
        train_rows,
        validation_rows,
        test_rows,
    )

    # --------------------------------------------------
    # Step 6: Save split manifests
    # --------------------------------------------------

    train_path = save_manifest(
        train_rows,
        TRAIN_MANIFEST_FILENAME,
    )

    validation_path = save_manifest(
        validation_rows,
        VALIDATION_MANIFEST_FILENAME,
    )

    test_path = save_manifest(
        test_rows,
        TEST_MANIFEST_FILENAME,
    )

    # --------------------------------------------------
    # Step 7: Print summaries
    # --------------------------------------------------

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
        "Dataset preparation complete."
    )

    print()
    print(
        "Saved manifests:"
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


if __name__ == "__main__":
    main()