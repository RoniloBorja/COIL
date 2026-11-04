from dataset.scanner import (
    scan_dataset,
    save_manifest,
    print_summary,
)


def main():
    print()
    print("Building PickleSense dataset manifest...")
    print()

    rows = scan_dataset()

    if not rows:
        raise RuntimeError(
            "No usable trials were found in data/raw."
        )

    print_summary(rows)

    manifest_path = save_manifest(
        rows
    )

    print(
        "Dataset manifest created:"
    )

    print(
        manifest_path
    )


if __name__ == "__main__":
    main()