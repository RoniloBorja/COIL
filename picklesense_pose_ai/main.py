import argparse

from scripts.build_manifest import main as build_manifest_main
from scripts.prepare_dataset import main as prepare_dataset_main
from scripts.train_model import main as train_model_main
from scripts.evaluate_model import main as evaluate_model_main


def main():
    parser = argparse.ArgumentParser(
        description=(
            "PickleSense Pose AI command launcher"
        )
    )

    parser.add_argument(
        "command",
        choices=[
            "build",
            "prepare",
            "train",
            "evaluate",
            "all",
        ],
        help=(
            "Pipeline command to run"
        ),
    )

    args = parser.parse_args()

    if args.command == "build":
        build_manifest_main()

    elif args.command == "prepare":
        prepare_dataset_main()

    elif args.command == "train":
        train_model_main()

    elif args.command == "evaluate":
        evaluate_model_main()

    elif args.command == "all":
        build_manifest_main()
        prepare_dataset_main()
        train_model_main()
        evaluate_model_main()


if __name__ == "__main__":
    main()