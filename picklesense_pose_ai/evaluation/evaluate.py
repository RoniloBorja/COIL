import json

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from config.settings import (
    MANIFEST_DIR,
    TEST_MANIFEST_FILENAME,
    BATCH_SIZE,
    NUM_WORKERS,
    PIN_MEMORY,
    MODEL_DIR,
    FINAL_MODEL_FILENAME,
    REPORT_DIR,
    PLOT_DIR,
    INDEX_TO_CLASS,
    STROKE_CLASSES,
)

from dataset.pose_dataset import (
    PickleSensePoseDataset,
)

from preprocessing.pose_normalizer import (
    PoseFeatureNormalizer,
)

from models.pose_tcn import (
    create_model,
)

from evaluation.metrics import (
    calculate_metrics,
    print_metrics,
    print_confusion_matrix,
)


EVALUATION_REPORT_FILENAME = (
    "test_evaluation.json"
)

CONFUSION_MATRIX_FILENAME = (
    "confusion_matrix.png"
)


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")

    return torch.device("cpu")


def load_test_dataset():
    test_manifest = (
        MANIFEST_DIR
        / TEST_MANIFEST_FILENAME
    )

    if not test_manifest.exists():
        raise FileNotFoundError(
            "Test manifest not found:\n"
            f"{test_manifest}\n\n"
            "Run dataset.splitter first."
        )

    normalizer = (
        PoseFeatureNormalizer()
    )

    normalizer.load()

    test_dataset = (
        PickleSensePoseDataset(
            manifest_path=test_manifest,
            normalizer=normalizer,
            training=False,
            augment=False,
        )
    )

    return (
        test_dataset,
        normalizer,
    )


def load_model(
    device,
):
    model_path = (
        MODEL_DIR
        / FINAL_MODEL_FILENAME
    )

    if not model_path.exists():
        raise FileNotFoundError(
            "Trained model not found:\n"
            f"{model_path}\n\n"
            "Run training first."
        )

    model = create_model()

    state_dict = torch.load(
        model_path,
        map_location=device,
    )

    model.load_state_dict(
        state_dict
    )

    model = model.to(
        device
    )

    model.eval()

    return model


@torch.no_grad()
def run_inference(
    model,
    data_loader,
    device,
):
    y_true = []
    y_pred = []

    probabilities_list = []

    trial_results = []

    softmax = torch.nn.Softmax(
        dim=1
    )

    for batch in data_loader:
        features = (
            batch["features"]
            .to(
                device,
                non_blocking=True,
            )
        )

        labels = (
            batch["label"]
            .to(
                device,
                non_blocking=True,
            )
        )

        logits = model(
            features
        )

        probabilities = (
            softmax(
                logits
            )
        )

        predictions = (
            torch.argmax(
                probabilities,
                dim=1,
            )
        )

        y_true.extend(
            labels
            .cpu()
            .numpy()
            .tolist()
        )

        y_pred.extend(
            predictions
            .cpu()
            .numpy()
            .tolist()
        )

        probabilities_cpu = (
            probabilities
            .cpu()
            .numpy()
        )

        probabilities_list.extend(
            probabilities_cpu.tolist()
        )

        batch_size = (
            labels.size(0)
        )

        for i in range(
            batch_size
        ):
            true_index = int(
                labels[i].item()
            )

            predicted_index = int(
                predictions[i].item()
            )

            confidence = float(
                probabilities_cpu[
                    i,
                    predicted_index,
                ]
            )

            trial_results.append(
                {
                    "participant_id": (
                        batch[
                            "participant_id"
                        ][i]
                    ),

                    "session_number": (
                        batch[
                            "session_number"
                        ][i]
                    ),

                    "trial_number": (
                        batch[
                            "trial_number"
                        ][i]
                    ),

                    "true_class": (
                        INDEX_TO_CLASS[
                            true_index
                        ]
                    ),

                    "predicted_class": (
                        INDEX_TO_CLASS[
                            predicted_index
                        ]
                    ),

                    "confidence": (
                        confidence
                    ),

                    "correct": (
                        true_index
                        == predicted_index
                    ),

                    "probabilities": {
                        STROKE_CLASSES[
                            class_index
                        ]: float(
                            probabilities_cpu[
                                i,
                                class_index,
                            ]
                        )

                        for class_index
                        in range(
                            len(
                                STROKE_CLASSES
                            )
                        )
                    },
                }
            )

    return (
        y_true,
        y_pred,
        probabilities_list,
        trial_results,
    )


def save_confusion_matrix_plot(
    confusion,
):
    PLOT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        PLOT_DIR
        / CONFUSION_MATRIX_FILENAME
    )

    figure = plt.figure(
        figsize=(9, 7)
    )

    axis = figure.add_subplot(
        111
    )

    image = axis.imshow(
        confusion,
    )

    figure.colorbar(
        image,
        ax=axis,
    )

    axis.set_xticks(
        range(
            len(
                STROKE_CLASSES
            )
        )
    )

    axis.set_yticks(
        range(
            len(
                STROKE_CLASSES
            )
        )
    )

    axis.set_xticklabels(
        STROKE_CLASSES,
        rotation=45,
        ha="right",
    )

    axis.set_yticklabels(
        STROKE_CLASSES
    )

    axis.set_xlabel(
        "Predicted Class"
    )

    axis.set_ylabel(
        "True Class"
    )

    axis.set_title(
        "PickleSense Pose-TCN Confusion Matrix"
    )

    for row in range(
        confusion.shape[0]
    ):
        for column in range(
            confusion.shape[1]
        ):
            axis.text(
                column,
                row,
                str(
                    int(
                        confusion[
                            row,
                            column,
                        ]
                    )
                ),
                ha="center",
                va="center",
            )

    figure.tight_layout()

    figure.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(
        figure
    )

    return output_path


def save_evaluation_report(
    metrics,
    trial_results,
):
    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        REPORT_DIR
        / EVALUATION_REPORT_FILENAME
    )

    report = {
        "accuracy": (
            metrics["accuracy"]
        ),

        "macro_precision": (
            metrics[
                "macro_precision"
            ]
        ),

        "macro_recall": (
            metrics[
                "macro_recall"
            ]
        ),

        "macro_f1": (
            metrics[
                "macro_f1"
            ]
        ),

        "per_class": (
            metrics[
                "per_class"
            ]
        ),

        "confusion_matrix": (
            metrics[
                "confusion_matrix"
            ].tolist()
        ),

        "trials": (
            trial_results
        ),
    }

    with open(
        output_path,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            report,
            file,
            indent=4,
        )

    return output_path


def evaluate_model():
    device = get_device()

    print()
    print(
        "PickleSense Pose-TCN Test Evaluation"
    )

    print(
        "------------------------------------"
    )

    print(
        f"Device: {device}"
    )

    test_dataset, _ = (
        load_test_dataset()
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
    )

    print(
        f"Test trials: "
        f"{len(test_dataset)}"
    )

    model = load_model(
        device
    )

    (
        y_true,
        y_pred,
        _,
        trial_results,
    ) = run_inference(
        model=model,
        data_loader=test_loader,
        device=device,
    )

    metrics = calculate_metrics(
        y_true,
        y_pred,
    )

    print_metrics(
        metrics
    )

    print_confusion_matrix(
        metrics[
            "confusion_matrix"
        ]
    )

    confusion_path = (
        save_confusion_matrix_plot(
            metrics[
                "confusion_matrix"
            ]
        )
    )

    report_path = (
        save_evaluation_report(
            metrics=metrics,
            trial_results=trial_results,
        )
    )

    print()
    print(
        "Evaluation complete."
    )

    print()
    print(
        "Report:"
    )

    print(
        report_path
    )

    print()
    print(
        "Confusion matrix plot:"
    )

    print(
        confusion_path
    )

    return metrics


def main():
    evaluate_model()


if __name__ == "__main__":
    main()