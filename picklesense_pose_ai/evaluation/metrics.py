import numpy as np

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

from config.settings import (
    STROKE_CLASSES,
)


def calculate_metrics(
    y_true,
    y_pred,
):
    """
    Calculate overall and per-class metrics.

    Parameters
    ----------
    y_true:
        Ground-truth class indices.

    y_pred:
        Predicted class indices.

    Returns
    -------
    dict
    """

    y_true = np.asarray(
        y_true,
        dtype=np.int64,
    )

    y_pred = np.asarray(
        y_pred,
        dtype=np.int64,
    )

    if len(y_true) != len(y_pred):
        raise ValueError(
            "y_true and y_pred must "
            "have the same length."
        )

    if len(y_true) == 0:
        raise ValueError(
            "No predictions available "
            "for evaluation."
        )

    accuracy = accuracy_score(
        y_true,
        y_pred,
    )

    macro_precision = precision_score(
        y_true,
        y_pred,
        average="macro",
        zero_division=0,
    )

    macro_recall = recall_score(
        y_true,
        y_pred,
        average="macro",
        zero_division=0,
    )

    macro_f1 = f1_score(
        y_true,
        y_pred,
        average="macro",
        zero_division=0,
    )

    confusion = confusion_matrix(
        y_true,
        y_pred,
        labels=list(
            range(
                len(STROKE_CLASSES)
            )
        ),
    )

    report = classification_report(
        y_true,
        y_pred,
        labels=list(
            range(
                len(STROKE_CLASSES)
            )
        ),
        target_names=STROKE_CLASSES,
        output_dict=True,
        zero_division=0,
    )

    per_class = {}

    for class_name in STROKE_CLASSES:
        class_metrics = report.get(
            class_name,
            {},
        )

        per_class[
            class_name
        ] = {
            "precision": float(
                class_metrics.get(
                    "precision",
                    0.0,
                )
            ),

            "recall": float(
                class_metrics.get(
                    "recall",
                    0.0,
                )
            ),

            "f1": float(
                class_metrics.get(
                    "f1-score",
                    0.0,
                )
            ),

            "support": int(
                class_metrics.get(
                    "support",
                    0,
                )
            ),
        }

    return {
        "accuracy": float(
            accuracy
        ),

        "macro_precision": float(
            macro_precision
        ),

        "macro_recall": float(
            macro_recall
        ),

        "macro_f1": float(
            macro_f1
        ),

        "confusion_matrix": (
            confusion
        ),

        "per_class": (
            per_class
        ),
    }


def print_metrics(
    metrics,
):
    print()
    print(
        "Evaluation Metrics"
    )

    print(
        "------------------"
    )

    print(
        "Accuracy:       "
        f"{metrics['accuracy'] * 100:.2f}%"
    )

    print(
        "Macro Precision:"
        f" {metrics['macro_precision'] * 100:.2f}%"
    )

    print(
        "Macro Recall:   "
        f"{metrics['macro_recall'] * 100:.2f}%"
    )

    print(
        "Macro F1:       "
        f"{metrics['macro_f1'] * 100:.2f}%"
    )

    print()
    print(
        "Per-Class Metrics"
    )

    print(
        "-----------------"
    )

    for class_name in STROKE_CLASSES:
        values = (
            metrics["per_class"][
                class_name
            ]
        )

        print(
            f"{class_name:10s} | "
            f"P: "
            f"{values['precision'] * 100:6.2f}% | "
            f"R: "
            f"{values['recall'] * 100:6.2f}% | "
            f"F1: "
            f"{values['f1'] * 100:6.2f}% | "
            f"N: "
            f"{values['support']}"
        )


def print_confusion_matrix(
    confusion,
):
    confusion = np.asarray(
        confusion
    )

    print()
    print(
        "Confusion Matrix"
    )

    print(
        "----------------"
    )

    header = (
        "True\\Pred".ljust(12)
    )

    for class_name in STROKE_CLASSES:
        header += (
            f"{class_name[:6]:>8}"
        )

    print(
        header
    )

    for row_index, class_name in enumerate(
        STROKE_CLASSES
    ):
        row_text = (
            class_name[:10]
            .ljust(12)
        )

        for value in confusion[
            row_index
        ]:
            row_text += (
                f"{int(value):>8}"
            )

        print(
            row_text
        )