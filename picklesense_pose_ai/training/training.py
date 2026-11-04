import json
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from config.settings import (
    MANIFEST_DIR,
    TRAIN_MANIFEST_FILENAME,
    VALIDATION_MANIFEST_FILENAME,
    BATCH_SIZE,
    LEARNING_RATE,
    WEIGHT_DECAY,
    MAX_EPOCHS,
    EARLY_STOPPING_PATIENCE,
    NUM_WORKERS,
    PIN_MEMORY,
    RANDOM_SEED,
    OUTPUT_DIR,
    MODEL_DIR,
    FINAL_MODEL_FILENAME,
)

from dataset.pose_dataset import (
    PickleSensePoseDataset,
    load_feature_sequences_for_normalizer,
)

from preprocessing.pose_normalizer import (
    PoseFeatureNormalizer,
)

from models.pose_tcn import (
    create_model,
)

from training.early_stopping import (
    EarlyStopping,
)


TRAINING_HISTORY_FILENAME = (
    "training_history.json"
)


def set_random_seed():
    torch.manual_seed(
        RANDOM_SEED
    )

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(
            RANDOM_SEED
        )


def get_device():
    if torch.cuda.is_available():
        return torch.device(
            "cuda"
        )

    return torch.device(
        "cpu"
    )


def create_datasets():
    train_manifest = (
        MANIFEST_DIR
        / TRAIN_MANIFEST_FILENAME
    )

    validation_manifest = (
        MANIFEST_DIR
        / VALIDATION_MANIFEST_FILENAME
    )

    if not train_manifest.exists():
        raise FileNotFoundError(
            "Train manifest not found:\n"
            f"{train_manifest}\n\n"
            "Run dataset.splitter first."
        )

    if not validation_manifest.exists():
        raise FileNotFoundError(
            "Validation manifest not found:\n"
            f"{validation_manifest}\n\n"
            "Run dataset.splitter first."
        )

    # --------------------------------------------------
    # Fit normalizer using TRAINING DATA ONLY
    # --------------------------------------------------

    print()
    print(
        "Loading training features "
        "for normalizer..."
    )

    train_feature_sequences = (
        load_feature_sequences_for_normalizer(
            train_manifest
        )
    )

    normalizer = (
        PoseFeatureNormalizer()
    )

    normalizer.fit(
        train_feature_sequences
    )

    normalizer_path = (
        normalizer.save()
    )

    print(
        f"Normalizer saved to:"
    )

    print(
        normalizer_path
    )

    # --------------------------------------------------
    # Create datasets
    # --------------------------------------------------

    train_dataset = (
        PickleSensePoseDataset(
            manifest_path=train_manifest,
            normalizer=normalizer,
            training=True,
            augment=True,
            random_seed=(
                RANDOM_SEED
            ),
        )
    )

    validation_dataset = (
        PickleSensePoseDataset(
            manifest_path=validation_manifest,
            normalizer=normalizer,
            training=False,
            augment=False,
            random_seed=(
                RANDOM_SEED
            ),
        )
    )

    return (
        train_dataset,
        validation_dataset,
        normalizer,
    )


def create_dataloaders(
    train_dataset,
    validation_dataset,
):
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
    )

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
    )

    return (
        train_loader,
        validation_loader,
    )


def train_one_epoch(
    model,
    data_loader,
    criterion,
    optimizer,
    device,
):
    model.train()

    total_loss = 0.0

    correct_predictions = 0
    total_samples = 0

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

        optimizer.zero_grad()

        logits = model(
            features
        )

        loss = criterion(
            logits,
            labels,
        )

        loss.backward()

        optimizer.step()

        batch_size = (
            labels.size(0)
        )

        total_loss += (
            loss.item()
            * batch_size
        )

        predictions = (
            torch.argmax(
                logits,
                dim=1,
            )
        )

        correct_predictions += (
            predictions
            .eq(labels)
            .sum()
            .item()
        )

        total_samples += (
            batch_size
        )

    average_loss = (
        total_loss
        / total_samples
    )

    accuracy = (
        correct_predictions
        / total_samples
    )

    return (
        average_loss,
        accuracy,
    )


@torch.no_grad()
def validate_one_epoch(
    model,
    data_loader,
    criterion,
    device,
):
    model.eval()

    total_loss = 0.0

    correct_predictions = 0
    total_samples = 0

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

        loss = criterion(
            logits,
            labels,
        )

        batch_size = (
            labels.size(0)
        )

        total_loss += (
            loss.item()
            * batch_size
        )

        predictions = (
            torch.argmax(
                logits,
                dim=1,
            )
        )

        correct_predictions += (
            predictions
            .eq(labels)
            .sum()
            .item()
        )

        total_samples += (
            batch_size
        )

    average_loss = (
        total_loss
        / total_samples
    )

    accuracy = (
        correct_predictions
        / total_samples
    )

    return (
        average_loss,
        accuracy,
    )


def save_training_history(
    history,
):
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        OUTPUT_DIR
        / TRAINING_HISTORY_FILENAME
    )

    with open(
        output_path,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            history,
            file,
            indent=4,
        )

    return output_path


def save_final_model(
    model,
):
    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        MODEL_DIR
        / FINAL_MODEL_FILENAME
    )

    torch.save(
        model.state_dict(),
        output_path,
    )

    return output_path


def train_model():
    set_random_seed()

    device = get_device()

    print()
    print(
        "PickleSense Pose-TCN Training"
    )

    print(
        "----------------------------"
    )

    print(
        f"Device: {device}"
    )

    # --------------------------------------------------
    # Dataset
    # --------------------------------------------------

    (
        train_dataset,
        validation_dataset,
        normalizer,
    ) = create_datasets()

    (
        train_loader,
        validation_loader,
    ) = create_dataloaders(
        train_dataset,
        validation_dataset,
    )

    print()
    print(
        f"Training trials: "
        f"{len(train_dataset)}"
    )

    print(
        f"Validation trials: "
        f"{len(validation_dataset)}"
    )

    # --------------------------------------------------
    # Model
    # --------------------------------------------------

    model = (
        create_model()
        .to(device)
    )

    criterion = (
        nn.CrossEntropyLoss()
    )

    optimizer = (
        torch.optim.AdamW(
            model.parameters(),
            lr=LEARNING_RATE,
            weight_decay=WEIGHT_DECAY,
        )
    )

    scheduler = (
        torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.5,
            patience=4,
        )
    )

    early_stopping = EarlyStopping(
        patience=(
            EARLY_STOPPING_PATIENCE
        ),
        min_delta=1e-4,
    )

    history = {
        "train_loss": [],
        "train_accuracy": [],
        "validation_loss": [],
        "validation_accuracy": [],
        "learning_rate": [],
    }

    # --------------------------------------------------
    # Training loop
    # --------------------------------------------------

    for epoch in range(
        1,
        MAX_EPOCHS + 1,
    ):
        (
            train_loss,
            train_accuracy,
        ) = train_one_epoch(
            model=model,
            data_loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
        )

        (
            validation_loss,
            validation_accuracy,
        ) = validate_one_epoch(
            model=model,
            data_loader=validation_loader,
            criterion=criterion,
            device=device,
        )

        scheduler.step(
            validation_loss
        )

        current_lr = (
            optimizer
            .param_groups[0]["lr"]
        )

        history[
            "train_loss"
        ].append(
            train_loss
        )

        history[
            "train_accuracy"
        ].append(
            train_accuracy
        )

        history[
            "validation_loss"
        ].append(
            validation_loss
        )

        history[
            "validation_accuracy"
        ].append(
            validation_accuracy
        )

        history[
            "learning_rate"
        ].append(
            current_lr
        )

        improved = early_stopping(
            validation_loss=(
                validation_loss
            ),
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            extra_state={
                "train_loss": (
                    train_loss
                ),
                "train_accuracy": (
                    train_accuracy
                ),
                "validation_accuracy": (
                    validation_accuracy
                ),
            },
        )

        marker = (
            " *"
            if improved
            else ""
        )

        print(
            f"Epoch "
            f"{epoch:03d}/{MAX_EPOCHS} | "
            f"Train Loss: "
            f"{train_loss:.4f} | "
            f"Train Acc: "
            f"{train_accuracy * 100:.2f}% | "
            f"Val Loss: "
            f"{validation_loss:.4f} | "
            f"Val Acc: "
            f"{validation_accuracy * 100:.2f}% | "
            f"LR: "
            f"{current_lr:.6f}"
            f"{marker}"
        )

        save_training_history(
            history
        )

        if (
            early_stopping.should_stop
        ):
            print()
            print(
                "Early stopping triggered."
            )
            break

    # --------------------------------------------------
    # Restore BEST model
    # --------------------------------------------------

    print()
    print(
        "Loading best validation model..."
    )

    checkpoint = (
        early_stopping.load_best_model(
            model=model,
            map_location=device,
        )
    )

    best_epoch = (
        checkpoint.get(
            "epoch"
        )
    )

    best_validation_loss = (
        checkpoint.get(
            "validation_loss"
        )
    )

    extra_state = (
        checkpoint.get(
            "extra_state",
            {},
        )
    )

    print(
        f"Best epoch: "
        f"{best_epoch}"
    )

    print(
        f"Best validation loss: "
        f"{best_validation_loss:.4f}"
    )

    if (
        "validation_accuracy"
        in extra_state
    ):
        print(
            "Best validation accuracy: "
            f"{extra_state['validation_accuracy'] * 100:.2f}%"
        )

    # --------------------------------------------------
    # Save final deployable model state
    # --------------------------------------------------

    final_model_path = (
        save_final_model(
            model
        )
    )

    history_path = (
        save_training_history(
            history
        )
    )

    print()
    print(
        "Training complete."
    )

    print()
    print(
        "Final best model:"
    )

    print(
        final_model_path
    )

    print()
    print(
        "Training history:"
    )

    print(
        history_path
    )

    return model


def main():
    train_model()


if __name__ == "__main__":
    main()