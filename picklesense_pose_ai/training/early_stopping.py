from pathlib import Path

import torch

from config.settings import (
    CHECKPOINT_DIR,
    BEST_MODEL_FILENAME,
)


class EarlyStopping:
    def __init__(
        self,
        patience=12,
        min_delta=0.0,
        checkpoint_path=None,
    ):
        self.patience = patience
        self.min_delta = min_delta

        if checkpoint_path is None:
            checkpoint_path = (
                CHECKPOINT_DIR
                / BEST_MODEL_FILENAME
            )

        self.checkpoint_path = Path(
            checkpoint_path
        )

        self.best_loss = None
        self.counter = 0
        self.should_stop = False

    def __call__(
        self,
        validation_loss,
        model,
        optimizer=None,
        epoch=None,
        extra_state=None,
    ):
        if self.best_loss is None:
            self.best_loss = (
                validation_loss
            )

            self._save_checkpoint(
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                validation_loss=validation_loss,
                extra_state=extra_state,
            )

            return True

        improvement = (
            self.best_loss
            - validation_loss
        )

        if improvement > self.min_delta:
            self.best_loss = (
                validation_loss
            )

            self.counter = 0

            self._save_checkpoint(
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                validation_loss=validation_loss,
                extra_state=extra_state,
            )

            return True

        self.counter += 1

        if self.counter >= self.patience:
            self.should_stop = True

        return False

    def _save_checkpoint(
        self,
        model,
        optimizer=None,
        epoch=None,
        validation_loss=None,
        extra_state=None,
    ):
        self.checkpoint_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        checkpoint = {
            "model_state_dict": (
                model.state_dict()
            ),

            "epoch": (
                epoch
            ),

            "validation_loss": (
                validation_loss
            ),

            "best_validation_loss": (
                self.best_loss
            ),
        }

        if optimizer is not None:
            checkpoint[
                "optimizer_state_dict"
            ] = optimizer.state_dict()

        if extra_state is not None:
            checkpoint[
                "extra_state"
            ] = extra_state

        torch.save(
            checkpoint,
            self.checkpoint_path,
        )

    def load_best_model(
        self,
        model,
        map_location="cpu",
    ):
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(
                "Best-model checkpoint "
                f"not found: "
                f"{self.checkpoint_path}"
            )

        checkpoint = torch.load(
            self.checkpoint_path,
            map_location=map_location,
        )

        model.load_state_dict(
            checkpoint[
                "model_state_dict"
            ]
        )

        return checkpoint