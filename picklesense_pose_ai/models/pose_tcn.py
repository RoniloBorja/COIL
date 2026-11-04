import torch
import torch.nn as nn

from config.settings import (
    FEATURE_COUNT,
    NUM_CLASSES,
    DROPOUT,
)


class ResidualTCNBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        dilation=1,
        dropout=DROPOUT,
    ):
        super().__init__()

        padding = (
            (kernel_size - 1)
            * dilation
            // 2
        )

        self.conv1 = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding=padding,
            dilation=dilation,
        )

        self.bn1 = nn.BatchNorm1d(
            out_channels
        )

        self.relu1 = nn.ReLU()

        self.dropout1 = nn.Dropout(
            dropout
        )

        self.conv2 = nn.Conv1d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding=padding,
            dilation=dilation,
        )

        self.bn2 = nn.BatchNorm1d(
            out_channels
        )

        self.dropout2 = nn.Dropout(
            dropout
        )

        if (
            in_channels
            != out_channels
        ):
            self.residual = nn.Conv1d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
            )

        else:
            self.residual = (
                nn.Identity()
            )

        self.relu_out = nn.ReLU()

    def forward(
        self,
        x,
    ):
        identity = (
            self.residual(
                x
            )
        )

        out = self.conv1(
            x
        )

        out = self.bn1(
            out
        )

        out = self.relu1(
            out
        )

        out = self.dropout1(
            out
        )

        out = self.conv2(
            out
        )

        out = self.bn2(
            out
        )

        out = self.dropout2(
            out
        )

        out = (
            out
            + identity
        )

        out = self.relu_out(
            out
        )

        return out


class PoseTCN(nn.Module):
    def __init__(
        self,
        input_channels=FEATURE_COUNT,
        num_classes=NUM_CLASSES,
        dropout=DROPOUT,
    ):
        super().__init__()

        # -----------------------------------------
        # Initial temporal feature extraction
        # -----------------------------------------

        self.input_conv = nn.Conv1d(
            in_channels=input_channels,
            out_channels=64,
            kernel_size=5,
            padding=2,
        )

        self.input_bn = nn.BatchNorm1d(
            64
        )

        self.input_relu = nn.ReLU()

        # -----------------------------------------
        # Residual temporal blocks
        # -----------------------------------------

        self.block1 = ResidualTCNBlock(
            in_channels=64,
            out_channels=128,
            kernel_size=3,
            dilation=1,
            dropout=dropout,
        )

        self.block2 = ResidualTCNBlock(
            in_channels=128,
            out_channels=128,
            kernel_size=3,
            dilation=2,
            dropout=dropout,
        )

        self.block3 = ResidualTCNBlock(
            in_channels=128,
            out_channels=256,
            kernel_size=3,
            dilation=4,
            dropout=dropout,
        )

        # -----------------------------------------
        # Global temporal pooling
        # -----------------------------------------

        self.global_pool = (
            nn.AdaptiveAvgPool1d(
                output_size=1
            )
        )

        # -----------------------------------------
        # Final classifier
        # -----------------------------------------

        self.classifier = nn.Sequential(
            nn.Flatten(),

            nn.Dropout(
                dropout
            ),

            nn.Linear(
                256,
                128,
            ),

            nn.ReLU(),

            nn.Dropout(
                dropout
            ),

            nn.Linear(
                128,
                num_classes,
            ),
        )

    def forward(
        self,
        x,
    ):
        """
        Expected input shape:

            [batch, 44, 64]

        Output shape:

            [batch, 6]

        Output values are logits.
        Do NOT apply softmax here because
        CrossEntropyLoss expects raw logits.
        """

        x = self.input_conv(
            x
        )

        x = self.input_bn(
            x
        )

        x = self.input_relu(
            x
        )

        x = self.block1(
            x
        )

        x = self.block2(
            x
        )

        x = self.block3(
            x
        )

        x = self.global_pool(
            x
        )

        logits = self.classifier(
            x
        )

        return logits


def create_model():
    return PoseTCN()


def count_parameters(
    model,
):
    return sum(
        parameter.numel()
        for parameter
        in model.parameters()
        if parameter.requires_grad
    )


if __name__ == "__main__":
    model = create_model()

    test_input = torch.randn(
        4,
        FEATURE_COUNT,
        64,
    )

    output = model(
        test_input
    )

    print(
        "Input shape:",
        test_input.shape,
    )

    print(
        "Output shape:",
        output.shape,
    )

    print(
        "Trainable parameters:",
        count_parameters(
            model
        ),
    )