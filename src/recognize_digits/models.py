import torch
from torch import nn

# LeNet neural network
class LeNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.block_1 = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=6,
                kernel_size=(5, 5),
                padding=2
            ),
            nn.Sigmoid(),
            nn.AvgPool2d(
                kernel_size=(2, 2),
                stride=2
            )
        )
        self.block_2 = nn.Sequential(
            nn.Conv2d(
                in_channels=6,
                out_channels=16,
                kernel_size=(5, 5)
            ),
            nn.Sigmoid(),
            nn.AvgPool2d(
                kernel_size=(2, 2),
                stride=2
            ),
            nn.Flatten()
        )

        self.block_3 = nn.Sequential(
            nn.Linear(
                in_features=5 * 5 * 16,
                out_features=120
            ),
            nn.Sigmoid(),
            nn.Linear(
                in_features=120,
                out_features=84
            ),
            nn.Sigmoid(),
            nn.Linear(
                in_features=84,
                out_features=10
            )
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_1 = self.block_1(x)
        x_2 = self.block_2(x_1)
        x_3 = self.block_3(x_2)
        return x_3