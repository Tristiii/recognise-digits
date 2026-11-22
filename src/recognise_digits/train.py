from utils import get_model

from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split

import torch
from torch import nn

from torch.utils.data import TensorDataset, DataLoader

from torchmetrics import Accuracy

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

def train_model() -> None:
    # Fetching data and preprocessing to tensors
    X, y = fetch_openml(
        name='mnist_784',
        return_X_y=True,
        as_frame=False,
        parser='liac-arff'
    )

    X, y = X.astype('float32'), y.astype('int64')
    X, y = torch.from_numpy(X).float().reshape(-1, 1, 28, 28) / 255, torch.from_numpy(y).long()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Creating DataLoaders
    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64)

    # Creating training environment
    torch.manual_seed(42)
    model = LeNet()
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    accuracy_fn = Accuracy(task='multiclass', num_classes=10)

    # Training and evaluating
    for epoch in range(25):
        model.train()
        train_loss = 0
        for X_batch, y_batch in train_loader:
            y_pred = model(X_batch)
            loss = loss_fn(y_pred, y_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)

        model.eval()
        with torch.inference_mode():
            for X_batch, y_batch in test_loader:
                y_pred = model(X_batch)
                accuracy_fn.update(y_pred, y_batch)

            accuracy = accuracy_fn.compute()
            accuracy_fn.reset()
            print(f'Epoch {epoch} | Loss: {train_loss:.3f} | Test accuracy: {accuracy:.2%}')

    # Saving the model
    torch.save(obj=model.state_dict(), f=get_model())
    print(f'Saved model to {get_model()}')