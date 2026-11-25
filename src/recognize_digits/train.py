from utils import get_model
from models import LeNet

from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split

import torch
from torch import nn

from torch.utils.data import TensorDataset, DataLoader

from torchmetrics import Accuracy

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
val_dataset = TensorDataset(X_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64)

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
        for X_batch, y_batch in val_loader:
            y_pred = model(X_batch)
            accuracy_fn.update(y_pred, y_batch)

        accuracy = accuracy_fn.compute()
        accuracy_fn.reset()
        print(f'Epoch {epoch} | Loss: {train_loss:.3f} | Validation accuracy: {accuracy:.2%}')

# Saving the model
torch.save(obj=model.state_dict(), f=get_model())
print(f'Saved model to {get_model()}')