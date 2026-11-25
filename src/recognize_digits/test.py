from models import LeNet
from utils import get_model

from sklearn.datasets import fetch_openml

import torch
from torch.utils.data import TensorDataset, DataLoader

from torchmetrics import Accuracy, Precision, Recall, F1Score, AUROC

X, y = fetch_openml(
    name='mnist_784',
    return_X_y=True,
    as_frame=False,
    parser='liac-arff'
)

X, y = X.astype('float32'), y.astype('int64')
X, y = torch.from_numpy(X).float().reshape(-1, 1, 28, 28) / 255, torch.from_numpy(y).long()

dataset = TensorDataset(X, y)
dataloader = DataLoader(dataset, batch_size=64, shuffle=False)

torch.manual_seed(42)

model = LeNet()
model.load_state_dict(
    torch.load(get_model())
)
accuracy_fn = Accuracy(task='multiclass', num_classes=10)
precision_fn = Precision(task='multiclass', num_classes=10)
recall_fn = Recall(task='multiclass', num_classes=10)
f1_fn = F1Score(task='multiclass', num_classes=10)
auroc_fn = AUROC(task='multiclass', num_classes=10)

for X_batch, y_batch in dataloader:
    y_pred = model(X_batch)

    accuracy_fn.update(y_pred, y_batch)
    precision_fn.update(y_pred, y_batch)
    recall_fn.update(y_pred, y_batch)
    f1_fn.update(y_pred, y_batch)
    auroc_fn.update(y_pred, y_batch)

accuracy = accuracy_fn.compute()
precision = precision_fn.compute()
recall = recall_fn.compute()
f1 = f1_fn.compute()
auroc = auroc_fn.compute()

print(f'Accuracy: {accuracy:.1%} | Precision: {precision: .1%} | Recall: {recall: .1%} | F1: {f1: .3f} | AUROC: {auroc: .3f}')
