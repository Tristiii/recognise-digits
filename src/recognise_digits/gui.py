from utils import get_model
from models import LeNet

import numpy as np
import cv2
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QLabel, QPushButton
)
from PySide6.QtGui import (
    QImage, QPainter, QPen, QPaintEvent, QMouseEvent
)
from PySide6.QtCore import (
    Qt, Signal
)

import torch

class DigitsRecognizer(QWidget):
    def __init__(self) -> None:
        super().__init__()
        self.setFixedSize(300, 400)
        self.setWindowTitle('Recognize digits')

        layout = QVBoxLayout()

        self.description = QLabel('Please write your digit below to be recognized.')
        layout.addWidget(self.description)

        self.canvas = Canvas()
        self.canvas.changed.connect(self.predict_digit)
        layout.addWidget(self.canvas)

        self.clear_button = QPushButton()
        self.clear_button.setText('Clear canvas')
        self.clear_button.clicked.connect(self.canvas.clear)
        self.clear_button.clicked.connect(self.clear)
        layout.addWidget(self.clear_button)

        self.prediction = QLabel('Please draw in the canvas')
        self.prediction.setStyleSheet(f"""
            background-color: orange;
            color: white;
            font-weight: bold;
            padding: 10px;
        """)
        self.prediction.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.prediction)

        self.setLayout(layout)

        self.model = LeNet()
        self.model.load_state_dict(
            torch.load(f=get_model())
        )

    def predict_digit(self) -> None:
        X = self.canvas.to_tensor()

        self.model.eval()
        with torch.inference_mode():
            y_pred = self.model(X)
            probs = torch.softmax(y_pred, dim=1)

            prediction = probs.argmax(dim=1).item()
            chance = torch.max(probs).item()

        self.prediction.setText(f'Prediction: {prediction} ({chance:.1%})')
        self.prediction.setStyleSheet(f"""
            background-color: blue;
            color: white;
            font-weight: bold;
            padding: 10px;
        """)

    def clear(self) -> None:
        self.prediction.setText('Please draw in the canvas')
        self.prediction.setStyleSheet(f"""
            background-color: orange;
            color: white;
            font-weight: bold;
            padding: 10px;
        """)


class Canvas(QWidget):
    changed = Signal()

    def __init__(self) -> None:
        super().__init__()
        self.setFixedSize(300, 300)

        self.image = QImage(300, 300, QImage.Format.Format_Grayscale8)
        self.image.fill(255)
        self.last_point = None

    def paintEvent(self, event: QPaintEvent) -> None:
        painter = QPainter(self)
        painter.drawImage(0, 0, self.image)

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self.last_point = event.position().toPoint()

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        if event.buttons() & Qt.MouseButton.LeftButton:
            current_point = event.position().toPoint()

            painter = QPainter(self.image)
            painter.setPen(QPen(Qt.GlobalColor.black, 15))
            painter.drawLine(self.last_point, current_point)
            self.last_point = current_point
            self.update()

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        self.last_point = None
        self.changed.emit()

    # Convert to tensor
    def to_tensor(self) -> torch.Tensor:
        bits = self.image.bits()
        image = np.frombuffer(bits, dtype=np.uint8).reshape(300, 300)

        blurred = cv2.GaussianBlur(image, (15, 15), 0)
        resized = cv2.resize(blurred, (28, 28))

        image = (255 - resized).astype(np.float32) / 255.0

        return torch.from_numpy(image).reshape(1, 1, 28, 28)

    def clear(self) -> None:
        self.image.fill(255)
        self.update()
        self.changed.emit()