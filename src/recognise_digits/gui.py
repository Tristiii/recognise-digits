from predict import predict

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

class DigitsRecogniser(QWidget):
    def __init__(self) -> None:
        super().__init__()
        self.setFixedSize(300, 400)
        self.setWindowTitle('Digits recogniser')

        layout = QVBoxLayout()

        self.description = QLabel('Please write your digit below to be recognised.')
        layout.addWidget(self.description)

        self.canvas = Canvas()
        self.canvas.changed.connect(self.predict_digit)
        layout.addWidget(self.canvas)

        self.clear_button = QPushButton()
        self.clear_button.setText('Clear canvas')
        self.clear_button.clicked.connect(self.canvas.clear)
        layout.addWidget(self.clear_button)

        self.prediction = QLabel()
        self.prediction.setStyleSheet(f"""
            background-color: blue;
            color: white;
            font-weight: bold;
            padding: 10px;
        """)
        self.prediction.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.prediction)

        self.setLayout(layout)

    def predict_digit(self) -> None:
        data = self.canvas.to_numpy()
        prediction = predict(data.reshape(1, -1))

        self.prediction.setText(f'Prediction: {prediction}')


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

            self.changed.emit()

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        self.last_point = None

    # NumPy array of image
    def to_numpy(self) -> np.ndarray:
        # Create NumPy array and reshape to image
        bits = self.image.bits()
        image = np.frombuffer(bits, dtype=np.uint8).reshape(300, 300)

        # Gaussian blur
        blurred = cv2.GaussianBlur(image, (15, 15), 0)

        # Resizing and scaling
        resized = cv2.resize(blurred, (8, 8))
        scaled = (resized / 255 * 16).astype(np.uint8)

        # Flattened and inversed array
        return (16 - scaled).flatten()

    def clear(self) -> None:
        self.image.fill(255)
        self.update()
        self.changed.emit()