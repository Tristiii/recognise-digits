from utils import get_model
from train import train_model
from gui import DigitsRecogniser

import sys
from PySide6.QtWidgets import QApplication

if not get_model().exists():
    train_model()

app = QApplication(sys.argv)

window = DigitsRecogniser()
window.show()

app.exec()