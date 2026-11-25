from gui import DigitsRecognizer

import sys
from PySide6.QtWidgets import QApplication

app = QApplication(sys.argv)

window = DigitsRecognizer()
window.show()

app.exec()