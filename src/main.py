import sys
from PyQt5.QtWidgets import (QApplication)
from windows.StarterWindow import StarterWindow



if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = StarterWindow()
    window.show()
    sys.exit(app.exec_())