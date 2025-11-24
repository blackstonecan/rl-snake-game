from PyQt5.QtWidgets import QMainWindow, QPushButton, QVBoxLayout, QWidget
from PyQt5.QtCore import Qt

from windows.MainWindow import MainWindow

class StarterWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Snake Game Multiplayer Starter")
        self.setFixedSize(300, 150)
        
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        layout = QVBoxLayout()
        
        start_btn = QPushButton("Start Snake Game Multiplayer")
        start_btn.clicked.connect(self.start_game)
        layout.addWidget(start_btn, alignment=Qt.AlignCenter)
        
        central_widget.setLayout(layout)
    
    def start_game(self):
        self.main_window = MainWindow()
        self.main_window.show()
        self.close()
