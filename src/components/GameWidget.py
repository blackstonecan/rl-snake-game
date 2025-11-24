from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QPushButton, 
                             QVBoxLayout, QHBoxLayout, QComboBox, QLabel)
from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtGui import QPainter, QColor, QFont

class GameWidget(QWidget):
    def __init__(self, game):
        super().__init__()
        self.game = game
        self.cell_size = 20
        self.setFixedSize(self.game.grid_size * self.cell_size, 
                         self.game.grid_size * self.cell_size)
    
    def paintEvent(self, event):
        painter = QPainter(self)
        
        # Draw grid
        painter.fillRect(0, 0, self.width(), self.height(), QColor(20, 20, 20))
        
        # Draw grid lines
        painter.setPen(QColor(40, 40, 40))
        for i in range(self.game.grid_size + 1):
            painter.drawLine(i * self.cell_size, 0, 
                           i * self.cell_size, self.height())
            painter.drawLine(0, i * self.cell_size, 
                           self.width(), i * self.cell_size)
        
        # Draw apples
        painter.setBrush(QColor(255, 0, 0))
        for apple_x, apple_y in self.game.apples:
            painter.drawEllipse(apple_x * self.cell_size + 2, 
                               apple_y * self.cell_size + 2,
                               self.cell_size - 4, self.cell_size - 4)
        
        # Draw snake 1 (Green) - always draw, even if dead
        for i, (x, y) in enumerate(self.game.snake1):
            if self.game.game_over1:
                # Dead - draw in dark gray
                if i == 0:
                    painter.setBrush(QColor(60, 60, 60))  # Dark gray head
                else:
                    painter.setBrush(QColor(40, 40, 40))  # Darker gray body
            else:
                # Alive - draw in green
                if i == 0:
                    painter.setBrush(QColor(0, 255, 0))  # Head - bright green
                else:
                    painter.setBrush(QColor(0, 200, 0))  # Body - darker green
            painter.drawRect(x * self.cell_size + 1, y * self.cell_size + 1,
                           self.cell_size - 2, self.cell_size - 2)
        
        # Draw snake 2 (Blue) - always draw, even if dead
        for i, (x, y) in enumerate(self.game.snake2):
            if self.game.game_over2:
                # Dead - draw in dark gray
                if i == 0:
                    painter.setBrush(QColor(60, 60, 60))  # Dark gray head
                else:
                    painter.setBrush(QColor(40, 40, 40))  # Darker gray body
            else:
                # Alive - draw in blue
                if i == 0:
                    painter.setBrush(QColor(0, 150, 255))  # Head - bright blue
                else:
                    painter.setBrush(QColor(0, 100, 200))  # Body - darker blue
            painter.drawRect(x * self.cell_size + 1, y * self.cell_size + 1,
                           self.cell_size - 2, self.cell_size - 2)
        
        # Draw game over
        if self.game.is_game_over():
            painter.setPen(QColor(255, 255, 255))
            painter.setFont(QFont('Arial', 30, QFont.Bold))
            
            winner = self.game.get_winner()
            if winner == 1:
                text = "SNAKE 1 (GREEN) WINS!"
                color = QColor(0, 255, 0)
            elif winner == 2:
                text = "SNAKE 2 (BLUE) WINS!"
                color = QColor(0, 150, 255)
            else:
                text = "DRAW!"
                color = QColor(255, 255, 0)
            
            painter.setPen(color)
            painter.drawText(self.rect(), Qt.AlignCenter, text)
