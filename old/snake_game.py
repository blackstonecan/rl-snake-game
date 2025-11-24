import sys
import random
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QPushButton, QVBoxLayout, QHBoxLayout
from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtGui import QPainter, QColor, QFont
from enum import Enum
import os

model_path = "killer_agent.pth"

large_models = ["main_agent.pth", "killer_agent.pth"]
mini_models = ["snake_agent.pth"]

try:
    from agent_middleware import AgentMiddleware
    from front.agents.agent_middleware_large import AgentMiddlewareLarge
    AGENT_AVAILABLE = True
except ImportError:
    AGENT_AVAILABLE = False
    print("Warning: agent_middleware not found. AI mode will use simple logic.")

class Direction(Enum):
    UP = (0, -1)
    DOWN = (0, 1)
    LEFT = (-1, 0)
    RIGHT = (1, 0)

class GameMode(Enum):
    HUMAN = "Human Play"
    AI = "AI Play"

class SnakeGame:
    def __init__(self, grid_size=30):
        self.grid_size = grid_size
        self.reset()
    
    def reset(self):
        self.snake = [(self.grid_size // 2, self.grid_size // 2)]
        self.direction = Direction.RIGHT
        self.apples = []
        self.score = 0
        self.game_over = False
        self.spawn_apples(3)
    
    def spawn_apples(self, count):
        spawned = 0
        while spawned < count:
            apple = (random.randint(0, self.grid_size - 1), 
                    random.randint(0, self.grid_size - 1))
            if apple not in self.snake and apple not in self.apples:
                self.apples.append(apple)
                spawned += 1
    
    def set_direction(self, direction, debug=False):
        # Prevent 180-degree turns
        if debug:
            print(f"[set_direction] Current: {self.direction.name}, Requested: {direction.name}")


        if (self.direction == Direction.UP and direction == Direction.DOWN):
            if debug:
                print(f"[set_direction] ✗ Direction blocked (180° turn) FROM UP to DOWN")
        elif (self.direction == Direction.DOWN and direction == Direction.UP):
            if debug:
                print(f"[set_direction] ✗ Direction blocked (180° turn) FROM DOWN to UP")
        elif (self.direction == Direction.LEFT and direction == Direction.RIGHT):
            if debug:
                print(f"[set_direction] ✗ Direction blocked (180° turn) FROM LEFT to RIGHT")
        elif (self.direction == Direction.RIGHT and direction == Direction.LEFT):
            if debug:
                print(f"[set_direction] ✗ Direction blocked (180° turn) FROM RIGHT to LEFT")
        else:
            self.direction = direction
            if debug:
                print(f"[set_direction] ✓ Direction changed to: {self.direction.name}")
    
    def step(self):
        if self.game_over:
            return
        
        # Calculate new head position
        head_x, head_y = self.snake[0]
        dx, dy = self.direction.value
        new_head = ((head_x + dx) % self.grid_size, (head_y + dy) % self.grid_size)
        
        # Check collision with self
        if new_head in self.snake:
            self.game_over = True
            return
        
        # Add new head
        self.snake.insert(0, new_head)
        
        # Check if apple eaten
        if new_head in self.apples:
            self.apples.remove(new_head)
            self.score += 1
            self.spawn_apples(1)
        else:
            # Remove tail if no apple eaten
            self.snake.pop()
    
    def get_state(self):
        """Returns game state for AI agent"""
        return {
            'snake': self.snake.copy(),
            'direction': self.direction,
            'apples': self.apples.copy(),
            'grid_size': self.grid_size,
            'score': self.score,
            'game_over': self.game_over
        }

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
        
        # Draw snake
        for i, (x, y) in enumerate(self.game.snake):
            if i == 0:
                painter.setBrush(QColor(0, 255, 0))  # Head
            else:
                painter.setBrush(QColor(0, 200, 0))  # Body
            painter.drawRect(x * self.cell_size + 1, y * self.cell_size + 1,
                           self.cell_size - 2, self.cell_size - 2)
        
        # Draw game over
        if self.game.game_over:
            painter.setPen(QColor(255, 255, 255))
            painter.setFont(QFont('Arial', 30, QFont.Bold))
            painter.drawText(self.rect(), Qt.AlignCenter, "GAME OVER")

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Snake Game - PyQt5")
        
        self.game = SnakeGame(30)
        self.mode = GameMode.HUMAN
        self.game_speed = 100  # milliseconds
        
        # Initialize agent middleware
        self.agent = None
        if AGENT_AVAILABLE:
            if os.path.exists(model_path):
                if model_path in large_models:
                    self.agent = AgentMiddlewareLarge(model_path)
                elif model_path in mini_models:
                    self.agent = AgentMiddleware(model_path)
                else:
                    raise ValueError("Model path not recognized in large or mini model lists.")

                print(f"Loaded trained agent from {model_path}")
            else:
                print("No trained model found. Train with train_agent.py first.")
        
        self.init_ui()
        
        # Game timer
        self.timer = QTimer()
        self.timer.timeout.connect(self.game_loop)
    
    def init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        layout = QVBoxLayout()
        
        # Game widget
        self.game_widget = GameWidget(self.game)
        layout.addWidget(self.game_widget, alignment=Qt.AlignCenter)
        
        # Controls
        control_layout = QHBoxLayout()
        
        self.human_btn = QPushButton("Human Play")
        self.human_btn.clicked.connect(self.set_human_mode)
        control_layout.addWidget(self.human_btn)
        
        self.ai_btn = QPushButton("AI Play")
        self.ai_btn.clicked.connect(self.set_ai_mode)
        control_layout.addWidget(self.ai_btn)
        
        self.start_btn = QPushButton("Start")
        self.start_btn.clicked.connect(self.start_game)
        control_layout.addWidget(self.start_btn)
        
        self.reset_btn = QPushButton("Reset")
        self.reset_btn.clicked.connect(self.reset_game)
        control_layout.addWidget(self.reset_btn)
        
        layout.addLayout(control_layout)
        
        # Score label
        self.score_label = QPushButton(f"Score: 0 | Mode: {self.mode.value}")
        self.score_label.setEnabled(False)
        layout.addWidget(self.score_label)
        
        central_widget.setLayout(layout)
        self.update_buttons()
    
    def set_human_mode(self):
        self.mode = GameMode.HUMAN
        self.update_buttons()
    
    def set_ai_mode(self):
        self.mode = GameMode.AI
        self.update_buttons()
    
    def update_buttons(self):
        self.score_label.setText(f"Score: {self.game.score} | Mode: {self.mode.value}")
        self.human_btn.setStyleSheet("background-color: lightblue" if self.mode == GameMode.HUMAN else "")
        self.ai_btn.setStyleSheet("background-color: lightblue" if self.mode == GameMode.AI else "")
    
    def start_game(self):
        if not self.timer.isActive():
            self.timer.start(self.game_speed)
    
    def reset_game(self):
        self.timer.stop()
        self.game.reset()
        self.game_widget.update()
        self.update_buttons()
    
    def game_loop(self):
        if self.game.game_over:
            self.timer.stop()
            self.game_widget.update()
            return
        
        if self.mode == GameMode.AI:
            self.ai_move()
        
        self.game.step()
        self.game_widget.update()
        self.update_buttons()
    
    def ai_move(self):
        """AI logic - uses trained agent if available, otherwise simple pathfinding"""
        if self.agent is not None:
            # Use trained agent
            game_state = self.game.get_state()
            direction = self.agent.get_action(game_state, epsilon=0.0, debug=False)
            self.game.set_direction(direction, debug=False)
        else:
            # Fallback to simple AI
            head_x, head_y = self.game.snake[0]
            
            # Find closest apple
            closest_apple = min(self.game.apples, 
                              key=lambda a: abs(a[0] - head_x) + abs(a[1] - head_y))
            
            apple_x, apple_y = closest_apple
            
            # Simple logic: move towards apple
            dx = (apple_x - head_x) % self.game.grid_size
            dy = (apple_y - head_y) % self.game.grid_size
            
            if dx > self.game.grid_size // 2:
                dx -= self.game.grid_size
            if dy > self.game.grid_size // 2:
                dy -= self.game.grid_size
            
            if abs(dx) > abs(dy):
                if dx > 0:
                    self.game.set_direction(Direction.RIGHT)
                else:
                    self.game.set_direction(Direction.LEFT)
            else:
                if dy > 0:
                    self.game.set_direction(Direction.DOWN)
                else:
                    self.game.set_direction(Direction.UP)
    
    def keyPressEvent(self, event):
        if self.mode == GameMode.HUMAN and not self.game.game_over:
            key = event.key()
            if key == Qt.Key_Up or key == Qt.Key_W:
                self.game.set_direction(Direction.UP)
            elif key == Qt.Key_Down or key == Qt.Key_S:
                self.game.set_direction(Direction.DOWN)
            elif key == Qt.Key_Left or key == Qt.Key_A:
                self.game.set_direction(Direction.LEFT)
            elif key == Qt.Key_Right or key == Qt.Key_D:
                self.game.set_direction(Direction.RIGHT)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())