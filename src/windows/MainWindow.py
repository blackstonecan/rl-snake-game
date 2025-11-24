import torch
from PyQt5.QtWidgets import (QMainWindow, QWidget, QPushButton, 
                             QVBoxLayout, QHBoxLayout, QComboBox, QLabel)
from PyQt5.QtCore import QTimer, Qt
from snake.snake_game_multiplayer import SnakeGameMultiplayer
from agents.agent_middleware_large import AgentMiddlewareLarge
from components.GameWidget import GameWidget


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Snake Game 2-Player Battle")
        
        self.game = SnakeGameMultiplayer(30)
        self.game_speed = 100  # milliseconds
        
        # Agent slots
        self.agent1 = None
        self.agent2 = None
        self.agent1_type = None  # '5x5' or '7x7'
        self.agent2_type = None
        
        self.init_ui()
        
        # Game timer
        self.timer = QTimer()
        self.timer.timeout.connect(self.game_loop)
    
    def init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        main_layout = QVBoxLayout()
        
        # Game widget
        self.game_widget = GameWidget(self.game)
        main_layout.addWidget(self.game_widget, alignment=Qt.AlignCenter)
        
        # Model selection
        model_layout = QHBoxLayout()
        
        # Snake 1 selection
        snake1_layout = QVBoxLayout()
        snake1_label = QLabel("Snake 1 (Green):")
        snake1_label.setStyleSheet("color: #00FF00; font-weight: bold;")
        self.snake1_combo = QComboBox()
        self.populate_model_combo(self.snake1_combo)
        snake1_layout.addWidget(snake1_label)
        snake1_layout.addWidget(self.snake1_combo)
        model_layout.addLayout(snake1_layout)
        
        # Snake 2 selection
        snake2_layout = QVBoxLayout()
        snake2_label = QLabel("Snake 2 (Blue):")
        snake2_label.setStyleSheet("color: #0096FF; font-weight: bold;")
        self.snake2_combo = QComboBox()
        self.populate_model_combo(self.snake2_combo)
        if self.snake2_combo.count() > 1:
            self.snake2_combo.setCurrentIndex(1)  # Select different by default
        snake2_layout.addWidget(snake2_label)
        snake2_layout.addWidget(self.snake2_combo)
        model_layout.addLayout(snake2_layout)
        
        main_layout.addLayout(model_layout)
        
        # Model control buttons
        model_button_layout = QHBoxLayout()
        
        load_btn = QPushButton("Load Models")
        load_btn.clicked.connect(self.load_models)
        model_button_layout.addWidget(load_btn)
        
        refresh_btn = QPushButton("Refresh Model List")
        refresh_btn.clicked.connect(self.refresh_models)
        model_button_layout.addWidget(refresh_btn)
        
        main_layout.addLayout(model_button_layout)
        
        # Control buttons
        control_layout = QHBoxLayout()
        
        self.start_btn = QPushButton("Start")
        self.start_btn.clicked.connect(self.start_game)
        control_layout.addWidget(self.start_btn)
        
        self.pause_btn = QPushButton("Pause")
        self.pause_btn.clicked.connect(self.pause_game)
        self.pause_btn.setEnabled(False)
        control_layout.addWidget(self.pause_btn)
        
        self.reset_btn = QPushButton("Reset")
        self.reset_btn.clicked.connect(self.reset_game)
        control_layout.addWidget(self.reset_btn)
        
        main_layout.addLayout(control_layout)
        
        # Speed control
        speed_layout = QHBoxLayout()
        speed_label = QLabel("Speed:")
        self.speed_combo = QComboBox()
        self.speed_combo.addItems(["Slow (200ms)", "Normal (100ms)", "Fast (50ms)", "Very Fast (25ms)"])
        self.speed_combo.setCurrentIndex(1)
        self.speed_combo.currentIndexChanged.connect(self.change_speed)
        speed_layout.addWidget(speed_label)
        speed_layout.addWidget(self.speed_combo)
        main_layout.addLayout(speed_layout)
        
        # Score labels
        self.score_label = QLabel()
        self.update_score_label()
        main_layout.addWidget(self.score_label)
        
        # Status label
        self.status_label = QLabel("Select models and click 'Load Models' to begin")
        self.status_label.setAlignment(Qt.AlignCenter)
        self.status_label.setStyleSheet("color: #888888; font-style: italic;")
        main_layout.addWidget(self.status_label)
        
        central_widget.setLayout(main_layout)
    
    def populate_model_combo(self, combo):
        """Add available models to combo box - scan directory for .pth files"""
        combo.addItem("Random Agent", None)
        
        # Scan current directory for .pth files
        import glob
        pth_files = glob.glob("*.pth")
        
        if not pth_files:
            print("No .pth files found in current directory")
            return
        
        # Sort files for consistent ordering
        pth_files.sort()
        
        for pth_file in pth_files:
            # Try to determine if 5x5 or 7x7 based on filename patterns
            display_name = pth_file
            
            combo.addItem(display_name, pth_file)
    
    def detect_agent_type(self, pth_file):
        """Try to detect if a model is 5x5 or 7x7 by loading it"""
        try:
            state_dict = torch.load(pth_file, map_location='cpu')
            # Check input layer size
            if 'fc1.weight' in state_dict:
                input_size = state_dict['fc1.weight'].shape[1]
                if input_size == 31:
                    return '5x5'
                elif input_size == 57:
                    return '7x7'
            return '7x7'  # Default to 7x7 if unsure
        except:
            return '7x7'  # Default to 7x7 on error
    
    def refresh_models(self):
        """Refresh the model list from directory"""
        print("Refreshing model list...")
        
        # Save current selections
        snake1_current = self.snake1_combo.currentText()
        snake2_current = self.snake2_combo.currentText()
        
        # Clear and repopulate
        self.snake1_combo.clear()
        self.snake2_combo.clear()
        self.populate_model_combo(self.snake1_combo)
        self.populate_model_combo(self.snake2_combo)
        
        # Try to restore selections
        snake1_idx = self.snake1_combo.findText(snake1_current)
        if snake1_idx >= 0:
            self.snake1_combo.setCurrentIndex(snake1_idx)
        
        snake2_idx = self.snake2_combo.findText(snake2_current)
        if snake2_idx >= 0:
            self.snake2_combo.setCurrentIndex(snake2_idx)
        elif self.snake2_combo.count() > 1:
            self.snake2_combo.setCurrentIndex(1)
        
        print(f"Found {self.snake1_combo.count() - 1} models")  # -1 for "Random Agent"
    
    def load_models(self):
        """Load selected models"""
        loaded = []
        
        # Load agent 1
        path = self.snake1_combo.currentData()
        if path:
            try:
                self.agent1 = AgentMiddlewareLarge(path)
                loaded.append(f"Snake 1: {self.snake1_combo.currentText()}")
                print(f"✓ Loaded Snake 1: {self.snake1_combo.currentText()}")
            except Exception as e:
                print(f"✗ Failed to load Snake 1: {e}")
                self.agent1 = None
                loaded.append(f"Snake 1: Random (load failed)")
        else:
            self.agent1 = None
            loaded.append(f"Snake 1: Random")
        
        # Load agent 2
        path = self.snake2_combo.currentData()
        if path:
            try:
                self.agent2 = AgentMiddlewareLarge(path)
                loaded.append(f"Snake 2: {self.snake2_combo.currentText()}")
                print(f"✓ Loaded Snake 2: {self.snake2_combo.currentText()}")
            except Exception as e:
                print(f"✗ Failed to load Snake 2: {e}")
                self.agent2 = None
                loaded.append(f"Snake 2: Random (load failed)")
        else:
            self.agent2 = None
            loaded.append(f"Snake 2: Random")
        
        # Update status
        self.status_label.setText(" | ".join(loaded))
        self.status_label.setStyleSheet("color: #00FF00; font-weight: bold;")
    
    def update_score_label(self):
        """Update score display"""
        text = f"🟢 Snake 1: {self.game.score1}  |  🔵 Snake 2: {self.game.score2}"
        
        if self.game.game_over1:
            text += "  |  🟢 DEAD"
        if self.game.game_over2:
            text += "  |  🔵 DEAD"
        
        self.score_label.setText(text)
        self.score_label.setAlignment(Qt.AlignCenter)
        self.score_label.setStyleSheet("font-size: 16px; font-weight: bold;")
    
    def start_game(self):
        """Start the game"""
        if not self.timer.isActive():
            self.timer.start(self.game_speed)
            self.start_btn.setEnabled(False)
            self.pause_btn.setEnabled(True)
    
    def pause_game(self):
        """Pause the game"""
        self.timer.stop()
        self.start_btn.setEnabled(True)
        self.pause_btn.setEnabled(False)
    
    def reset_game(self):
        """Reset the game"""
        self.timer.stop()
        self.game.reset()
        self.game_widget.update()
        self.update_score_label()
        self.start_btn.setEnabled(True)
        self.pause_btn.setEnabled(False)
    
    def change_speed(self):
        """Change game speed"""
        speeds = [200, 100, 50, 25]
        self.game_speed = speeds[self.speed_combo.currentIndex()]
        if self.timer.isActive():
            self.timer.setInterval(self.game_speed)
    
    def game_loop(self):
        """Main game loop"""
        if self.game.is_game_over():
            self.timer.stop()
            self.game_widget.update()
            self.update_score_label()
            self.start_btn.setEnabled(True)
            self.pause_btn.setEnabled(False)
            return
        
        # Get states
        state1 = self.game.get_state(1)
        state2 = self.game.get_state(2)
        
        # Agent 1 action (only if alive)
        if not state1['game_over']:
            if self.agent1:
                try:
                    direction1 = self.agent1.get_action(state1, state2, epsilon=0.0)
                    self.game.set_direction(1, direction1)
                except Exception as e:
                    print(f"Error in agent 1: {e}")
            # Random movement if no agent - let game handle it
        
        # Agent 2 action (only if alive)
        if not state2['game_over']:
            if self.agent2:
                try:
                    direction2 = self.agent2.get_action(state2, state1, epsilon=0.0)
                    self.game.set_direction(2, direction2)
                except Exception as e:
                    print(f"Error in agent 2: {e}")
        
        # Execute game step
        self.game.step()
        
        # Update display
        self.game_widget.update()
        self.update_score_label()

