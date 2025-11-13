import torch
import numpy as np
from snake_model import SnakeNet
import sys
import os

# Ensure snake_game module is importable
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from snake_game import Direction

class AgentMiddleware:
    def __init__(self, model_path=None):
        self.model = SnakeNet(input_size=31, hidden_size=128, output_size=3)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        
        if model_path:
            self.load_model(model_path)
        
        # Map relative actions to actual directions
        # Actions: 0=forward, 1=left_turn, 2=right_turn
        self.action_map = {
            'UP':    [Direction.UP,    Direction.LEFT,  Direction.RIGHT],
            'DOWN':  [Direction.DOWN,  Direction.RIGHT, Direction.LEFT],
            'LEFT':  [Direction.LEFT,  Direction.DOWN,  Direction.UP],
            'RIGHT': [Direction.RIGHT, Direction.UP,    Direction.DOWN]
        }
    
    def load_model(self, path):
        """Load trained model weights"""
        self.model.load_state_dict(torch.load(path, map_location=self.device))
        self.model.eval()
    
    def save_model(self, path):
        """Save model weights"""
        torch.save(self.model.state_dict(), path)
    
    def get_rotated_grid(self, snake, direction, grid_size, apples):
        """
        Get 5x5 grid centered on snake head, rotated so current direction is UP
        Returns grid where: 0=empty, -1=snake body, 1=apple
        """
        head_x, head_y = snake[0]
        grid = np.zeros((5, 5), dtype=np.float32)
        
        # Rotation mapping: how many 90° clockwise rotations needed to make direction point UP
        rotation_map = {
            'UP': 0,
            'RIGHT': 1,
            'DOWN': 2,
            'LEFT': 3
        }
        
        rotations = rotation_map[direction.name]
        
        # Get 5x5 area around head
        for dy in range(-2, 3):
            for dx in range(-2, 3):
                # Apply rotation before translation
                if rotations == 0:
                    rx, ry = dx, dy
                elif rotations == 1:
                    rx, ry = -dy, dx
                elif rotations == 2:
                    rx, ry = -dx, -dy
                else:  # rotations == 3
                    rx, ry = dy, -dx
                
                # Translate to world coordinates (with wraparound)
                world_x = (head_x + rx) % grid_size
                world_y = (head_y + ry) % grid_size
                
                # Grid coordinates (after rotation, center is always at 2,2)
                grid_x = dx + 2
                grid_y = dy + 2
                
                # Check if position has snake body
                if (world_x, world_y) in snake[1:]:  # Exclude head
                    grid[grid_y][grid_x] = -1.0
                
                # Check if position has apple
                elif (world_x, world_y) in apples:
                    grid[grid_y][grid_x] = 1.0
        
        return grid.flatten()
    
    def get_apple_directions(self, head, apples, direction, grid_size):
        """
        Get direction vectors to 3 apples, rotated so current direction is UP
        Returns flattened array of [dx1, dy1, dx2, dy2, dx3, dy3]
        All normalized by grid_size
        """
        head_x, head_y = head
        
        # Rotation angles
        rotation_map = {
            'UP': 0,
            'RIGHT': -90,
            'DOWN': 180,
            'LEFT': 90
        }
        
        angle = np.radians(rotation_map[direction.name])
        cos_a = np.cos(angle)
        sin_a = np.sin(angle)
        
        apple_vectors = []
        
        for apple in apples:
            apple_x, apple_y = apple
            
            # Calculate shortest distance considering wraparound
            dx = apple_x - head_x
            dy = apple_y - head_y
            
            # Adjust for wraparound
            if abs(dx) > grid_size / 2:
                dx = dx - np.sign(dx) * grid_size
            if abs(dy) > grid_size / 2:
                dy = dy - np.sign(dy) * grid_size
            
            # Rotate vector
            rotated_dx = dx * cos_a - dy * sin_a
            rotated_dy = dx * sin_a + dy * cos_a
            
            # Normalize by grid size
            rotated_dx /= grid_size
            rotated_dy /= grid_size
            
            apple_vectors.extend([rotated_dx, rotated_dy])
        
        # Ensure exactly 3 apples (pad with zeros if needed)
        while len(apple_vectors) < 6:
            apple_vectors.extend([0.0, 0.0])
        
        return np.array(apple_vectors[:6], dtype=np.float32)
    
    def get_observation(self, game_state):
        """
        Convert game state to neural network input
        Returns: numpy array of size 31 (25 grid + 6 apple directions)
        """
        snake = game_state['snake']
        direction = game_state['direction']
        apples = game_state['apples']
        grid_size = game_state['grid_size']
        
        # Get 5x5 rotated grid
        grid = self.get_rotated_grid(snake, direction, grid_size, apples)
        
        # Get apple direction vectors
        apple_dirs = self.get_apple_directions(snake[0], apples, direction, grid_size)
        
        # Concatenate
        observation = np.concatenate([grid, apple_dirs])
        
        return observation
    
    def get_action(self, game_state, epsilon=0.0, debug=False):
        """
        Get action from model
        epsilon: exploration rate (0 = always exploit, 1 = always explore)
        Returns: Direction enum
        """
        current_direction = game_state['direction']
        
        # Epsilon-greedy exploration
        if np.random.random() < epsilon:
            # Random action (0=forward, 1=left, 2=right)
            action_idx = np.random.randint(0, 3)
        else:
            # Get observation
            obs = self.get_observation(game_state)
            
            if debug:
                print(f"Grid (5x5):\n{obs[:25].reshape(5,5)}")
                print(f"Apple vectors: {obs[25:]}")
            
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
            
            # Forward pass
            with torch.no_grad():
                q_values = self.model(obs_tensor)
            
            # Get best action
            action_idx = q_values.argmax().item()
            
            if debug:
                print(f"Q-values: {q_values[0].cpu().numpy()}, Chosen action: {action_idx}")
        
        # Convert relative action to absolute direction
        # 0 = forward, 1 = left turn, 2 = right turn
        chosen_direction = self.action_map[current_direction.name][action_idx]
        
        if debug:
            action_names = ['FORWARD', 'LEFT', 'RIGHT']
            print(f"Current: {current_direction.name}, Action: {action_names[action_idx]}, New: {chosen_direction.name}")
        
        return chosen_direction