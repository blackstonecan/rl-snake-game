import torch
import numpy as np
from snake_model_large import SnakeNetLarge
import sys
import os
from enum import Enum

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

class Direction(Enum):
    UP = (0, -1)
    DOWN = (0, 1)
    LEFT = (-1, 0)
    RIGHT = (1, 0)

class AgentMiddlewareLarge:
    def __init__(self, model_path=None):
        self.model = SnakeNetLarge(input_size=57, hidden_size=128, output_size=3)
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
    
    def get_rotated_grid(self, snake, direction, grid_size, apples, opponent_snake):
        """
        Get 7x7 grid centered on snake head, rotated so current direction is UP
        Returns grid where: 0=empty, -1=your body, -2=opponent body, -3=opponent head, 1=apple
        """
        head_x, head_y = snake[0]
        grid = np.zeros((7, 7), dtype=np.float32)
        
        # Rotation mapping
        rotation_map = {
            'UP': 0,
            'RIGHT': 1,
            'DOWN': 2,
            'LEFT': 3
        }
        
        rotations = rotation_map[direction.name]
        
        # Get 7x7 area around head
        for dy in range(-3, 4):
            for dx in range(-3, 4):
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
                
                # Grid coordinates (after rotation, center is always at 3,3)
                grid_x = dx + 3
                grid_y = dy + 3
                
                # Check if position has your snake body
                if (world_x, world_y) in snake[1:]:  # Exclude head
                    grid[grid_y][grid_x] = -1.0
                
                # Check if position has opponent head
                elif opponent_snake and len(opponent_snake) > 0 and (world_x, world_y) == opponent_snake[0]:
                    grid[grid_y][grid_x] = -3.0
                
                # Check if position has opponent body
                elif opponent_snake and (world_x, world_y) in opponent_snake[1:]:
                    grid[grid_y][grid_x] = -2.0
                
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
    
    def get_opponent_head_direction(self, head, opponent_head, direction, grid_size):
        """
        Get direction vector to opponent head, rotated so current direction is UP
        Returns [dx, dy] normalized by grid_size
        """
        if not opponent_head:
            return np.array([0.0, 0.0], dtype=np.float32)
        
        head_x, head_y = head
        opp_x, opp_y = opponent_head
        
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
        
        # Calculate shortest distance considering wraparound
        dx = opp_x - head_x
        dy = opp_y - head_y
        
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
        
        return np.array([rotated_dx, rotated_dy], dtype=np.float32)
    
    def get_observation(self, game_state, opponent_state=None):
        """
        Convert game state to neural network input
        Returns: numpy array of size 57 (49 grid + 6 apple directions + 2 opponent head)
        """
        snake = game_state['snake']
        direction = game_state['direction']
        apples = game_state['apples']
        grid_size = game_state['grid_size']
        
        opponent_snake = opponent_state['snake'] if opponent_state else None
        
        # Get 7x7 rotated grid
        grid = self.get_rotated_grid(snake, direction, grid_size, apples, opponent_snake)
        
        # Get apple direction vectors
        apple_dirs = self.get_apple_directions(snake[0], apples, direction, grid_size)
        
        # Get opponent head direction
        opponent_head = opponent_snake[0] if opponent_snake else None
        opponent_head_dir = self.get_opponent_head_direction(snake[0], opponent_head, direction, grid_size)
        
        # Concatenate
        observation = np.concatenate([grid, apple_dirs, opponent_head_dir])
        
        return observation
    
    def get_action(self, game_state, opponent_state=None, epsilon=0.0, debug=False):
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
            obs = self.get_observation(game_state, opponent_state)
            
            if debug:
                print(f"Grid (7x7):\n{obs[:49].reshape(7,7)}")
                print(f"Apple vectors: {obs[49:55]}")
                print(f"Opponent head: {obs[55:]}")
            
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