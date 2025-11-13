import random
from enum import Enum

class Direction(Enum):
    UP = (0, -1)
    DOWN = (0, 1)
    LEFT = (-1, 0)
    RIGHT = (1, 0)

class SnakeGameMultiplayer:
    def __init__(self, grid_size=30):
        self.grid_size = grid_size
        self.reset()
    
    def reset(self):
        """Reset game with two snakes"""
        # Place snakes on opposite sides
        self.snake1 = [(self.grid_size // 4, self.grid_size // 2)]
        self.snake2 = [(3 * self.grid_size // 4, self.grid_size // 2)]
        
        self.direction1 = Direction.RIGHT
        self.direction2 = Direction.LEFT
        
        self.apples = []
        self.score1 = 0
        self.score2 = 0
        self.game_over1 = False
        self.game_over2 = False
        
        self.spawn_apples(3)
    
    def spawn_apples(self, count):
        """Spawn apples avoiding both snakes"""
        spawned = 0
        max_attempts = 10000
        attempts = 0
        
        while spawned < count and attempts < max_attempts:
            apple = (random.randint(0, self.grid_size - 1), 
                    random.randint(0, self.grid_size - 1))
            
            if (apple not in self.snake1 and 
                apple not in self.snake2 and 
                apple not in self.apples):
                self.apples.append(apple)
                spawned += 1
            
            attempts += 1
    
    def set_direction(self, player, direction):
        """Set direction for a player"""
        if player == 1:
            current = self.direction1

            # Prevent 180-degree turns
            blocked = False
            if (current == Direction.UP and direction == Direction.DOWN):
                blocked = True
            if (current == Direction.DOWN and direction == Direction.UP):
                blocked = True
            if (current == Direction.LEFT and direction == Direction.RIGHT):
                blocked = True
            if (current == Direction.RIGHT and direction == Direction.LEFT):
                blocked = True
            if not blocked:
                self.direction1 = direction
        else:
            current = self.direction2

            # Prevent 180-degree turns
            blocked = False
            if (current == Direction.UP and direction == Direction.DOWN):
                blocked = True
            if (current == Direction.DOWN and direction == Direction.UP):
                blocked = True
            if (current == Direction.LEFT and direction == Direction.RIGHT):
                blocked = True
            if (current == Direction.RIGHT and direction == Direction.LEFT):
                blocked = True
            if not blocked:
                self.direction2 = direction
    
    def step(self):
        """Execute one game step for both snakes"""
        if self.game_over1 and self.game_over2:
            return
        
        # Move snake 1
        if not self.game_over1:
            head_x, head_y = self.snake1[0]
            dx, dy = self.direction1.value
            new_head1 = ((head_x + dx) % self.grid_size, (head_y + dy) % self.grid_size)
            
            # Check collision with self
            if new_head1 in self.snake1:
                self.game_over1 = True
            
            # Check collision with snake2
            if new_head1 in self.snake2:
                self.game_over1 = True
            
            if not self.game_over1:
                self.snake1.insert(0, new_head1)
                
                # Check if apple eaten
                if new_head1 in self.apples:
                    self.apples.remove(new_head1)
                    self.score1 += 1
                    self.spawn_apples(1)
                else:
                    self.snake1.pop()
        
        # Move snake 2
        if not self.game_over2:
            head_x, head_y = self.snake2[0]
            dx, dy = self.direction2.value
            new_head2 = ((head_x + dx) % self.grid_size, (head_y + dy) % self.grid_size)
            
            # Check collision with self
            if new_head2 in self.snake2:
                self.game_over2 = True
            
            # Check collision with snake1 (including head-to-head)
            if new_head2 in self.snake1:
                self.game_over2 = True
            
            if not self.game_over2:
                self.snake2.insert(0, new_head2)
                
                # Check if apple eaten
                if new_head2 in self.apples:
                    self.apples.remove(new_head2)
                    self.score2 += 1
                    self.spawn_apples(1)
                else:
                    self.snake2.pop()
    
    def get_state(self, player):
        """Get game state for a specific player"""
        if player == 1:
            return {
                'snake': self.snake1.copy(),
                'direction': self.direction1,
                'apples': self.apples.copy(),
                'grid_size': self.grid_size,
                'score': self.score1,
                'game_over': self.game_over1
            }
        else:
            return {
                'snake': self.snake2.copy(),
                'direction': self.direction2,
                'apples': self.apples.copy(),
                'grid_size': self.grid_size,
                'score': self.score2,
                'game_over': self.game_over2
            }
    
    def is_game_over(self):
        """Check if game is over for both players"""
        return self.game_over1 or self.game_over2
    
    def get_winner(self):
        """Get winner (1, 2, or 0 for draw)"""
        if not self.game_over1 and self.game_over2:
            return 1
        elif self.game_over1 and not self.game_over2:
            return 2
        elif self.game_over1 and self.game_over2:
            return 0  # Draw
        else:
            return None  # Game not over