import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random
from snake_game import SnakeGame, Direction
from agent_middleware import AgentMiddleware

class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (np.array(states), np.array(actions), np.array(rewards), 
                np.array(next_states), np.array(dones))
    
    def __len__(self):
        return len(self.buffer)

class SnakeTrainer:
    def __init__(self, grid_size=30):
        self.grid_size = grid_size
        self.game = SnakeGame(grid_size)
        self.agent = AgentMiddleware()
        
        self.device = self.agent.device
        self.optimizer = optim.Adam(self.agent.model.parameters(), lr=0.001)
        self.criterion = nn.MSELoss()
        
        self.replay_buffer = ReplayBuffer(capacity=10000)
        self.batch_size = 64
        self.gamma = 0.95  # Discount factor
        
        # Training parameters
        self.epsilon_start = 1.0
        self.epsilon_end = 0.01
        self.epsilon_decay = 0.995
        self.epsilon = self.epsilon_start
        
        # Statistics
        self.episode_rewards = []
        self.episode_scores = []
    
    def calculate_reward(self, prev_state, current_state, action):
        """Calculate reward for the current step"""
        reward = 0
        
        # Check if game over
        if current_state['game_over']:
            return -10
        
        # Reward for eating apple
        if current_state['score'] > prev_state['score']:
            reward += 10
        
        # Small penalty for each step (encourages efficiency)
        reward -= 0.01
        
        # Optional: reward for moving closer to nearest apple
        prev_head = prev_state['snake'][0]
        curr_head = current_state['snake'][0]
        
        # Find closest apple
        def min_distance(pos, apples):
            if not apples:
                return 0
            distances = []
            for apple in apples:
                dx = abs(pos[0] - apple[0])
                dy = abs(pos[1] - apple[1])
                # Consider wraparound
                dx = min(dx, self.grid_size - dx)
                dy = min(dy, self.grid_size - dy)
                distances.append(dx + dy)
            return min(distances)
        
        prev_dist = min_distance(prev_head, prev_state['apples'])
        curr_dist = min_distance(curr_head, current_state['apples'])
        
        if curr_dist < prev_dist:
            reward += 0.1
        
        return reward
    
    def train_step(self):
        """Perform one training step"""
        if len(self.replay_buffer) < self.batch_size:
            return 0
        
        # Sample from replay buffer
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)
        
        # Convert to tensors
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        
        # Current Q values
        current_q_values = self.agent.model(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # Next Q values
        with torch.no_grad():
            next_q_values = self.agent.model(next_states).max(1)[0]
            target_q_values = rewards + (1 - dones) * self.gamma * next_q_values
        
        # Compute loss
        loss = self.criterion(current_q_values, target_q_values)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
    
    def direction_to_action_idx(self, current_dir, chosen_dir):
        """Convert direction to action index (0-2)"""
        # 0=forward, 1=left, 2=right
        action_map = {
            'UP':    {'UP': 0,    'LEFT': 1,  'RIGHT': 2},
            'DOWN':  {'DOWN': 0,  'RIGHT': 1, 'LEFT': 2},
            'LEFT':  {'LEFT': 0,  'DOWN': 1,  'UP': 2},
            'RIGHT': {'RIGHT': 0, 'UP': 1,    'DOWN': 2}
        }
        return action_map[current_dir.name][chosen_dir.name]
    
    def train_episode(self):
        """Train for one episode"""
        self.game.reset()
        episode_reward = 0
        step_count = 0
        max_steps = 1000  # Prevent infinite loops
        
        while not self.game.game_over and step_count < max_steps:
            # Get current state
            prev_state = self.game.get_state()
            prev_obs = self.agent.get_observation(prev_state)
            
            # Get action
            action_dir = self.agent.get_action(prev_state, epsilon=self.epsilon)
            self.game.set_direction(action_dir)
            
            # Take step
            self.game.step()
            
            # Get new state
            curr_state = self.game.get_state()
            curr_obs = self.agent.get_observation(curr_state)
            
            # Calculate reward
            reward = self.calculate_reward(prev_state, curr_state, action_dir)
            episode_reward += reward
            
            # Convert action to index
            action_idx = self.direction_to_action_idx(prev_state['direction'], action_dir)
            
            # Store in replay buffer
            self.replay_buffer.push(prev_obs, action_idx, reward, curr_obs, 
                                   1.0 if curr_state['game_over'] else 0.0)
            
            # Train
            loss = self.train_step()
            
            step_count += 1
        
        # Decay epsilon
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
        
        # Store statistics
        self.episode_rewards.append(episode_reward)
        self.episode_scores.append(self.game.score)
        
        return episode_reward, self.game.score
    
    def train(self, num_episodes=1000, save_path='snake_agent.pth', save_interval=100):
        """Train the agent"""
        print(f"Training on device: {self.device}")
        print(f"Starting training for {num_episodes} episodes...\n")
        
        best_score = 0
        
        for episode in range(1, num_episodes + 1):
            reward, score = self.train_episode()
            
            # Print progress
            if episode % 10 == 0:
                avg_reward = np.mean(self.episode_rewards[-10:])
                avg_score = np.mean(self.episode_scores[-10:])
                print(f"Episode {episode}/{num_episodes} | "
                      f"Epsilon: {self.epsilon:.3f} | "
                      f"Avg Reward: {avg_reward:.2f} | "
                      f"Avg Score: {avg_score:.2f} | "
                      f"Best Score: {best_score}")
            
            # Save best model
            if score > best_score:
                best_score = score
                self.agent.save_model(f"best_{save_path}")
            
            # Save checkpoint
            if episode % save_interval == 0:
                self.agent.save_model(save_path)
                print(f"Model saved to {save_path}")
        
        # Final save
        self.agent.save_model(save_path)
        print(f"\nTraining complete! Final model saved to {save_path}")
        print(f"Best score achieved: {best_score}")
        
        return self.episode_rewards, self.episode_scores

if __name__ == '__main__':
    # Create trainer
    trainer = SnakeTrainer(grid_size=30)
    
    # Train the agent
    rewards, scores = trainer.train(num_episodes=1000, save_path='snake_agent.pth')
    
    print("\nTraining statistics:")
    print(f"Final average reward (last 100 episodes): {np.mean(rewards[-100:]):.2f}")
    print(f"Final average score (last 100 episodes): {np.mean(scores[-100:]):.2f}")