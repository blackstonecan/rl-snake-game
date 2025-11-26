import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random
from game.snake_game_multiplayer import SnakeGameMultiplayer, Direction
from model.agent_middleware_large import AgentMiddlewareLarge
import os

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

class KillerTrainer:
    def __init__(self, opponent_model_path):
        self.grid_size = 30
        self.game = SnakeGameMultiplayer(self.grid_size)
        
        # Killer agent (7x7, learning)
        self.killer_agent = AgentMiddlewareLarge()
        
        # Opponent agent (5x5, frozen)
        print(f"Loading opponent from {opponent_model_path}")

        self.opponent_agent = AgentMiddlewareLarge(opponent_model_path)
        self.opponent_agent.model.eval()  # Freeze opponent
        
        self.device = self.killer_agent.device
        self.optimizer = optim.Adam(self.killer_agent.model.parameters(), lr=0.001)
        self.criterion = nn.MSELoss()
        
        self.replay_buffer = ReplayBuffer(capacity=10000)
        self.batch_size = 64
        self.gamma = 0.95
        
        # Training parameters
        self.epsilon_start = 1.0
        self.epsilon_end = 0.01
        self.epsilon_decay = 0.995
        self.epsilon = self.epsilon_start
        
        # Statistics
        self.episode_rewards = []
        self.episode_scores = []
        self.episode_kills = []
        self.episode_wins = []
    
    def calculate_reward(self, prev_state, current_state, opponent_prev_state, opponent_current_state):
        """
        Calculate reward for killer agent
        Aggressive rewards: prioritize killing opponent
        """
        reward = 0
        
        # Check if killer died
        if current_state['game_over']:
            if opponent_current_state['game_over']:
                return -200
            else:
                return -300
        
        # MAIN GOAL: Kill opponent
        if opponent_current_state['game_over'] and not opponent_prev_state['game_over']:
            reward += 100  # Killed opponent!
        
        # Reward for eating apple (need to survive)
        if current_state['score'] > prev_state['score']:
            reward += 20
        
        # Reward for moving closer to opponent
        prev_head = prev_state['snake'][0]
        curr_head = current_state['snake'][0]
        opp_prev_head = opponent_prev_state['snake'][0]
        opp_curr_head = opponent_current_state['snake'][0]
        
        def distance(pos1, pos2):
            dx = abs(pos1[0] - pos2[0])
            dy = abs(pos1[1] - pos2[1])
            # Consider wraparound
            dx = min(dx, self.grid_size - dx)
            dy = min(dy, self.grid_size - dy)
            return dx + dy
        
        prev_dist = distance(prev_head, opp_prev_head)
        curr_dist = distance(curr_head, opp_curr_head)
        
        if curr_dist < prev_dist:
            reward += 2  # Moving toward opponent
        elif curr_dist > prev_dist:
            reward -= 1  # Moving away from opponent
        
        # Penalty if opponent is getting longer
        if opponent_current_state['score'] > opponent_prev_state['score']:
            reward -= 1
        
        # Small step penalty
        reward -= 0.1
        
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
        current_q_values = self.killer_agent.model(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # Next Q values
        with torch.no_grad():
            next_q_values = self.killer_agent.model(next_states).max(1)[0]
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
        max_steps = 1000
        killed_opponent = False
        
        while not self.game.is_game_over() and step_count < max_steps:
            # Get current states
            killer_prev_state = self.game.get_state(1)
            opponent_prev_state = self.game.get_state(2)
            
            # Only proceed if killer is alive
            if not killer_prev_state['game_over']:
                killer_prev_obs = self.killer_agent.get_observation(killer_prev_state, opponent_prev_state)
                
                # Killer action
                killer_action_dir = self.killer_agent.get_action(
                    killer_prev_state, 
                    opponent_prev_state, 
                    epsilon=self.epsilon
                )
                self.game.set_direction(1, killer_action_dir)
            
            # Opponent action (if alive and agent exists)
            if not opponent_prev_state['game_over']:
                if self.opponent_agent:
                    opponent_action_dir = self.opponent_agent.get_action(
                        opponent_prev_state, 
                        epsilon=0.0  # No exploration for frozen opponent
                    )
                    self.game.set_direction(2, opponent_action_dir)
                else:
                    # Random action if no opponent model
                    random_dir = random.choice([Direction.UP, Direction.DOWN, Direction.LEFT, Direction.RIGHT])
                    self.game.set_direction(2, random_dir)
            
            # Take step
            self.game.step()
            
            # Get new states
            killer_curr_state = self.game.get_state(1)
            opponent_curr_state = self.game.get_state(2)
            
            # Only store experience if killer was alive before this step
            if not killer_prev_state['game_over']:
                killer_curr_obs = self.killer_agent.get_observation(killer_curr_state, opponent_curr_state)
                
                # Calculate reward
                reward = self.calculate_reward(
                    killer_prev_state, 
                    killer_curr_state,
                    opponent_prev_state,
                    opponent_curr_state
                )
                episode_reward += reward
                
                # Check if killed opponent
                if opponent_curr_state['game_over'] and not opponent_prev_state['game_over']:
                    killed_opponent = True
                
                # Convert action to index
                action_idx = self.direction_to_action_idx(
                    killer_prev_state['direction'], 
                    killer_action_dir
                )
                
                # Store in replay buffer
                self.replay_buffer.push(
                    killer_prev_obs, 
                    action_idx, 
                    reward, 
                    killer_curr_obs,
                    1.0 if killer_curr_state['game_over'] else 0.0
                )
                
                # Train
                loss = self.train_step()
            
            step_count += 1
        
        # Decay epsilon
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
        
        # Get winner
        winner = self.game.get_winner()
        won = (winner == 1)
        
        # Store statistics
        self.episode_rewards.append(episode_reward)
        self.episode_scores.append(self.game.score1)
        self.episode_kills.append(1 if killed_opponent else 0)
        self.episode_wins.append(1 if won else 0)
        
        return episode_reward, self.game.score1, killed_opponent, won
    
    def train(self, num_episodes, save_path, best_save_path):
        """Train the killer agent"""
        print(f"Training Killer Agent on device: {self.device}")
        print(f"Starting training for {num_episodes} episodes...\n")
        
        best_kills = 0
        
        for episode in range(1, num_episodes + 1):
            reward, score, killed, won = self.train_episode()
            
            # Print progress
            if episode % 10 == 0:
                avg_reward = np.mean(self.episode_rewards[-10:])
                avg_score = np.mean(self.episode_scores[-10:])
                avg_kills = np.mean(self.episode_kills[-10:])
                avg_wins = np.mean(self.episode_wins[-10:])
                
                print(f"Episode {episode}/{num_episodes} | "
                      f"Epsilon: {self.epsilon:.3f} | "
                      f"Avg Reward: {avg_reward:.2f} | "
                      f"Avg Score: {avg_score:.2f} | "
                      f"Kill Rate: {avg_kills:.2f} | "
                      f"Win Rate: {avg_wins:.2f}")
            
            # Save best model (based on kills)
            recent_kills = sum(self.episode_kills[-100:]) if len(self.episode_kills) >= 100 else sum(self.episode_kills)
            if recent_kills > best_kills:
                best_kills = recent_kills
                self.killer_agent.save_model(best_save_path)
            
            # Save checkpoint
            if episode % 100 == 0:
                self.killer_agent.save_model(save_path)
                print(f"Model saved to {save_path}")
        
        # Final save
        self.killer_agent.save_model(save_path)
        print(f"\nTraining complete! Final model saved to {save_path}")
        print(f"Best kill count (last 100 episodes): {best_kills}")
        
        return self.episode_rewards, self.episode_scores, self.episode_kills, self.episode_wins

if __name__ == '__main__':
    save_path = './agents/killer_agent.pth'
    save_best_path = './agents/best_killer_agent.pth'
    
    opponent_model_path = './agents/defender_agent.pth'

    # Create trainer
    trainer = KillerTrainer(opponent_model_path)
    
    # Train the killer agent
    rewards, scores, kills, wins = trainer.train(
        num_episodes=1000, 
        save_path=save_path,
        best_save_path=save_best_path
    )
    
    print("\nTraining statistics:")
    print(f"Final average reward (last 100 episodes): {np.mean(rewards[-100:]):.2f}")
    print(f"Final average score (last 100 episodes): {np.mean(scores[-100:]):.2f}")
    print(f"Final kill rate (last 100 episodes): {np.mean(kills[-100:]):.2f}")
    print(f"Final win rate (last 100 episodes): {np.mean(wins[-100:]):.2f}")