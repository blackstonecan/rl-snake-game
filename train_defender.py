import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random
from snake_game_multiplayer import SnakeGameMultiplayer, Direction
from agent_middleware_large import AgentMiddlewareLarge
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

class DefenderTrainer:
    def __init__(self, grid_size=30, opponent_model_path='killer_agent.pth'):
        self.grid_size = grid_size
        self.game = SnakeGameMultiplayer(grid_size)
        
        # Defender agent (7x7, learning) - Using Large for better field of view
        self.defender_agent = AgentMiddlewareLarge()
        
        # Opponent agent (Frozen) - Defaults to Killer for best training
        self.opponent_agent = None
        if os.path.exists(opponent_model_path):
            print(f"Loading opponent from {opponent_model_path}")

            self.opponent_agent = AgentMiddlewareLarge(opponent_model_path)    
            self.opponent_agent.model.eval()  # Freeze opponent
        else:
            print(f"Warning: {opponent_model_path} not found. Opponent will play randomly.")
        
        self.device = self.defender_agent.device
        self.optimizer = optim.Adam(self.defender_agent.model.parameters(), lr=0.0005) # Lower LR for stability
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
        self.episode_steps_alive = []
        self.episode_wins = []
    
    def calculate_reward(self, prev_state, current_state, opponent_prev_state, opponent_current_state):
        """
        Calculate reward for DEFENDER agent
        Defensive rewards: Prioritize survival, eating apples, and keeping distance.
        """
        reward = 0
        
        # 1. CRITICAL: Survival Penalty/Reward
        if current_state['game_over']:
            # Massive penalty for dying
            return -500
        
        # Small reward just for staying alive every step
        reward += 1
        
        # 2. WINNING: If opponent died and we are alive
        if opponent_current_state['game_over'] and not opponent_prev_state['game_over']:
            reward += 50  # We outlived them! Good job.
        
        # 3. FOOD: Reward for eating apple (necessary to not starve/grow)
        if current_state['score'] > prev_state['score']:
            reward += 30
        
        # 4. SPACING: Logic to keep distance from opponent
        prev_head = prev_state['snake'][0]
        curr_head = current_state['snake'][0]
        opp_curr_head = opponent_current_state['snake'][0]
        
        def distance(pos1, pos2):
            dx = abs(pos1[0] - pos2[0])
            dy = abs(pos1[1] - pos2[1])
            # Consider wraparound
            dx = min(dx, self.grid_size - dx)
            dy = min(dy, self.grid_size - dy)
            return dx + dy
        
        prev_dist = distance(prev_head, opp_curr_head)
        curr_dist = distance(curr_head, opp_curr_head)
        
        # DANGER ZONE LOGIC
        danger_radius = 8
        
        if curr_dist < danger_radius:
            # We are in danger!
            if curr_dist > prev_dist:
                # Moving AWAY from opponent -> GOOD
                reward += 5 
            elif curr_dist < prev_dist:
                # Moving TOWARDS opponent -> BAD
                reward -= 10
        else:
            # We are safe. 
            # Slight penalty for being too far? No, just exist.
            pass

        # 5. Penalty for trapping self (simple heuristic)
        # If our body length is high, punish creating loops? (Optional, skipped for now)
        
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
        current_q_values = self.defender_agent.model(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # Next Q values
        with torch.no_grad():
            next_q_values = self.defender_agent.model(next_states).max(1)[0]
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
        # Handle cases where next dir is opposite (shouldn't happen with valid logic but safe to default)
        if chosen_dir.name not in action_map[current_dir.name]:
            return 0 # Maintain course
        return action_map[current_dir.name][chosen_dir.name]
    
    def train_episode(self):
        """Train for one episode"""
        self.game.reset()
        episode_reward = 0
        step_count = 0
        max_steps = 2000 # Defenders need to last longer
        
        while not self.game.is_game_over() and step_count < max_steps:
            # Get current states
            def_prev_state = self.game.get_state(1)
            opp_prev_state = self.game.get_state(2)
            
            # Only proceed if defender is alive
            if not def_prev_state['game_over']:
                def_prev_obs = self.defender_agent.get_observation(def_prev_state, opp_prev_state)
                
                # Defender action
                def_action_dir = self.defender_agent.get_action(
                    def_prev_state, 
                    opp_prev_state, 
                    epsilon=self.epsilon
                )
                self.game.set_direction(1, def_action_dir)
            
            # Opponent action (if alive and agent exists)
            if not opp_prev_state['game_over']:
                if self.opponent_agent:
                    # Opponent plays optimally (no epsilon)
                    opp_action_dir = self.opponent_agent.get_action(opp_prev_state, def_prev_state, epsilon=0.0)
                    self.game.set_direction(2, opp_action_dir)
                else:
                    # Random action if no opponent model
                    random_dir = random.choice([Direction.UP, Direction.DOWN, Direction.LEFT, Direction.RIGHT])
                    self.game.set_direction(2, random_dir)
            
            # Take step
            self.game.step()
            
            # Get new states
            def_curr_state = self.game.get_state(1)
            opp_curr_state = self.game.get_state(2)
            
            # Only store experience if defender was alive before this step
            if not def_prev_state['game_over']:
                def_curr_obs = self.defender_agent.get_observation(def_curr_state, opp_curr_state)
                
                # Calculate reward
                reward = self.calculate_reward(
                    def_prev_state, 
                    def_curr_state,
                    opp_prev_state,
                    opp_curr_state
                )
                episode_reward += reward
                
                # Convert action to index
                action_idx = self.direction_to_action_idx(
                    def_prev_state['direction'], 
                    def_action_dir
                )
                
                # Store in replay buffer
                self.replay_buffer.push(
                    def_prev_obs, 
                    action_idx, 
                    reward, 
                    def_curr_obs,
                    1.0 if def_curr_state['game_over'] else 0.0
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
        self.episode_steps_alive.append(step_count)
        self.episode_wins.append(1 if won else 0)
        
        return episode_reward, self.game.score1, step_count, won
    
    def train(self, num_episodes=1000, save_path='defender_agent.pth', save_interval=100):
        """Train the defender agent"""
        print(f"Training Defender Agent on device: {self.device}")
        print(f"Opponent: {self.opponent_agent.__class__.__name__ if self.opponent_agent else 'Random'}")
        print(f"Starting training for {num_episodes} episodes...\n")
        
        best_steps = 0
        
        for episode in range(1, num_episodes + 1):
            reward, score, steps, won = self.train_episode()
            
            # Print progress
            if episode % 10 == 0:
                avg_reward = np.mean(self.episode_rewards[-10:])
                avg_score = np.mean(self.episode_scores[-10:])
                avg_steps = np.mean(self.episode_steps_alive[-10:])
                avg_wins = np.mean(self.episode_wins[-10:])
                
                print(f"Episode {episode}/{num_episodes} | "
                      f"Eps: {self.epsilon:.2f} | "
                      f"Reward: {avg_reward:.1f} | "
                      f"Score: {avg_score:.1f} | "
                      f"Steps: {avg_steps:.1f} | "
                      f"Win%: {avg_wins:.2f}")
            
            # Save best model (based on survival/steps)
            recent_steps = np.mean(self.episode_steps_alive[-50:]) if len(self.episode_steps_alive) >= 50 else np.mean(self.episode_steps_alive)
            if recent_steps > best_steps and episode > 50:
                best_steps = recent_steps
                self.defender_agent.save_model(f"best_{save_path}")
            
            # Save checkpoint
            if episode % save_interval == 0:
                self.defender_agent.save_model(save_path)
                print(f"Model saved to {save_path}")
        
        # Final save
        self.defender_agent.save_model(save_path)
        print(f"\nTraining complete! Final model saved to {save_path}")
        print(f"Best avg steps (last 50 episodes): {best_steps:.1f}")
        
        return self.episode_rewards, self.episode_scores, self.episode_steps_alive, self.episode_wins

if __name__ == '__main__':
    # Create trainer - Trains against 'killer_agent.pth' if available
    # Assuming killer agent is the 7x7 one you just trained
    trainer = DefenderTrainer(grid_size=30, opponent_model_path='killer_agent.pth')
    
    # Train the defender agent
    rewards, scores, steps, wins = trainer.train(
        num_episodes=1000, 
        save_path='defender_agent.pth'
    )
    
    print("\nTraining statistics:")
    print(f"Final average reward (last 100): {np.mean(rewards[-100:]):.2f}")
    print(f"Final average steps alive (last 100): {np.mean(steps[-100:]):.2f}")
    print(f"Final win rate (last 100): {np.mean(wins[-100:]):.2f}")