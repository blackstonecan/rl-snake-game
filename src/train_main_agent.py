import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random
from snake.snake_game_multiplayer import SnakeGameMultiplayer, Direction
from agents.agent_middleware_large import AgentMiddlewareLarge
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

class OpponentPool:
    """Manages different opponent types for curriculum learning"""
    def __init__(self, killer_path='killer_agent.pth', peaceful_path='defender_agent.pth'):
        self.opponents = {}
        self.checkpoints = {}  # For self-play
        
        # Load killer agent (7x7, aggressive)
        print(f"✓ Loaded killer agent from {killer_path}")
        killer = AgentMiddlewareLarge(killer_path)
        killer.model.eval()
        self.opponents['killer'] = killer
        
        # Load peaceful agent (5x5, apple-focused)
        print(f"✓ Loaded peaceful agent from {peaceful_path}")
        peaceful = AgentMiddlewareLarge(peaceful_path)
        peaceful.model.eval()
        self.opponents['peaceful'] = peaceful
    
    def add_checkpoint(self, name, agent_copy):
        """Add a checkpoint for self-play"""
        self.checkpoints[name] = agent_copy
    
    def get_opponent_weights(self, episode):
        """
        Get opponent sampling weights based on curriculum
        Phase 1 (0-600):   70% killer, 30% peaceful
        Phase 2 (600-1000): 40% killer, 20% peaceful, 40% self
        Phase 3 (1000+):    10% killer, 10% peaceful, 80% self
        """
        if episode < 600:
            # Phase 1: Learn defense
            return {
                'killer': 0.7,
                'peaceful': 0.3,
                'self': 0.0
            }
        elif episode < 1000:
            # Phase 2: Transition
            return {
                'killer': 0.4,
                'peaceful': 0.2,
                'self': 0.4
            }
        else:
            # Phase 3: Self-play dominance
            return {
                'killer': 0.1,
                'peaceful': 0.1,
                'self': 0.8
            }
    
    def sample_opponent(self, episode):
        """Sample an opponent based on curriculum"""
        weights = self.get_opponent_weights(episode)
        
        # Build available opponents
        choices = []
        probs = []
        
        if self.opponents['killer'] and weights['killer'] > 0:
            choices.append('killer')
            probs.append(weights['killer'])
        
        if self.opponents['peaceful'] and weights['peaceful'] > 0:
            choices.append('peaceful')
            probs.append(weights['peaceful'])
        
        if len(self.checkpoints) > 0 and weights['self'] > 0:
            choices.append('self')
            probs.append(weights['self'])
        
        if not choices:
            return None, None
        
        # Normalize probabilities
        total = sum(probs)
        probs = [p/total for p in probs]
        
        # Sample
        choice = np.random.choice(choices, p=probs)
        
        if choice == 'killer':
            return self.opponents['killer'], 'killer'
        elif choice == 'peaceful':
            return self.opponents['peaceful'], 'peaceful'
        else:  # self
            # Randomly pick from checkpoints
            checkpoint_name = random.choice(list(self.checkpoints.keys()))
            return self.checkpoints[checkpoint_name], f'self_{checkpoint_name}'

class MainAgentTrainer:
    def __init__(self, grid_size=30, killer_path='killer_agent.pth', peaceful_path='defender_agent.pth'):
        self.grid_size = grid_size
        self.game = SnakeGameMultiplayer(grid_size)
        
        # Main agent (7x7, learning)
        self.main_agent = AgentMiddlewareLarge()
        
        # Opponent pool
        self.opponent_pool = OpponentPool(killer_path, peaceful_path)
        
        self.device = self.main_agent.device
        self.optimizer = optim.Adam(self.main_agent.model.parameters(), lr=0.001)
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
        self.episode_wins = []
        self.episode_opponent_types = []
    
    def calculate_reward(self, prev_state, current_state, opponent_prev_state, opponent_current_state):
        """
        Calculate reward for main agent
        Balanced rewards: survival, apples, winning
        """
        reward = 0
        
        # Check if main died
        if current_state['game_over']:
            if opponent_current_state['game_over']:
                return -100
            else:
                return -200
        
        # Win condition: opponent dies
        if opponent_current_state['game_over'] and not opponent_prev_state['game_over']:
            reward += 50  # Victory!
        
        # Reward for eating apple
        if current_state['score'] > prev_state['score']:
            reward += 10
        
        # Reward for being longer than opponent
        my_length = len(current_state['snake'])
        opp_length = len(opponent_current_state['snake'])
        if my_length > opp_length:
            reward += 0.5
        
        # Penalty if opponent eats apple (competition)
        if opponent_current_state['score'] > opponent_prev_state['score']:
            reward -= 1
        
        # Small step penalty (encourages efficiency)
        reward -= 0.01
        
        # Bonus for survival when close to opponent (defensive awareness)
        def distance(pos1, pos2):
            dx = abs(pos1[0] - pos2[0])
            dy = abs(pos1[1] - pos2[1])
            dx = min(dx, self.grid_size - dx)
            dy = min(dy, self.grid_size - dy)
            return dx + dy
        
        my_head = current_state['snake'][0]
        opp_head = opponent_current_state['snake'][0]
        dist = distance(my_head, opp_head)
        
        if dist < 5:  # Close proximity
            reward += 0.1  # Survived near opponent
        
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
        current_q_values = self.main_agent.model(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # Next Q values
        with torch.no_grad():
            next_q_values = self.main_agent.model(next_states).max(1)[0]
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
    
    def train_episode(self, episode):
        """Train for one episode"""
        self.game.reset()
        episode_reward = 0
        step_count = 0
        max_steps = 1000
        
        # Sample opponent
        opponent_agent, opponent_type = self.opponent_pool.sample_opponent(episode)
        
        if opponent_agent is None:
            print("Warning: No opponent available, skipping episode")
            return 0, 0, False, None
        
        while not self.game.is_game_over() and step_count < max_steps:
            # Get current states
            main_prev_state = self.game.get_state(1)
            opponent_prev_state = self.game.get_state(2)
            
            # Main agent action (if alive)
            if not main_prev_state['game_over']:
                main_prev_obs = self.main_agent.get_observation(main_prev_state, opponent_prev_state)
                
                main_action_dir = self.main_agent.get_action(
                    main_prev_state, 
                    opponent_prev_state, 
                    epsilon=self.epsilon
                )
                self.game.set_direction(1, main_action_dir)
            
            # Opponent action (if alive)
            if not opponent_prev_state['game_over']:
                if opponent_type == 'peaceful':
                    # 5x5 agent doesn't need opponent state
                    opponent_action_dir = opponent_agent.get_action(
                        opponent_prev_state, 
                        epsilon=0.0
                    )
                else:
                    # 7x7 agent (killer or self) needs opponent state
                    opponent_action_dir = opponent_agent.get_action(
                        opponent_prev_state,
                        main_prev_state,  # Main is opponent to them
                        epsilon=0.0
                    )
                self.game.set_direction(2, opponent_action_dir)
            
            # Take step
            self.game.step()
            
            # Get new states
            main_curr_state = self.game.get_state(1)
            opponent_curr_state = self.game.get_state(2)
            
            # Only store experience if main was alive
            if not main_prev_state['game_over']:
                main_curr_obs = self.main_agent.get_observation(main_curr_state, opponent_curr_state)
                
                # Calculate reward
                reward = self.calculate_reward(
                    main_prev_state, 
                    main_curr_state,
                    opponent_prev_state,
                    opponent_curr_state
                )
                episode_reward += reward
                
                # Convert action to index
                action_idx = self.direction_to_action_idx(
                    main_prev_state['direction'], 
                    main_action_dir
                )
                
                # Store in replay buffer
                self.replay_buffer.push(
                    main_prev_obs, 
                    action_idx, 
                    reward, 
                    main_curr_obs,
                    1.0 if main_curr_state['game_over'] else 0.0
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
        self.episode_wins.append(1 if won else 0)
        self.episode_opponent_types.append(opponent_type)
        
        return episode_reward, self.game.score1, won, opponent_type
    
    def save_checkpoint(self, episode):
        """Save checkpoint for self-play"""
        # Create a copy of the model
        checkpoint_agent = AgentMiddlewareLarge()
        checkpoint_agent.model.load_state_dict(self.main_agent.model.state_dict())
        checkpoint_agent.model.eval()
        
        # Add to opponent pool
        self.opponent_pool.add_checkpoint(f'ep{episode}', checkpoint_agent)
        
        # Keep only last 3 checkpoints to save memory
        if len(self.opponent_pool.checkpoints) > 3:
            oldest = min(self.opponent_pool.checkpoints.keys())
            del self.opponent_pool.checkpoints[oldest]
    
    def train(self, num_episodes=2000, save_path='main_agent.pth', save_interval=100, checkpoint_interval=200):
        """Train the main agent"""
        print(f"Training Main Competitive Agent on device: {self.device}")
        print(f"Starting training for {num_episodes} episodes...\n")
        
        best_win_rate = 0
        
        for episode in range(1, num_episodes + 1):
            reward, score, won, opponent_type = self.train_episode(episode)
            
            # Print progress
            if episode % 10 == 0:
                avg_reward = np.mean(self.episode_rewards[-10:])
                avg_score = np.mean(self.episode_scores[-10:])
                avg_win = np.mean(self.episode_wins[-10:])
                
                # Count opponent types in last 10
                recent_opponents = self.episode_opponent_types[-10:]
                killer_pct = sum(1 for o in recent_opponents if o == 'killer') / 10
                peaceful_pct = sum(1 for o in recent_opponents if o == 'peaceful') / 10
                self_pct = sum(1 for o in recent_opponents if 'self' in o) / 10
                
                print(f"Episode {episode}/{num_episodes} | "
                      f"Epsilon: {self.epsilon:.3f} | "
                      f"Reward: {avg_reward:.2f} | "
                      f"Score: {avg_score:.2f} | "
                      f"Win Rate: {avg_win:.2f} | "
                      f"Opp: K{killer_pct:.1f}/P{peaceful_pct:.1f}/S{self_pct:.1f}")
            
            # Save best model (based on win rate)
            if len(self.episode_wins) >= 100:
                recent_win_rate = np.mean(self.episode_wins[-100:])
                if recent_win_rate > best_win_rate:
                    best_win_rate = recent_win_rate
                    self.main_agent.save_model(f"best_{save_path}")
                    print(f"New best win rate: {best_win_rate:.2%}")
            
            # Save checkpoint for self-play
            if episode % checkpoint_interval == 0:
                self.save_checkpoint(episode)
                print(f"Checkpoint saved for self-play (episode {episode})")
            
            # Save model
            if episode % save_interval == 0:
                self.main_agent.save_model(save_path)
                print(f"Model saved to {save_path}")
        
        # Final save
        self.main_agent.save_model(save_path)
        print(f"\nTraining complete! Final model saved to {save_path}")
        print(f"Best win rate achieved: {best_win_rate:.2%}")
        
        return self.episode_rewards, self.episode_scores, self.episode_wins

if __name__ == '__main__':
    # Create trainer
    trainer = MainAgentTrainer(
        grid_size=30,
        killer_path='killer_agent.pth',
        peaceful_path='defender_agent.pth'
    )
    
    # Train the main agent
    rewards, scores, wins = trainer.train(
        num_episodes=2000,
        save_path='main_agent.pth',
        save_interval=100,
        checkpoint_interval=200
    )
    
    print("\nTraining statistics:")
    print(f"Final average reward (last 100 episodes): {np.mean(rewards[-100:]):.2f}")
    print(f"Final average score (last 100 episodes): {np.mean(scores[-100:]):.2f}")
    print(f"Final win rate (last 100 episodes): {np.mean(wins[-100:]):.2%}")
    
    # Stats by opponent type
    print("\nPerformance by opponent type (last 100 episodes):")
    recent_data = list(zip(trainer.episode_opponent_types[-100:], wins[-100:]))
    
    for opp_type in ['killer', 'peaceful', 'self']:
        type_wins = [w for o, w in recent_data if opp_type in o]
        if type_wins:
            print(f"  vs {opp_type}: {np.mean(type_wins):.2%} win rate ({len(type_wins)} games)")