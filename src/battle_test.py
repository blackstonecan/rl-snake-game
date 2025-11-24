#!/usr/bin/env python3
"""
Battle Test Script
Runs N games between two models and provides statistics
"""
import argparse
import os
import sys
import numpy as np
from tqdm import tqdm
from snake.snake_game_multiplayer import SnakeGameMultiplayer
from agents.agent_middleware_large import AgentMiddlewareLarge

class BattleTest:
    def __init__(self, model1_path, model2_path, grid_size=30):
        self.grid_size = grid_size
        self.game = SnakeGameMultiplayer(grid_size)
        
        # Load models
        self.agent1 = self.load_agent(model1_path, "Agent 1")
        self.agent2 = self.load_agent(model2_path, "Agent 2")
        
        # Statistics
        self.results = []
        self.scores1 = []
        self.scores2 = []
        self.lengths1 = []
        self.lengths2 = []
        self.steps = []
    
    def load_agent(self, path, name):
        """Load an agent model"""
        if path.lower() == 'random':
            print(f"{name}: Random Agent")
            return None, 'random'
        
        agent = AgentMiddlewareLarge(path)
        
        print(f"{name}: Loaded from {path}")
        return agent
    
    def run_game(self):
        """Run a single game"""
        self.game.reset()
        steps = 0
        max_steps = 10000
        
        while not self.game.is_game_over() and steps < max_steps:
            # Get states
            state1 = self.game.get_state(1)
            state2 = self.game.get_state(2)
            
            # Agent 1 action
            if not state1['game_over']:
                if self.agent1:
                    direction1 = self.agent1.get_action(state1, state2, epsilon=0.0)
                    self.game.set_direction(1, direction1)
            
            # Agent 2 action
            if not state2['game_over']:
                if self.agent2:
                    direction2 = self.agent2.get_action(state2, state1, epsilon=0.0)
                    self.game.set_direction(2, direction2)
            
            self.game.step()
            steps += 1
        
        # Record results
        winner = self.game.get_winner()
        self.results.append(winner)
        self.scores1.append(self.game.score1)
        self.scores2.append(self.game.score2)
        self.lengths1.append(len(self.game.snake1))
        self.lengths2.append(len(self.game.snake2))
        self.steps.append(steps)
        
        return winner
    
    def run_tournament(self, num_games=1000):
        """Run multiple games"""
        print(f"\nRunning {num_games} games...")
        print("=" * 60)
        
        for i in tqdm(range(num_games), desc="Playing"):
            self.run_game()
        
        print("\n" + "=" * 60)
        self.print_statistics()
    
    def print_statistics(self):
        """Print tournament statistics"""
        total_games = len(self.results)
        
        # Count wins
        wins1 = self.results.count(1)
        wins2 = self.results.count(2)
        draws = self.results.count(0)
        
        # Calculate percentages
        win_rate1 = (wins1 / total_games) * 100
        win_rate2 = (wins2 / total_games) * 100
        draw_rate = (draws / total_games) * 100
        
        print("\n" + "=" * 60)
        print("BATTLE RESULTS")
        print("=" * 60)
        print(f"Total Games: {total_games}")
        print()
        
        # Overall results
        print("Overall Results:")
        print(f"  Agent 1 Wins: {wins1:4d} ({win_rate1:5.1f}%)")
        print(f"  Agent 2 Wins: {wins2:4d} ({win_rate2:5.1f}%)")
        print(f"  Draws:        {draws:4d} ({draw_rate:5.1f}%)")
        print()
        
        # Score statistics
        print("Score Statistics:")
        print(f"  Agent 1 - Avg: {np.mean(self.scores1):.2f}, "
              f"Max: {np.max(self.scores1)}, Min: {np.min(self.scores1)}")
        print(f"  Agent 2 - Avg: {np.mean(self.scores2):.2f}, "
              f"Max: {np.max(self.scores2)}, Min: {np.min(self.scores2)}")
        print()
        
        # Length statistics
        print("Final Length Statistics:")
        print(f"  Agent 1 - Avg: {np.mean(self.lengths1):.2f}, "
              f"Max: {np.max(self.lengths1)}, Min: {np.min(self.lengths1)}")
        print(f"  Agent 2 - Avg: {np.mean(self.lengths2):.2f}, "
              f"Max: {np.max(self.lengths2)}, Min: {np.min(self.lengths2)}")
        print()
        
        # Game duration
        print("Game Duration:")
        print(f"  Avg Steps: {np.mean(self.steps):.1f}")
        print(f"  Max Steps: {np.max(self.steps)}")
        print(f"  Min Steps: {np.min(self.steps)}")
        print()
        
        # Head-to-head comparison
        print("Head-to-Head:")
        if wins1 > wins2:
            advantage = ((wins1 - wins2) / total_games) * 100
            print(f"  🏆 Agent 1 dominates with {advantage:.1f}% advantage")
        elif wins2 > wins1:
            advantage = ((wins2 - wins1) / total_games) * 100
            print(f"  🏆 Agent 2 dominates with {advantage:.1f}% advantage")
        else:
            print(f"  ⚖️  Perfectly balanced!")
        print()
        
        # Performance rating
        print("Performance Rating:")
        rating1 = self.calculate_rating(wins1, draws, self.scores1, self.lengths1)
        rating2 = self.calculate_rating(wins2, draws, self.scores2, self.lengths2)
        print(f"  Agent 1: {rating1:.1f}/100")
        print(f"  Agent 2: {rating2:.1f}/100")
        print()
        
        # Detailed win analysis
        print("Detailed Analysis:")
        
        # Games where agent 1 won
        agent1_wins_indices = [i for i, r in enumerate(self.results) if r == 1]
        if agent1_wins_indices:
            avg_score_when_win1 = np.mean([self.scores1[i] for i in agent1_wins_indices])
            avg_steps_when_win1 = np.mean([self.steps[i] for i in agent1_wins_indices])
            print(f"  When Agent 1 wins: Avg score {avg_score_when_win1:.1f}, Avg steps {avg_steps_when_win1:.1f}")
        
        # Games where agent 2 won
        agent2_wins_indices = [i for i, r in enumerate(self.results) if r == 2]
        if agent2_wins_indices:
            avg_score_when_win2 = np.mean([self.scores2[i] for i in agent2_wins_indices])
            avg_steps_when_win2 = np.mean([self.steps[i] for i in agent2_wins_indices])
            print(f"  When Agent 2 wins: Avg score {avg_score_when_win2:.1f}, Avg steps {avg_steps_when_win2:.1f}")
        
        print("=" * 60)
    
    def calculate_rating(self, wins, draws, scores, lengths):
        """Calculate a performance rating out of 100"""
        total = len(self.results)
        win_score = (wins / total) * 50  # 50% weight on wins
        draw_score = (draws / total) * 10  # 10% weight on draws
        avg_score = (np.mean(scores) / 20) * 20  # 20% weight on apples (normalized to 20)
        avg_length = (np.mean(lengths) / 30) * 20  # 20% weight on length (normalized to 30)
        
        return min(100, win_score + draw_score + avg_score + avg_length)

def main():
    parser = argparse.ArgumentParser(description='Battle test two snake agents')
    parser.add_argument('agent1', type=str, help='Path to agent 1 model (or "random")')
    parser.add_argument('agent2', type=str, help='Path to agent 2 model (or "random")')
    parser.add_argument('-n', '--num-games', type=int, default=100,
                       help='Number of games to play (default: 100)')
    parser.add_argument('-g', '--grid-size', type=int, default=30,
                       help='Grid size (default: 30)')
    
    args = parser.parse_args()
    
    # Run tournament
    battle = BattleTest(args.agent1, args.agent2, args.grid_size)
    battle.run_tournament(args.num_games)

if __name__ == '__main__':
    main()