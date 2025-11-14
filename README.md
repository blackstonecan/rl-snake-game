# Snake Game AI - Multiplayer Battle Arena

A reinforcement learning project featuring competitive AI agents that learn to play multiplayer snake using Deep Q-Learning (DQN). Watch AI snakes battle it out in a PyQt5 GUI environment!

## Features

- **Multiplayer Snake Game**: Classic snake game with 2-player competitive gameplay
- **Deep Reinforcement Learning**: AI agents trained using DQN with experience replay
- **Multiple Agent Types**:
  - **Main Agent**: Balanced agent trained with curriculum learning and self-play
  - **Killer Agent**: Aggressive agent focused on eliminating opponents
  - **Defender Agent**: Defensive agent focused on survival and apple collection
- **Interactive GUI**: PyQt5-based interface to watch AI battles in real-time
- **Curriculum Learning**: Progressive training from simple to complex opponents
- **Self-Play Training**: Agents improve by competing against past versions of themselves

## Project Structure

```
root/
├── snake_game_multiplayer.py       # Core game engine
├── snake_game_multiplayer_gui.py   # PyQt5 GUI for watching battles
├── agent_middleware_large.py       # Agent wrapper with observation processing
├── agent_middleware_structure.py   # Agent wrapper structure
├── snake_model_large.py            # Neural network architecture
├── train_main_agent.py             # Main agent training script
├── train_killer.py                 # Killer agent training script
├── train_defender.py               # Defender agent training script
├── battle_test.py                  # Automated battle testing
├── models/                         # Pre-trained agent models
│   ├── best_main_agent.pth
│   ├── best_killer_agent.pth
│   ├── best_defender_agent.pth
│   └── ...
└── requirements.txt                # Python dependencies
```

## Installation

### Prerequisites

- Python 3.8 or higher
- PyTorch (CPU or GPU version)
- PyQt5

### Setup

1. Clone the repository:
```bash
git clone https://github.com/blackstonecan/rl-snake-game.git
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Watch AI Battles (GUI)

Launch the interactive GUI to watch pre-trained agents battle:

```bash
python snake_game_multiplayer_gui.py
```

**GUI Controls:**
- Select agents from the dropdown menus for Snake 1 (Green) and Snake 2 (Blue)
- Click "Load Models" to load the selected agents
- Click "Start" to begin the battle
- Adjust speed with the speed slider (Slow/Normal/Fast/Very Fast)
- Click "Reset" to restart the game
- Click "Refresh Model List" to reload available models from the models/ directory

### Train New Agents

#### Train Main Agent
The main agent uses curriculum learning and self-play:

```bash
python train_main_agent.py
```

Features:
- Trains against killer and defender agents
- Progressive curriculum (70% killer → 40% diverse → 80% self-play)
- Saves checkpoints for self-play every 200 episodes
- Saves best model based on win rate

#### Train Killer Agent
Aggressive agent focused on eliminating opponents:

```bash
python train_killer.py
```

Features:
- Rewards opponent elimination
- 7x7 vision grid (includes opponent awareness)
- Aggressive playstyle

#### Train Defender Agent
Defensive agent focused on survival:

```bash
python train_defender.py
```

Features:
- Rewards survival and apple collection
- Defensive playstyle
- Apple-focused strategy

### Run Automated Battles

Test agents against each other programmatically:

```bash
python battle_test.py
```

```bash
python battle_test.py killer_agent.pth main_agent.pth -n 100
```

## Training Details

### Neural Network Architecture

- **Input**: State representation (7x7 or 5x5 grid observation)
- **Architecture**:
  - Fully connected layers
  - ReLU activations
  - Output: Q-values for 3 actions (forward, left, right)

### Observation Space

The agent observes:
- Snake head position
- Snake body positions
- Opponent positions (for 7x7 agents)
- Apple locations
- Wall positions
- Current direction

### Action Space

3 discrete actions:
- **Forward** (0): Continue in current direction
- **Left** (1): Turn left relative to current direction
- **Right** (2): Turn right relative to current direction

### Reward Structure

#### Main Agent
- **Win**: +50 (opponent dies)
- **Apple**: +10 (collected apple)
- **Length advantage**: +0.5 (longer than opponent)
- **Opponent apple**: -1 (opponent collected apple)
- **Death**: -200 (agent dies)
- **Draw**: -100 (both die)
- **Survival near opponent**: +0.1 (within 5 tiles)
- **Step penalty**: -0.01 (encourages efficiency)

#### Killer Agent
- **Kill opponent**: +100
- **Damage to opponent**: +5 (opponent loses length)
- **Apple**: +5
- **Death**: -50

#### Defender Agent
- **Survival**: +0.5 per step
- **Apple**: +10
- **Growth**: +2 per length increase
- **Death**: -100

## Hyperparameters

### Training Parameters
- **Episodes**: 2000 (configurable)
- **Replay Buffer Size**: 10,000 experiences
- **Batch Size**: 64
- **Learning Rate**: 0.001 (Adam optimizer)
- **Discount Factor (γ)**: 0.95
- **Epsilon Decay**: 0.995 (ε-greedy exploration)
- **Epsilon Range**: 1.0 → 0.01

### Game Parameters
- **Grid Size**: 30x30
- **Max Steps per Episode**: 1000
- **Apples**: Multiple (configurable)

## Model Files

Pre-trained models are saved in the `models/` directory:
- `best_main_agent.pth` - Best performing main agent
- `best_killer_agent.pth` - Best performing killer agent
- `best_defender_agent.pth` - Best performing defender agent
- Additional checkpoint models with version numbers

## Performance Metrics

During training, the following metrics are tracked:
- **Win Rate**: Percentage of games won
- **Average Score**: Average apples collected per episode
- **Average Reward**: Mean reward per episode
- **Opponent Distribution**: Percentage of games vs each opponent type

## Curriculum Learning Phases

The main agent training follows a 3-phase curriculum:

1. **Phase 1 (Episodes 0-600)**: Defense Training
   - 70% vs Killer agent
   - 30% vs Defender agent
   - Learn survival skills

2. **Phase 2 (Episodes 600-1000)**: Diversification
   - 40% vs Killer agent
   - 20% vs Defender agent
   - 40% vs Self (past checkpoints)
   - Balance offense and defense

3. **Phase 3 (Episodes 1000+)**: Self-Play Mastery
   - 10% vs Killer agent
   - 10% vs Defender agent
   - 80% vs Self
   - Continuous self-improvement

## Troubleshooting

### No models found in GUI
- Ensure `.pth` files are in the `models/` directory
- Click "Refresh Model List" button
- Check file permissions

### Training crashes
- Reduce batch size if running out of memory
- Ensure required agents exist for main agent training
- Check PyTorch installation (CPU vs GPU)

### GUI not launching
- Verify PyQt5 installation: `pip install PyQt5`
- Check Python version compatibility (3.8+)

## Dependencies

- `numpy>=1.21.0` - Numerical computations
- `torch>=1.13.0` - Deep learning framework
- `PyQt5>=5.15.0` - GUI framework
- `tqdm>=4.60.0` - Progress bars (optional)

## Acknowledgments

Built using PyTorch and PyQt5 for reinforcement learning research and education.
