# Snake Game AI - Multiplayer Battle Arena

A reinforcement learning project featuring competitive AI agents that
learn to play multiplayer snake using Deep Q-Learning (DQN). Watch AI
snakes battle it out in a PyQt5 GUI environment!

## Features

-   **Multiplayer Snake Game**: Classic snake game with 2-player
    competitive gameplay
-   **Deep Reinforcement Learning**: AI agents trained using DQN with
    experience replay
-   **Multiple Agent Types**:
    -   **Main Agent**: Balanced agent trained with curriculum learning
        and self-play
    -   **Killer Agent**: Aggressive agent focused on eliminating
        opponents
    -   **Defender Agent**: Defensive agent focused on survival and
        apple collection
-   **Interactive GUI**: PyQt5-based interface to watch AI battles
-   **Curriculum Learning**
-   **Self-Play Training**

## Project Structure

    ./
    ├── gui/
    │   ├── snake_game_multiplayer_gui.py
    │   └── __init__.py
    ├── game/
    │   ├── snake_game_multiplayer.py
    │   └── __init__.py
    ├── model/
    │   ├── agent_middleware_large.py
    │   ├── agent_middleware_structure.py
    │   ├── snake_model_large.py
    │   └── __init__.py
    ├── agents/
    │   ├── best_main_agent.pth
    │   ├── best_killer_agent.pth
    │   ├── best_defender_agent.pth
    │   ├── main_agent.pth
    │   ├── killer_agent.pth
    │   └── defender_agent.pth
    ├── train/
    │   ├── train_main_agent.py
    │   ├── train_killer.py
    │   ├── train_defender.py
    │   └── __init__.py
    ├── test/
    │   ├── battle_test.py
    │   └── __init__.py
    └── requirements.txt

## Installation

### Prerequisites

-   Python 3.8+
-   PyTorch
-   PyQt5

### Setup

``` bash
pip install -r requirements.txt
```

## Usage

### Watch AI Battles (GUI)

Run from project root:

``` bash
python -m gui.snake_game_multiplayer_gui
```

### Run Automated Battles

``` bash
python -m test.battle_test killer_agent.pth main_agent.pth -n 100
```

### Training

``` bash
python -m train.train_main_agent
python -m train.train_killer
python -m train.train_defender
```

## Model Files

Models are in `agents/`.

## Troubleshooting

-   If GUI shows no models → ensure `.pth` files are in `agents/`.
-   If running as module, always start from project root.