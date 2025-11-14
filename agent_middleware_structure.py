import torch
import numpy as np
from snake_model_large import SnakeNetLarge
import sys
import os
from enum import Enum

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

"""
This module defines the AgentMiddlewareLarge class which serves as an interface
between the Snake game environment and a large neural network model for decision making.
"""

class Direction(Enum):
    UP = (0, -1)
    DOWN = (0, 1)
    LEFT = (-1, 0)
    RIGHT = (1, 0)

class AgentMiddlewareLarge:
    def __init__(self, model_path=None):
        pass

    def get_action(self, game_state, opponent_state=None, epsilon=0.0):
        """
        Get action from model
        epsilon: exploration rate (0 = always exploit, 1 = always explore)
        Returns: Direction enum
        """
        pass