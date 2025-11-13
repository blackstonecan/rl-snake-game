import torch
import torch.nn as nn
import torch.nn.functional as F

class SnakeNet(nn.Module):
    def __init__(self, input_size=31, hidden_size=128, output_size=3):
        """
        Snake neural network
        Input: 5x5 grid (25) + 3 apples (6 values: 3x(dx, dy)) = 31
        Grid encoding: -1=snake body, 0=empty, 1=apple
        Output: 3 actions (forward, left turn, right turn)
        """
        super(SnakeNet, self).__init__()
        
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x