import torch
import torch.nn as nn
import torch.nn.functional as F

class SnakeNetLarge(nn.Module):
    def __init__(self, input_size=57, hidden_size=128, output_size=3):
        """
        Snake neural network for 7x7 grid
        Input: 7x7 grid (49) + 3 apples (6 values) + opponent head (2 values) = 57
        Grid encoding: 0=empty, -1=your body, -2=opponent body, -3=opponent head, 1=apple
        Output: 3 actions (forward, left turn, right turn)
        """
        super(SnakeNetLarge, self).__init__()
        
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x