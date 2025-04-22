import torch.nn as nn
import torch.nn.functional as F

class N_ODE(nn.Module):
    def __init__(self, input_dim=3, out_dim=2, hidden_dim=64):
        super(N_ODE, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, out_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x