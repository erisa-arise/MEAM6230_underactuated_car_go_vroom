import torch.nn as nn
import torch.nn.functional as F
import torch

class N_ODE(nn.Module):
    def __init__(self, input_dim=4, out_dim=2, hidden_dim=32):
        super(N_ODE, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, out_dim)

    def forward(self, x):
        #putting theta into nn is bad, use sin and cos of theta so its smooth
        with torch.no_grad():
            x = torch.cat([x[:, :2], torch.cos(x[:, 2]).reshape(-1, 1), torch.sin(x[:, 2]).reshape(-1, 1)], dim=1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x