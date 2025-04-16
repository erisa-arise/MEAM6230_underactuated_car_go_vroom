import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from tqdm import tqdm

# === Load dataset ===
data = np.load("racecar_elliptical_dataset.npz")
X = torch.tensor(data["X"], dtype=torch.float32)
X_dot = torch.tensor(data["X_dot"], dtype=torch.float32)

# === Define NODE dynamics model ===
class NeuralODEFunc(nn.Module):
    def __init__(self, dim=3, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, dim)
        )

    def forward(self, x):
        return self.net(x)

# === Training parameters ===
batch_size = 64
lr = 1e-3
epochs = 200

dataset = TensorDataset(X, X_dot)
loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

model = NeuralODEFunc()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
criterion = nn.MSELoss()

# === Training loop with tqdm ===
losses = []
for epoch in tqdm(range(epochs), desc="Training Epochs"):
    epoch_loss = 0.0
    for x_batch, x_dot_batch in loader:
        optimizer.zero_grad()
        pred_dot = model(x_batch)
        loss = criterion(pred_dot, x_dot_batch)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item() * x_batch.size(0)
    epoch_loss /= len(dataset)
    losses.append(epoch_loss)

# === Save model ===
torch.save(model.state_dict(), "node_racecar_model.pth")
print("✅ Model saved to node_racecar_model.pth")

# === Plot training loss ===
plt.plot(losses)
plt.title("Training Loss")
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.grid(True)
plt.tight_layout()
plt.show()
