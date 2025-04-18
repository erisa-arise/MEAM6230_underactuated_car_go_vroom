import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from n_ode import N_ODE
import torch.nn.functional as F
import matplotlib.pyplot as plt
import os

class node_dataset(Dataset):
    def __init__(self, x_path, y_path):
        self.X = torch.tensor(np.load(x_path), dtype=torch.float32)
        self.Y = torch.tensor(np.load(y_path), dtype=torch.float32)
    
    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]

def train(ds, epochs=10, batch_size = 32):
    dataloader = DataLoader(ds, batch_size=batch_size, shuffle=True)
    model = N_ODE()
    optim = torch.optim.Adam(model.parameters(), lr=0.003)

    epoch_avg_losses = []
    for i in range(epochs):
        loss_sum = 0
        num_batches = 0
        for j, (x_batch, y_batch) in enumerate(dataloader):
            if i==0 and j==0:
                print("First batch X: ")
                print(x_batch)
                print("First batch Y: ")
                print(y_batch)
            num_batches += 1
            pred = model(x_batch)
            loss = F.mse_loss(pred, y_batch)

            loss_sum += loss.item()

            optim.zero_grad()

            loss.backward()

            optim.step()
        loss_avg = loss_sum / num_batches
        epoch_avg_losses.append(loss_avg)
    plt.plot(epoch_avg_losses)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.yscale('log')
    plt.show()
    
    

if __name__ == "__main__":
    filepath = os.path.dirname(os.path.abspath(__file__))
    x_path = f"{filepath}/../data/state_history.npy"
    y_path = f"{filepath}/../data/control_history.npy"

    ds = node_dataset(x_path, y_path)

    train(ds)