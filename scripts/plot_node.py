import torch
import numpy as np
import matplotlib.pyplot as plt
from train_node import NeuralODEFunc

# === Load model ===
model = NeuralODEFunc()
model.load_state_dict(torch.load("node_racecar_model.pth"))
model.eval()

# === Define grid over state space (x, y) ===
x_range = np.linspace(-20, 20, 30)
y_range = np.linspace(-20, 20, 30)
Xg, Yg = np.meshgrid(x_range, y_range)

# Fixed yaw value for visualization
yaw_fixed = 0.0

# === Evaluate model on grid ===
U = np.zeros_like(Xg)
V = np.zeros_like(Yg)

with torch.no_grad():
    for i in range(Xg.shape[0]):
        for j in range(Xg.shape[1]):
            x_input = torch.tensor([Xg[i, j], Yg[i, j], yaw_fixed], dtype=torch.float32)
            x_dot = model(x_input)
            U[i, j] = x_dot[0].item()  # dx/dt
            V[i, j] = x_dot[1].item()  # dy/dt

# === Plot vector field ===
plt.figure(figsize=(8, 6))
plt.quiver(Xg, Yg, U, V, color='blue', scale=100, headwidth=3)
plt.title(f"Learned Vector Field at Yaw = {yaw_fixed:.2f} rad")
plt.xlabel("x [m]")
plt.ylabel("y [m]")
plt.axis("equal")
plt.grid(True)
plt.tight_layout()
plt.show()