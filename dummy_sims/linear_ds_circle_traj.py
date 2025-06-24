import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import casadi as ca

# Parameters
dt = 0.05
steps = 300
alpha = 0.8
R = 10.0  # Radius of circular trajectory
ref_lookahead = 2.0  # Arc length to project forward
obstacle_center = np.array([0.0, 10.5])
obstacle_radius = 1.0
safety_margin = 0.5
sigma = 0.01  # Noise standard deviation

# Initial state
state = np.array([10.0, 0.0])
trajectory = []
ref_trajectory = []

# First-order CBF QP solver
def cbf_qp_control(state, u_nom, obstacle_center, obstacle_radius, safety_margin=0.5, gamma=1.0):
    x, y = state
    x_o, y_o = obstacle_center
    r_s = obstacle_radius + safety_margin

    dx = x - x_o
    dy = y - y_o
    h = dx**2 + dy**2 - r_s**2

    # Gradient of h
    grad_h = np.array([2 * dx, 2 * dy])
    grad_h = grad_h.reshape(1, 2)  # Row vector

    # Define optimization variables
    u = ca.SX.sym("u", 2)

    # Objective: minimize deviation from nominal
    obj = ca.sumsqr(u - u_nom)

    # Constraint: ∇h · u + γ h ≥ 0
    cbf_constraint = ca.mtimes(grad_h, u)[0] + gamma * h

    nlp = {"x": u, "f": obj, "g": cbf_constraint}
    solver = ca.nlpsol("solver", "ipopt", nlp, {
        "ipopt.print_level": 0,
        "print_time": 0,
        "ipopt.tol": 1e-6
    })

    try:
        sol = solver(lbg=0, ubg=ca.inf)
        u_safe = np.array(sol["x"].full()).flatten()
        return u_safe
    except RuntimeError:
        return u_nom  # fallback

# Circular trajectory tracking helpers
def closest_point_on_circle(state, R):
    x, y = state
    theta = np.arctan2(y, x)
    return R * np.array([np.cos(theta), np.sin(theta)]), theta

def project_reference(theta, arc_length, R):
    dtheta = arc_length / R
    theta_ref = theta + dtheta
    return R * np.array([np.cos(theta_ref), np.sin(theta_ref)])

# Simulation loop
for _ in range(steps):
    closest_point, theta = closest_point_on_circle(state, R)
    ref_point = project_reference(theta, ref_lookahead, R)

    # Nominal control law: track reference point
    u_nom = -alpha * (state - ref_point)

    # Apply CBF QP
    u_safe = cbf_qp_control(state, u_nom, obstacle_center, obstacle_radius, safety_margin)

    # Euler integration with small noise
    state = state + u_safe * dt + np.random.normal(0, sigma, size=state.shape)
    trajectory.append(state.copy())
    ref_trajectory.append(ref_point.copy())

trajectory = np.array(trajectory)
ref_trajectory = np.array(ref_trajectory)

# Plotting
fig, ax = plt.subplots()
ax.set_aspect('equal')
ax.set_xlim(-12, 12)
ax.set_ylim(-12, 12)

# Obstacle
obs = plt.Circle(obstacle_center, obstacle_radius, color='red', alpha=0.6, label="Obstacle")
ax.add_patch(obs)

# Safety buffer (for visualization)
obs_buffer = plt.Circle(obstacle_center, obstacle_radius + safety_margin, color='red', alpha=0.2, linestyle='--', fill=False)
ax.add_patch(obs_buffer)

# Desired circular path
circle = plt.Circle((0, 0), R, color='gray', linestyle='--', fill=False, label='Desired Path')
ax.add_patch(circle)

# Plot handles
point, = ax.plot([], [], 'bo', label="Robot")
trail, = ax.plot([], [], 'b-', lw=1, label="Path")
ref_dot, = ax.plot([], [], 'gx', label="Ref Point")  # Green X

def update(frame):
    point.set_data(trajectory[frame, 0], trajectory[frame, 1])
    trail.set_data(trajectory[:frame+1, 0], trajectory[:frame+1, 1])
    ref_dot.set_data(ref_trajectory[frame, 0], ref_trajectory[frame, 1])
    return point, trail, ref_dot

ani = FuncAnimation(fig, update, frames=len(trajectory), interval=dt * 1000, blit=True)
plt.title("Circular Path Tracking with Obstacle Avoidance via 1st-order CBF")
plt.legend()
plt.grid(True)
plt.show()
