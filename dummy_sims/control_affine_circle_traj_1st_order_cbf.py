import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import casadi as ca

# Parameters
dt = 0.05
steps = 400
R = 10.0  # Circular path radius
ref_lookahead = 1.5
obstacle_center = np.array([0.0, 10.5])
obstacle_radius = 1.0
safety_margin = 0.5
v_max = 1.5
omega_max = 2.5

# Noise parameters
sigma_pos = 0.01
sigma_theta = 0.01

# PID gains
kp_pos = 1.5
kp_theta = 2.0

# Initial state [x, y, theta]
state = np.array([10.0, 0.0, np.pi / 2])
trajectory = []
ref_trajectory = []

def control_to_state_derivative(control):
    v, omega = control
    x_dot = v * np.cos(state[2])
    y_dot = v * np.sin(state[2])
    theta_dot = omega
    return np.array([[x_dot, y_dot, theta_dot]]).T

# First-order CBF QP solver for Dubins vehicle
def cbf_qp_control(state, u_nom, obstacle_center, obstacle_radius, safety_margin=0.5, gamma=1.0):
    x, y, theta = state
    x_o, y_o = obstacle_center
    r_s = obstacle_radius + safety_margin
    x_dot_nom = control_to_state_derivative(u_nom)

    dx = x - x_o
    dy = y - y_o
    h = dx**2 + dy**2 - r_s**2
    dh_dx = ca.vertcat(2 * dx, 2 * dy, 0)  # ∇h = [∂h/∂x, ∂h/∂y, ∂h/∂theta]

    u = ca.MX.sym("u", 3)  # Control input x_dot, y_dot, omega
    control = ca.MX.sym("control", 2)  # Control input v, omega

    Q = ca.diag([1/v_max, 1/v_max, 1/omega_max])
    obj = ca.mtimes([u.T, Q, u])

    constraints = []

    # Input constraints
    constraints.append(control)
    actuation_lower_bounds = [-v_max, -omega_max]
    actuation_upper_bounds = [v_max, omega_max]

    # CBF constraint: ∇h · (u + u_nom) + γ h ≥ 0
    cbf_constraint = ca.dot(dh_dx, u + x_dot_nom) + gamma * h
    constraints.append(cbf_constraint)
    cbf_lower_bounds = [0]
    cbf_upper_bounds = [ca.inf]

    # Dynamics Constraint: u = [v * cos(theta), v * sin(theta), omega]
    constraints.append(x_dot_nom[0] + u[0] - control[0] * ca.cos(theta))
    constraints.append(x_dot_nom[1] + u[1] - control[0] * ca.sin(theta))
    constraints.append(x_dot_nom[2] + u[2] - control[1])
    dynamics_lower_bounds = [0, 0, 0]
    dynamics_upper_bounds = [0, 0, 0]

    nlp = {"x": ca.vertcat(u, control), "f": obj, "g": ca.vertcat(*constraints)}
    solver = ca.nlpsol("solver", "ipopt", nlp, {
        "ipopt.print_level": 0,
        "print_time": 0,
        "ipopt.tol": 1e-6
    })

    lower_bounds = ca.vertcat(actuation_lower_bounds, cbf_lower_bounds, dynamics_lower_bounds)
    upper_bounds = ca.vertcat(actuation_upper_bounds, cbf_upper_bounds, dynamics_upper_bounds)

    try:
        sol = solver(lbg=lower_bounds, ubg=upper_bounds)
        u_safe = np.array(sol["x"].full()).flatten()
        control_safe = u_safe[3:]
        return control_safe
    except RuntimeError:
        print("QP solver failed, using nominal control")
        return u_nom  # fallback

# Circle tracking helpers
def closest_point_on_circle(state, R):
    x, y = state[:2]
    theta = np.arctan2(y, x)
    return R * np.array([np.cos(theta), np.sin(theta)]), theta

def project_reference(theta, arc_length, R):
    dtheta = arc_length / R
    theta_ref = theta + dtheta
    return R * np.array([np.cos(theta_ref), np.sin(theta_ref)]), theta_ref

# PID nominal control for Dubins vehicle
def compute_nominal_control(state, ref_point):
    x, y, theta = state
    x_ref, y_ref = ref_point

    # Position Error
    dx = x_ref - x
    dy = y_ref - y
    pos_error = np.hypot(dx, dy)

    # Heading Error
    heading_desired = np.arctan2(dy, dx)
    heading_error = np.arctan2(np.sin(heading_desired - theta), np.cos(heading_desired - theta))

    # Feedback Control
    v = kp_pos * pos_error
    omega = kp_theta * heading_error
    return np.array([v, omega])

# Simulation loop
for _ in range(steps):
    closest_point, theta_closest = closest_point_on_circle(state, R)
    ref_point, theta_ref = project_reference(theta_closest, ref_lookahead, R)

    u_nom = compute_nominal_control(state, ref_point)
    u_safe = cbf_qp_control(state, u_nom, obstacle_center, obstacle_radius, safety_margin)

    v, omega = u_safe
    x, y, theta = state

    # Euler integration of Dubins dynamics
    x += v * np.cos(theta) * dt + np.random.normal(0, sigma_pos)
    y += v * np.sin(theta) * dt + np.random.normal(0, sigma_pos)
    theta += omega * dt + np.random.normal(0, sigma_theta)
    theta = (theta + np.pi) % (2 * np.pi) - np.pi 

    state = np.array([x, y, theta])
    trajectory.append(state[:2].copy())
    ref_trajectory.append(ref_point.copy())

trajectory = np.array(trajectory)
ref_trajectory = np.array(ref_trajectory)

# Plotting
fig, ax = plt.subplots()
ax.set_aspect('equal')
ax.set_xlim(-12, 12)
ax.set_ylim(-12, 12)

# Obstacle and safety margin
obs = plt.Circle(obstacle_center, obstacle_radius, color='red', alpha=0.6, label="Obstacle")
ax.add_patch(obs)
obs_buffer = plt.Circle(obstacle_center, obstacle_radius + safety_margin, color='red', alpha=0.2, linestyle='--', fill=False)
ax.add_patch(obs_buffer)

# Desired circular path
circle = plt.Circle((0, 0), R, color='gray', linestyle='--', fill=False, label='Desired Path')
ax.add_patch(circle)

# Plot handles
point, = ax.plot([], [], 'bo', label="Robot")
trail, = ax.plot([], [], 'b-', lw=1, label="Path")
ref_dot, = ax.plot([], [], 'gx', label="Ref Point") 

def update(frame):
    point.set_data(trajectory[frame, 0], trajectory[frame, 1])
    trail.set_data(trajectory[:frame+1, 0], trajectory[:frame+1, 1])
    ref_dot.set_data(ref_trajectory[frame, 0], ref_trajectory[frame, 1])
    return point, trail, ref_dot

ani = FuncAnimation(fig, update, frames=len(trajectory), interval=dt * 1000, blit=True)
plt.title("Control Affine Circle Trajectory with 1st Order CBF")
plt.legend()
plt.grid(True)
plt.show()
