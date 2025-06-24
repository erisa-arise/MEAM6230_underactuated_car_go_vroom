import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import casadi as ca

# Parameters
dt = 0.05
steps = 300
R = 10.0  # Circular path radius
ref_lookahead = 1.0
v_ref = 1.0
obstacle_center = np.array([0.0, 10.5])
obstacle_radius = 1.0
safety_margin = 0.5

# Noise parameters
sigma_pos = 0.01
sigma_vel = 0.01
sigma_theta = 0.01

# PID gains
kp_pos = 2.0
kp_v = 1.0
kp_theta = 2.0

# Initial state [x, y, v, theta]
state = np.array([10.0, 0.0, 0.0, np.pi / 2])
trajectory = []
ref_trajectory = []

# First-order CBF QP solver for an extended Dubins vehicle
def hocbf_qp_control(state, u_nom, obstacle_center, obstacle_radius, safety_margin=0.5, gamma1=1.8, gamma2=1.0):
    x, y, v, theta = state
    x_o, y_o = obstacle_center
    r_s = obstacle_radius + safety_margin

    dx = x - x_o
    dy = y - y_o
    h = dx**2 + dy**2 - r_s**2
    dh_dx = ca.vertcat(2 * dx, 2 * dy, 0)  # ∇h = [∂h/∂x, ∂h/∂y, ∂h/∂theta]

    a = ca.SX.sym("a")
    omega = ca.SX.sym("omega")
    u = ca.vertcat(a, omega)

    dx_dt = ca.vertcat(v * ca.cos(theta), v * ca.sin(theta), omega)
    dh_dt = ca.dot(dh_dx, dx_dt)

    d2h_dt2 = 2*a*(dx*np.cos(theta) + dy*np.sin(theta)) + 2*v**2 + 2*v*(-dx*np.sin(theta) + dy*np.cos(theta))*omega

    obj = ca.sumsqr(u - u_nom)

    hocbf_constraint = d2h_dt2 + gamma1 * dh_dt + gamma2 * h

    obj = ca.sumsqr(u - u_nom)

    nlp = {"x": u, "f": obj, "g": hocbf_constraint}
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

# Circle tracking helpers
def closest_point_on_circle(state, R):
    x, y = state[:2]
    theta = np.arctan2(y, x)
    return R * np.array([np.cos(theta), np.sin(theta)]), theta

def project_reference(theta, arc_length, R):
    dtheta = arc_length / R
    theta_ref = theta + dtheta
    return R * np.array([np.cos(theta_ref), np.sin(theta_ref)]), theta_ref

# PID nominal control for extended Dubins vehicle
def compute_nominal_control(state, ref_point):
    x, y, v, theta = state
    x_ref, y_ref = ref_point

    # --- Position Error ---
    dx = x_ref - x
    dy = y_ref - y
    d_pos = np.hypot(dx, dy)

    # Desired heading to reference point
    heading_desired = np.arctan2(dy, dx)
    heading_error = np.arctan2(np.sin(heading_desired - theta), np.cos(heading_desired - theta))

    # --- Velocity Control (projected onto tangent direction) ---
    dv = v_ref - v

    # --- Control Inputs ---
    a = kp_v * dv + kp_pos * d_pos 

    omega = kp_theta * heading_error
    return np.array([a, omega])

# Simulation loop
for _ in range(steps):
    closest_point, theta_closest = closest_point_on_circle(state, R)
    ref_point, theta_ref = project_reference(theta_closest, ref_lookahead, R)

    u_nom = compute_nominal_control(state, ref_point, theta_ref)
    u_safe = hocbf_qp_control(state, u_nom, obstacle_center, obstacle_radius, safety_margin)

    a, omega = u_safe
    x, y, v, theta = state

    # Euler integration of Dubins dynamics
    x += v * np.cos(theta) * dt + np.random.normal(0, sigma_pos)
    y += v * np.sin(theta) * dt + np.random.normal(0, sigma_pos)
    v += a * dt + np.random.normal(0, sigma_vel)
    theta += omega * dt + np.random.normal(0, sigma_theta)
    theta = (theta + np.pi) % (2 * np.pi) - np.pi  # Normalize

    state = np.array([x, y, v, theta])
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
ref_dot, = ax.plot([], [], 'gx', label="Ref Point")  # Green X

def update(frame):
    point.set_data(trajectory[frame, 0], trajectory[frame, 1])
    trail.set_data(trajectory[:frame+1, 0], trajectory[:frame+1, 1])
    ref_dot.set_data(ref_trajectory[frame, 0], ref_trajectory[frame, 1])
    return point, trail, ref_dot

ani = FuncAnimation(fig, update, frames=len(trajectory), interval=dt * 1000, blit=True)
plt.title("Extended Control Affine Circle Trajectory with 1nd Order CBF")
plt.legend()
plt.grid(True)
plt.show()
