import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation

class GenerateDataset():
    def __init__(self):
        # define an elliptical track
        self.a = 20.0
        self.b = 8.0
        self.num_laps = 1

        # define car parameters
        self.L = 2.5
        self.max_v = 6.0
        self.min_v = -1.0
        self.delta_max = np.pi / 4
        self.vel_std = 0.0
        self.angle_std = 0.0

        # define simulation parameters
        self.total_time = 100.0
        self.dt = 0.1

        # trajectory tracking parameters
        self.v = 3.0
        self.lookahead_time = 1.0
        self.lookahead_steps = int(self.lookahead_time / self.dt)

        # PD controller parameters
        self.Kp = 3.0
        self.Kd = 8
        
    def simiulation_rollout(self):
        t = np.arange(0, self.total_time, self.dt)
        N = len(t)

        x = np.zeros(N)
        y = np.zeros(N)
        yaw = np.zeros(N)
        errors = np.zeros(N)
        x_refs = np.zeros(N)
        y_refs = np.zeros(N)

        # set initial state
        x[0], y[0], yaw[0] = self.a, 0, np.pi / 2
        prev_error = 0.0

        # Updated simulation loop with lookahead point
        for i in range(1, N):
            theta = (i + self.lookahead_steps) * self.v * self.dt / (0.5 * (self.a + self.b))

            x_ref = self.a * np.cos(theta)
            y_ref = self.b * np.sin(theta)
            x_refs[i] = x_ref
            y_refs[i] = y_ref

            dx = x_ref - x[i - 1]
            dy = y_ref - y[i - 1]
            alpha = np.arctan2(dy, dx) - yaw[i - 1] 
            alpha = np.arctan2(np.sin(alpha), np.cos(alpha)) 

            Ld = np.sqrt(dx**2 + dy**2)
            delta = np.arctan2(2 * self.L * np.sin(alpha), Ld) + np.random.normal(0, self.angle_std)
            delta = np.clip(delta, -self.delta_max, self.delta_max)  # limit steering angle

            # Calculate the arc length error (distance to reference point)
            error = Ld - 1.0 
            d_error = (error - prev_error)
            prev_error = error 

            # PD controller to adjust speed based on the error
            v_modulated = self.Kp * error + self.Kd * d_error
            v_modulated = np.clip(v_modulated, self.min_v, self.max_v)  

            # Update state with Ackermann dynamics using modulated speed
            x[i] = x[i - 1] + v_modulated * np.cos(yaw[i - 1]) * self.dt
            y[i] = y[i - 1] + v_modulated * np.sin(yaw[i - 1]) * self.dt
            yaw[i] = yaw[i - 1] + v_modulated / self.L * np.tan(delta) * self.dt
            errors[i] = error

        x_dot = np.gradient(x, self.dt)
        y_dot = np.gradient(y, self.dt)
        yaw_dot = np.gradient(yaw, self.dt)

        X = np.vstack((x, y, yaw)).T
        X_dot = np.vstack((x_dot, y_dot, yaw_dot)).T
        ref_points = np.vstack((x_refs, y_refs)).T

        return X, X_dot, t, ref_points, errors
        
    def plot_trajectory(self, X):
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.plot(X[:, 0], X[:, 1], label="Ackermann on Elliptical Track")
        ellipse = plt.matplotlib.patches.Ellipse((0, 0), 2 * self.a, 2 * self.b,
                                                edgecolor='gray', facecolor='none',
                                                linestyle='--', label='Ideal Elliptical Track')
        ax.add_patch(ellipse)
        ax.set_aspect('equal')
        ax.set_title("Car Trajectory on Elliptical Track with Lookahead Steering")
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
        ax.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    def plot_errors(self, errors):
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(errors, label="Arc Length Error", color='orange')
        ax.set_title("Arc Length Error Over Time")
        ax.set_xlabel("Time Steps")
        ax.set_ylabel("Error [m]")
        ax.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    def animate_trajectory(self, X, ref_points, interval=50):
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.set_xlim(-self.a * 1.2, self.a * 1.2)
        ax.set_ylim(-self.b * 1.5, self.b * 1.5)
        ax.set_aspect('equal')
        ax.set_title("Ackermann Steering Simulation")
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
        ax.grid(True)

        # Elliptical guide
        ellipse = plt.matplotlib.patches.Ellipse((0, 0), 2 * self.a, 2 * self.b,
                                                edgecolor='gray', facecolor='none', linestyle='--')
        ax.add_patch(ellipse)

        # Plots
        path_line, = ax.plot([], [], 'b-', lw=1, label="Trajectory")
        car_dot, = ax.plot([], [], 'ro', label="Car")
        heading_arrow = ax.quiver([], [], [], [], color='r', scale=20)
        pursuit_dot, = ax.plot([], [], 'go', label="Pursuit Point")

        ax.legend()

        def init():
            path_line.set_data([], [])
            car_dot.set_data([], [])
            pursuit_dot.set_data([], [])
            return path_line, car_dot, heading_arrow, pursuit_dot

        def update(i):
            path_line.set_data(X[:i+1, 0], X[:i+1, 1])
            car_dot.set_data(X[i, 0], X[i, 1])
            pursuit_dot.set_data(ref_points[i, 0], ref_points[i, 1])

            # Update heading arrow
            dx = np.cos(X[i, 2])
            dy = np.sin(X[i, 2])
            heading_arrow.set_offsets([X[i, 0], X[i, 1]])
            heading_arrow.set_UVC(dx, dy)

            return path_line, car_dot, heading_arrow, pursuit_dot

        ani = animation.FuncAnimation(fig, update, frames=len(X), init_func=init,
                                    blit=False, interval=interval)
        plt.show()

dataset_generator = GenerateDataset()
X, X_dot, t, ref_points, errors = dataset_generator.simiulation_rollout()
np.savez("racecar_elliptical_dataset.npz", X=X, X_dot=X_dot, t=t)

# dataset_generator.plot_errors(errors)
# dataset_generator.plot_trajectory(X)
dataset_generator.animate_trajectory(X, ref_points)