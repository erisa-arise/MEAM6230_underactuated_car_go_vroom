#!/usr/bin/env python3

import sys
import os
import rclpy
import torch
import math
import casadi
import numpy as np

from casadi import MX, SX, DM, Function

from rclpy.node import Node
from rclpy.publisher import Publisher
from rclpy.subscription import Subscription

from geometry_msgs.msg import Quaternion, Vector3, PoseStamped
from ackermann_msgs.msg import AckermannDriveStamped
from mocap4r2_msgs.msg import RigidBodies
from nav_msgs.msg import Odometry, Path

from reactive_car.srv import GenerateNominalTrajectory
from utils.n_ode import N_ODE
from typing import Tuple, List


class NeuralODEController(Node):
    def __init__(self) -> None:
        super().__init__('Neural_ODE_Controller')
        self.srv = self.create_service(GenerateNominalTrajectory,'generate_nominal_trajectory',self.generate_nominal_trajectory_callback)
        # self.odom_subscriber: Subscription = self.create_subscription(RigidBodies, '/odom_topic', self.odom_callback, 10)
        self.odom_subscriber: Subscription = self.create_subscription(Odometry, '/ego_racecar/odom', self.odom_callback, 10)
        self.ackermann_publisher: Publisher = self.create_publisher(AckermannDriveStamped, '/ackermann_cmd', 10)
        self.nominal_trajectory_publisher: Publisher = self.create_publisher(Path, '/nominal_trajectory', 10)

        self.latest_position: Vector3 | None = None
        self.latest_quaternion: Quaternion | None = None

        # load the neural ODE model here
        self.file_path = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(self.file_path, 'utils', 'n_ode_model.pth')
        self.model: N_ODE = N_ODE()
        self.model.load_state_dict(torch.load(model_path))  
        self.model.eval()

        # car parameters
        self.L: float = 1.0
        self.v_max: float = 2.0
        self.delta_max: float = casadi.pi/4

        # cost parameter
        self.lambda_: float = 1.0         
        
        # control barrier function parameters
        self.a: float = 3.5
        self.b: float = 2.5
        self.gamma: float = 1.0
        self.ellipse_center: Tuple[float, float] = (0.5, 0.0)

        # control lyapunov function parameters
        self.alpha: float = 1.0
        self.lookahead_index: int = 5

        # trajectory rollout parameters
        self.dt: float = 0.01
        self.rollout_length: int = 100
        self.nominal_trajectory: np.ndarray | None = None

        self.get_logger().info('Initialized NODE Controller')

    def generate_nominal_trajectory_callback(self, request, response):
        """
        Callback function for the GenerateNominalTrajectory service. Generates a nominal trajectory 
        using the neural ODE model.
        """
        state_0: np.ndarray = np.array([[self.latest_position.x, self.latest_position.y, self.get_yaw_from_quaternion(self.latest_quaternion)]]).T
        self.get_logger().info(f'Generating nominal trajectory with initial state: {state_0}')
        self.rollout_nominal_trajectory(state_0)

        response.success = True
        response.message = "Generated Nominal Trajectory"
        self.get_logger().info('Service called: returning nominal trajectory generation success.')
        return response
    
    def rollout_nominal_trajectory(self, state_0: np.ndarray) -> None:
        """
        Computes the nominal trajectory using the neural ODE model.
        
        Args:
            state_0 (np.ndarray): The initial state of the car in the form [x, y, theta]
        Returns: 
            None
        """
        self.nominal_trajectory = np.zeros((3, self.rollout_length), dtype=np.float32)

        current_state: torch.Tensor = torch.tensor(state_0.T, dtype=torch.float32)
        for i in range(self.rollout_length):
            with torch.no_grad():
                n_ode_output: torch.Tensor = self.model(current_state)
            x_dot: np.ndarray = self.map_n_ode_to_x_dot(n_ode_output)
            current_state = (current_state + x_dot.T * self.dt).float()
            self.nominal_trajectory[:, i] = current_state.squeeze().numpy()

        # publish nominal trajectory
        path_msg: Path = Path()
        path_msg.header.frame_id = 'map'
        path_msg.header.stamp = self.get_clock().now().to_msg()
        for i in range(self.rollout_length):
            pose: PoseStamped = PoseStamped()
            pose.header.frame_id = 'map'
            pose.header.stamp = self.get_clock().now().to_msg()
            pose.pose.position.x = float(self.nominal_trajectory[0, i])
            pose.pose.position.y = float(self.nominal_trajectory[1, i])
            pose.pose.position.z = 0.0
            pose.pose.orientation.x = 0.0
            pose.pose.orientation.y = 0.0
            pose.pose.orientation.z = 0.0
            pose.pose.orientation.w = 1.0
            path_msg.poses.append(pose)
        self.nominal_trajectory_publisher.publish(path_msg)
        self.get_logger().info('Published nominal trajectory')

    def map_n_ode_to_x_dot(self, n_ode_output: torch.Tensor) -> np.ndarray:
        """
        Converts the Neural ODE output to x_dot using the car dynamics model.

        Args:
            n_ode_output (torch.Tensor): The output of the Neural ODE model
        Returns:
            x_dot (np.ndarray): The x_dot vector in the form [x_dot, y_dot, theta_dot]
        """
        v: float = n_ode_output[0][0].item()
        delta: float = n_ode_output[0][1].item()
        x_dot: np.ndarray = np.array([[v * math.cos(delta), v * math.sin(delta), v/self.L * math.tan(delta)]]).T
        return x_dot

    def odom_callback(self, msg: RigidBodies) -> None:
        """
        Callback function for the VICON odometry subscriber. Publishes the safe control command to the car.
        
        Args:
            msg (RigidBodies): The VICON odometry message
        Returns:
            None
        """
        self.latest_odom = msg

        # extract orientation and position from the VICON ROS2 message
        # self.latest_quaternion = msg.rigidbodies.pose.orientation
        # self.latest_position = msg.rigidbodies.pose.position
        # yaw = self.get_yaw_from_quaternion(self.latest_quaternion)

        # extract orientation and position from the Sim Odometry message
        self.latest_quaternion = msg.pose.pose.orientation
        self.latest_position = msg.pose.pose.position
        yaw = self.get_yaw_from_quaternion(self.latest_quaternion)

        if self.nominal_trajectory is None:
            self.get_logger().warn('Nominal trajectory not set. Skipping control.')
            return

        # create the state tensor and query the Neural ODE
        state: np.ndarray = np.array([[self.latest_position.x, self.latest_position.y, yaw]]).T

        # compute the safe control
        v_safe, delta_safe = self.compute_safe_control(state) 
        
        # publish the safe velocitya and steering angle
        ackermann_msg: AckermannDriveStamped = AckermannDriveStamped()
        ackermann_msg.drive.speed = v_safe
        ackermann_msg.drive.steering_angle = delta_safe
        self.ackermann_publisher.publish(ackermann_msg)    

    def compute_safe_control(self, state: np.ndarray) -> Tuple[float]:
        """
        Gets candidate trackpoints and chooses the smallest residual control.

        Args:
            state (np.ndarray): The current state of the car in the form [x, y, theta]
        Returns:
            v_safe (float): The safe velocity
            delta_safe (float): The safe steering angle
        """
        # determine the safe control input by scanning across the next self.lookahead_index points on the track
        track_points: np.ndarray = self.calculate_track_points(state)

        state_tensor: torch.Tensor = torch.tensor(state.T, dtype=torch.float32)
        n_ode_output = self.model(state_tensor)
        state_xdot: np.ndarray = self.map_n_ode_to_x_dot(n_ode_output)

        controls: List[Tuple[float]] = []
        for i in range(self.lookahead_index):
            track_point = track_points[:, i:i+1]
            track_point_velocity: torch.Tensor = self.model(torch.tensor(track_point.T, dtype=torch.float32))
            track_point_xdot: np.ndarray = self.map_n_ode_to_x_dot(track_point_velocity)
            error_state: np.ndarray = state - track_point
            u_opt, control_opt, epsilon_opt = self.solve_control_optimization(state, error_state, track_point_xdot, state_xdot)
            residual_control_norm = (u_opt[0]/self.v_max)**2 + (u_opt[1]/self.v_max)**2 + (u_opt[2]/self.delta_max)**2
            controls.append((control_opt[0], control_opt[1], residual_control_norm))

        # choose the smallest residual control
        v_safe, delta_safe, residual_control_norm_safe = min(controls, key=lambda x: x[2])

        return v_safe, delta_safe
    
    def calculate_track_points(self, state: np.ndarray) -> np.ndarray:
        """
        Computes the next self.lookahead_index points on the track.
        
        Args:
            state (np.ndarray): The current state of the car in the form [x, y, theta]
        Returns:
            track_points (np.ndarray): The next self.lookahead_index points on the track
        """
        closest_index: int = np.argmin(np.linalg.norm(self.nominal_trajectory - state, axis=0))
        track_points: np.ndarray = np.zeros((3, self.lookahead_index))
        for i in range(self.lookahead_index):
            track_points[:, i] = self.nominal_trajectory[:, (closest_index + i) % self.nominal_trajectory.shape[1]]
        return track_points
    
    def solve_control_optimization(self, state: np.ndarray, error_state: np.ndarray, 
                                    track_point_xdot: np.ndarray, state_xdot: np.ndarray) -> Tuple[float]:
        """
        Solves for the desired safe velocity and steering angle for the given trackpoint.

        Args:
            state (np.ndarray): The current state of the car in the form [x, y, theta]
            error_state (np.ndarray): The error state of the car in the form [x, y]
            track_point_xdot (np.ndarray): The x_dot vector of the trackpoint in the form [x_dot, y_dot, theta_dot]
            state_xdot (np.ndarray): The x_dot vector of the current state queried from the n_ode in the form [x_dot, y_dot, theta_dot]
        Returns:
            v (float): The desired safe velocity
            delta (float): The desired safe steering angle
        """
        # decision variables
        u: MX = casadi.MX.sym('u', 3, 1)
        control: MX = casadi.MX.sym('control', 2, 1)
        # slack variable
        epsilon: MX = casadi.MX.sym('epsilon') 
        
        Q: DM = casadi.diag([1/self.v_max, 1/self.v_max, 1/self.delta_max])
        cost = casadi.mtimes([u.T, Q, u]) + self.lambda_ * epsilon

        constraints: List[MX] = []
        
        # input limit constriants
        constraints.append(control)
        actuation_lower_bound: np.ndarray = np.array([-self.v_max, -self.delta_max])
        actuation_upper_bound: np.ndarray = np.array([self.v_max, self.delta_max])

        # CLF and CBF constraint
        constraints.append(casadi.dot(self.control_lyapunov_function_gradient_2d(error_state), state_xdot - track_point_xdot) - epsilon)
        constraints.append(casadi.dot(self.control_boundary_function_gradient_2d(state), state_xdot + u))
        clf_cbf_lower_bound: np.ndarray = np.array([-casadi.inf,
                                                    -self.gamma*self.control_boundary_function_2d(state)])
        clf_cbf_upper_bound: np.ndarray = np.array([-self.alpha*self.control_lyapunov_function_2d(error_state),
                                                    casadi.inf])

        # Dynamics Constraint
        constraints.append(state_xdot[0] + u[0] - control[0]*np.cos(state[2]))
        constraints.append(state_xdot[1] + u[1] - control[0]*np.sin(state[2]))
        constraints.append(state_xdot[2] + u[2] - (control[0]/self.L)*casadi.tan(control[1]))
        dynamics_lower_bound: np.ndarray = np.zeros(3)
        dynamics_upper_bound: np.ndarray = np.zeros(3)
        
        nlp = {
            'x': casadi.vertcat(casadi.reshape(u, -1, 1), casadi.reshape(control, -1, 1), epsilon),
            'f': cost,                  
            'g': casadi.vertcat(*constraints)  
        }
        
        solver: Function = casadi.nlpsol('solver','ipopt', nlp)
        lower_bound = casadi.vertcat(actuation_lower_bound, clf_cbf_lower_bound, dynamics_lower_bound)
        upper_bound = casadi.vertcat(actuation_upper_bound, clf_cbf_upper_bound, dynamics_upper_bound)
        x_guess: np.ndarray = np.zeros((6, 1))
        sol = solver(lbg=lower_bound, ubg=upper_bound, x0=casadi.vertcat(x_guess))
        x_opt = sol['x'].full().flatten()

        u_opt: np.ndarray = x_opt[0:3]
        control_opt: np.ndarray = x_opt[3:5]
        epsilon_opt: float = x_opt[5]

        return u_opt, control_opt, epsilon_opt
    
    def control_boundary_function_2d(self, state: np.ndarray) -> float:
        """
        Computes the control barrier function.
        
        Args:
            state (np.ndarray): The current state of the car in the form [x, y, theta]
        Returns:
            b_x (np.ndarray): The control barrier function
        """
        b_x: float = 1 - ((state[0][0] - self.ellipse_center[0]) / self.a)**2 - ((state[1][0] - self.ellipse_center[1]) / self.b)**2
        return b_x

    def control_boundary_function_gradient_2d(self, state: np.ndarray) -> casadi.DM:
        """
        Computes the gradient of the control barrier function.

        Args:
            state (np.ndarray): The current state of the car in the form [x, y, theta]
        Returns:
            db_dx (casadi.DM): The gradient of the control barrier function
        """
        db_dx: casadi.DM = casadi.DM(3, 1)
        db_dx[0] = -2 * (state[0] - self.ellipse_center[0]) / (self.a**2)
        db_dx[1] = -2 * (state[1] - self.ellipse_center[1]) / (self.b**2)
        db_dx[2] = 0
        return db_dx

    def control_lyapunov_function_2d(self, error_state: np.ndarray) -> float:
        """
        Computes the control lyapunov function.

        Args:
            error_state (np.ndarray): The error state of the car in the form [x, y]
        Returns:
            V (float): The control lyapunov function
        """
        V = (error_state[0][0]**2 + error_state[1][0]**2)**0.5
        return V

    def control_lyapunov_function_gradient_2d(self, error_state: np.ndarray) -> casadi.DM:
        """
        Computes the gradient of the control lyapunov function.

        Args:
            error_state (np.ndarray): The error state of the car in the form [x, y]
        Returns:
            dV_dx (casadi.DM): The gradient of the control lyapunov function
        """
        dV_dx: casadi.DM = casadi.DM(3, 1)
        dV_dx[0] = 0.5 * (error_state[0]**2 + error_state[1]**2)**(-0.5) * 2 * error_state[0]
        dV_dx[1] = 0.5 * (error_state[0]**2 + error_state[1]**2)**(-0.5) * 2 * error_state[1]
        dV_dx[2] = 0
        return dV_dx
    
    def get_yaw_from_quaternion(self, q: Quaternion) -> float:
        """
        Converts a quaternion to yaw angle.
        
        Args:
            q (Quaternion): The quaternion to convert
        Returns:
            yaw (float): The yaw angle in radians
        """
        siny_cosp: float = 2 * (q.w * q.z + q.x * q.y)
        cosy_cosp: float = 1 - 2 * (q.y * q.y + q.z * q.z)
        return math.atan2(siny_cosp, cosy_cosp)

def main(args=None):
    rclpy.init(args=args)
    node: NeuralODEController = NeuralODEController()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

