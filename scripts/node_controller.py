#!/usr/bin/env python3

import sys
import os
import rclpy
import torch
import math
import casadi
import numpy as np

from casadi import MX, DM, SX

from rclpy.node import Node
from rclpy.publisher import Publisher
from rclpy.subscription import Subscription

from geometry_msgs.msg import Quaternion, Vector3
from ackermann_msgs.msg import AckermannDrive
from mocap4r2_msgs.msg import RigidBodies

from reactive_car.srv import GenerateNominalTrajectory
from utils.n_ode import N_ODE
from typing import Tuple, List


class NeuralODEController(Node):
    def __init__(self) -> None:
        super().__init__('Neural_ODE_Controller')
        self.srv = self.create_service(GenerateNominalTrajectory,'generate_nominal_trajectory',self.generate_nominal_trajectory_callback)
        self.vicon_subscriber: Subscription = self.create_subscription(RigidBodies, '/odom_topic', self.odom_callback, 10)
        self.ackermann_publisher: Publisher = self.create_publisher(AckermannDrive, '/ackermann_cmd', 10)

        self.latest_odom: RigidBodies | None = None
        self.nominal_trajectory: np.ndarray[np.float32] | None = None

        # load the neural ODE model here
        self.model: N_ODE = N_ODE()
        self.model.load_state_dict(torch.load('model_weights.pth'))
        self.model.eval()

        # car parameters
        self.L: float = 1.0
        self.v_max: float = 2.0
        self.delta_max: float = math.pi / 4

        # control barrier function parameters
        self.a: float = 3.5
        self.b: float = 2.5
        self.gamma: float = 1.0
        self.ellipse_center: Tuple[float, float] = (0.5, 0.0)

        # control lyapunov function parameters
        self.alpha: float = 1.0
        self.lookahead_index: int = 5

        # trajectory rollout parameters
        self.dt: float = 0.05
        self.rollout_length: int = 100
        self.rollout_state_history: np.ndarray[np.float32] = np.zeros((3, self.rollout_length), dtype=np.float32)

        self.get_logger().info('Initialized NODE Controller')

    def generate_nominal_trajectory_callback(self, request, response):
        """
        Callback function for the GenerateNominalTrajectory service. Generates a nominal trajectory 
        using the neural ODE model.
        """

        state_0: np.ndarray[np.float32] = np.array([[self.latest_odom.rigidbodies.pose.position.x],
                                                    [self.latest_odom.rigidbodies.pose.position.y],
                                                    [self.get_yaw_from_quaternion(self.latest_odom.rigidbodies.pose.orientation)]])
        self.get_logger().info(f'Generating nominal trajectory with initial state: {state_0}')
        self.rollout_nominal_trajectory(state_0)

        response.success = True
        response.message = "Generated Nominal Trajectory"
        self.get_logger().info('Service called: returning nominal trajectory generation success.')
        return response
    
    def rollout_nominal_trajectory(self, state_0: np.ndarray[np.float32]) -> None:
        """
        Computes the nominal trajectory using the neural ODE model.
        
        Args:
            state_0 (np.ndarray): The initial state of the car in the form [x, y, theta]
        Returns: 
            None
        """
        current_state: torch.Tensor = torch.tensor(state_0, dtype=torch.float32).unsqueeze(0)
        for i in range(self.rollout_length):
            with torch.no_grad():
                n_ode_output = self.model(current_state)
            x_dot: np.ndarray[np.float32] = self.map_n_ode_to_x_dot(n_ode_output)
            current_state = current_state + x_dot * self.dt
            self.rollout_state_history[:, i] = current_state.squeeze().numpy()

    def odom_callback(self, msg: RigidBodies) -> None:
        """
        Callback function for the VICON odometry subscriber. Publishes the safe control command to the car.
        
        Args:
            msg (RigidBodies): The VICON odometry message
        Returns:
            None
        """
        self.latest_odom = msg

        if self.nominal_trajectory is None:
            self.get_logger().warn('Nominal trajectory not set. Skipping control.')
            return
        
        # extract orientation and position from the VICON ROS2 message
        orientation: Quaternion = msg.rigidbodies.pose.orientation
        position: Vector3 = msg.rigidbodies.pose.position
        yaw: float = self.get_yaw_from_quaternion(orientation)

        # create the state tensor and query the Neural ODE
        state = np.array([position.x, position.y, yaw])  

        # compute the safe control
        v_safe, delta_safe = self.compute_safe_control(state) 
        
        # publish the safe velocitya and steering angle
        ackermann_msg: AckermannDrive = AckermannDrive()
        ackermann_msg.speed = v_safe
        ackermann_msg.steering_angle = delta_safe
        self.ackermann_publisher.publish(ackermann_msg)    
    
    def map_n_ode_to_x_dot(self, n_ode_output: torch.Tensor) -> np.ndarray[np.float32]:
        """
        Converts the Neural ODE output to x_dot using the car dynamics model.

        Args:
            n_ode_output (torch.Tensor): The output of the Neural ODE model
        Returns:
            x_dot (np.ndarray): The x_dot vector in the form [x_dot, y_dot, theta_dot]
        """
        v: float = n_ode_output[0][0].item()
        delta: float = n_ode_output[0][1].item()
        x_dot: np.ndarray[np.float32] = np.array([v * math.cos(delta), v * math.sin(delta), v/self.L * math.tan(delta)])
        return x_dot

    def compute_safe_control(self, state: np.ndarray[np.float32]) -> Tuple[float]:
        """
        Gets candidate trackpoints and chooses the smallest residual control.

        Args:
            state (np.ndarray): The current state of the car in the form [x, y, theta]
        Returns:
            v_safe (float): The safe velocity
            delta_safe (float): The safe steering angle
        """
        # determine the safe control input by scanning across the next self.lookahead_index points on the track
        track_points: np.ndarray[np.float32] = self.calculate_track_points_2d(state)

        controls: List[Tuple[float]] = []
        for i in range(self.lookahead_index):
            track_point = track_points[:, i]
            track_point_velocity: torch.Tensor = self.model(torch.tensor(track_point, dtype=torch.float32).unsqueeze(0))
            track_point_xdot: np.ndarray[np.float32] = self.map_n_ode_to_x_dot(track_point_velocity)
            error_state: np.ndarray[np.float32] = state - track_point
            residual_control: Tuple[float] = self.solve_control_optimization(state, error_state, track_point_xdot)
            controls.append(residual_control)

        # choose the smallest residual control
        v_safe, delta_safe = min(controls, key=lambda x: abs(x[0])/self.v_max + abs(x[1])/self.delta_max)

        return v_safe, delta_safe
    
    def calculate_track_points_2d(self, state: np.ndarray[np.float32]) -> np.ndarray[np.float32]:
        """
        Computes the next self.lookahead_index points on the track.
        
        Args:
            state (np.ndarray): The current state of the car in the form [x, y, theta]
        Returns:
            track_points (np.ndarray): The next self.lookahead_index points on the track
        """
        closest_index: int = np.argmin(np.linalg.norm(self.nominal_trajectory - state, axis=0))
        track_points: np.ndarray[np.float32] = np.zeros((2, self.lookahead_index))
        for i in range(self.lookahead_index):
            track_points[:, i] = self.nominal_trajectory[:, (closest_index + i) % self.nominal_trajectory.shape[1]]
        return track_points
    
    def solve_control_optimization(self, state: np.ndarray[np.float32], error_state: np.ndarray[np.float32], 
                                    track_point_xdot: np.ndarray[np.float32]) -> Tuple[float]:
        x: MX = casadi.MX.sym('x')
        y: MX = casadi.MX.sym('y')
        theta: MX = casadi.MX.sym('theta')

        lambda_: float = 1.0  # Weight TODO: Tune
        epsilon: MX = casadi.MX.sym('epsilon')  # Slack variable

        v: MX = casadi.MX.sym('v')
        delta: MX = casadi.MX.sym('delta')
        delta_upper_bound: float = casadi.pi / 4  # 45 degrees
        delta_lower_bound: float = -casadi.pi / 4

        cost: MX = v**2 + delta**2 + lambda_*epsilon

        constraints: List[MX] = []
        constraint_lower_bound: List[float] = []
        constraint_upper_bound: List[float] = []

        constraints.append(delta)
        constraint_lower_bound.append(delta_lower_bound)
        constraint_upper_bound.append(delta_upper_bound)

        casadi.fabs
        
        return v, delta
    
    def control_boundary_function_2d(self, state: np.ndarray[np.float32]) -> np.ndarray[np.float32]:
        """
        Computes the control barrier function.
        
        Args:
            state (np.ndarray): The current state of the car in the form [x, y, theta]
        Returns:
            b_x (np.ndarray): The control barrier function
        """
        b_x: np.ndarray[np.float32] = 1 - np.array([[
            ((state[0] - self.ellipse_center[0]) / self.a)**2,
            ((state[1] - self.ellipse_center[1]) / self.b)**2]])
        return b_x

    def control_boundary_function_gradient_2d(self, state: np.ndarray[np.float32]) -> casadi.DM:
        """
        Computes the gradient of the control barrier function.

        Args:
            state (np.ndarray): The current state of the car in the form [x, y, theta]
        Returns:
            db_dx (casadi.DM): The gradient of the control barrier function
        """
        db_dx: casadi.DM = casadi.DM(2, 1)
        db_dx[0] = -2 * (state[0] - self.ellipse_center[0]) / (self.a**2)
        db_dx[1] = -2 * (state[1] - self.ellipse_center[1]) / (self.b**2)
        return db_dx

    def control_lyapunov_function_2d(self, error_state: np.ndarray[np.float32]):
        """
        Computes the control lyapunov function.

        Args:
            error_state (np.ndarray): The error state of the car in the form [x, y]
        Returns:
            V (float): The control lyapunov function
        """
        V = (error_state[0]**2 + error_state[1]**2)**0.5
        return V

    def control_lyapunov_function_gradient_2d(self, error_state: np.ndarray[np.float32]) -> casadi.DM:
        """
        Computes the gradient of the control lyapunov function.

        Args:
            error_state (np.ndarray): The error state of the car in the form [x, y]
        Returns:
            dV_dx (casadi.DM): The gradient of the control lyapunov function
        """
        dV_dx: casadi.DM = casadi.DM(2, 1)
        dV_dx[0] = 0.5 * (error_state[0]**2 + error_state[1]**2)**(-0.5) * 2 * error_state[0]
        dV_dx[1] = 0.5 * (error_state[0]**2 + error_state[1]**2)**(-0.5) * 2 * error_state[1]
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

