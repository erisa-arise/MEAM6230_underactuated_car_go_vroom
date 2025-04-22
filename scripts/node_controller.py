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

        # control barrier function parameters
        self.a: float = 3.5
        self.b: float = 2.5
        self.gamma: float = 1.0
        self.ellipse_center: Tuple[float, float] = (0.5, 0.0)

        # control lyapunov function parameters
        self.nominal_trajectory: np.ndarray[np.float32] | None = None
        self.alpha: float = 1.0

        # trajectory rollout parameters
        self.dt: float = 0.05
        self.rollout_length: int = 100
        self.rollout_state_history: np.ndarray[np.float32] = np.zeros((3, self.rollout_length), dtype=np.float32)

        self.get_logger().info('Initialized NODE Controller')

    def generate_nominal_trajectory_callback(self, request, response):
        response.success = True
        response.message = "Generated Nominal Trajectory"
        self.get_logger().info('Service called: returning nominal trajectory generation success.')
        return response

    def odom_callback(self, msg: RigidBodies):
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
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            n_ode_output = self.model(state_tensor) # Waiting on josh's n_ode input script  

        # convert Neural ODE output to x_dot and then compute the safe control
        x_dot: np.ndarray[np.float32] = self.map_n_ode_to_x_dot(n_ode_output)
        v_safe, delta_safe = self.compute_safe_control(x_dot) 
        
        # publish the safe velocitya and steering angle
        ackermann_msg: AckermannDrive = AckermannDrive()
        ackermann_msg.speed = v_safe
        ackermann_msg.steering_angle = delta_safe
        self.ackermann_publisher.publish(ackermann_msg)    

    def get_yaw_from_quaternion(self, q: Quaternion) -> float:
        # Convert quaternion to yaw
        siny_cosp: float = 2 * (q.w * q.z + q.x * q.y)
        cosy_cosp: float = 1 - 2 * (q.y * q.y + q.z * q.z)
        return math.atan2(siny_cosp, cosy_cosp)
    
    def map_n_ode_to_x_dot(self, n_ode_output: torch.Tensor) -> np.ndarray[np.float32]:
        # Converts using the car dynamics model 
        v: float = n_ode_output[0][0].item()
        delta: float = n_ode_output[0][1].item()
        x_dot: np.ndarray[np.float32] = np.array([v * math.cos(delta), v * math.sin(delta), v/self.L * math.tan(delta)])
        return x_dot

    def compute_safe_control(self, state: np.ndarray[np.float32], x_dot: np.ndarray[np.float32]) -> Tuple[float]:
        # Computes the safe control using the control barrier function, control lyapunov function, and solves a QP using Casadi
        x: MX = casadi.MX.sym('x')
        y: MX = casadi.MX.sym('y')
        theta: MX = casadi.MX.sym('theta')

        gamma: float = 1.0  # Weight TODO: Tune
        epsilon: MX = casadi.MX.sym('epsilon')  # Slack variable

        v: MX = casadi.MX.sym('v')
        delta: MX = casadi.MX.sym('delta')
        delta_upper_bound: float = casadi.pi / 4  # 45 degrees
        delta_lower_bound: float = -casadi.pi / 4

        cost: MX = v**2 + delta**2 + gamma*epsilon

        constraints: List[MX] = []
        constraint_lower_bound: List[float] = []
        constraint_upper_bound: List[float] = []

        constraints.append(delta)
        constraint_lower_bound.append(delta_lower_bound)
        constraint_upper_bound.append(delta_upper_bound)

        casadi.fabs
        
        return v, delta
    
    def control_boundary_function_2d(self, state: np.ndarray[np.float32]) -> np.ndarray[np.float32]:
        # defines the control barrier function for an elliptical boundary around the track
        b_x: np.ndarray[np.float32] = 1 - np.array([[
            ((state[0] - self.ellipse_center[0]) / self.a)**2,
            ((state[1] - self.ellipse_center[1]) / self.b)**2]])
        return b_x

    def control_boundary_function_gradient_2d(self, state: np.ndarray[np.float32]) -> casadi.DM:
        # defines the gradient for an elliptical boundary around the track
        db_dx: casadi.DM = casadi.DM(2, 1)
        db_dx[0] = -2 * (state[0] - self.ellipse_center[0]) / (self.a**2)
        db_dx[1] = -2 * (state[1] - self.ellipse_center[1]) / (self.b**2)
        return db_dx

    def control_lyapunov_function_2d(self, state: np.ndarray[np.float32]):
        # V(x)
        pass

    def control_lyapunov_function_gradient_2d(self, state: np.ndarray[np.float32]):
        # V'(x)
        pass

    def rollout_nominal_trajectory(self, state_0: np.ndarray[np.float32]):
        # computes the nominal trajectory using the neural ODE
        current_state: torch.Tensor = torch.tensor(state_0, dtype=torch.float32).unsqueeze(0)
        for i in range(self.rollout_length):
            with torch.no_grad():
                n_ode_output = self.model(current_state)
            x_dot: np.ndarray[np.float32] = self.map_n_ode_to_x_dot(n_ode_output)
            current_state = current_state + x_dot * self.dt
            self.rollout_state_history[:, i] = current_state.squeeze().numpy()

    def calculate_track_point_2d(self, state: np.ndarray[np.float32]):
        pass

def main(args=None):
    rclpy.init(args=args)
    node: NeuralODEController = NeuralODEController()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

